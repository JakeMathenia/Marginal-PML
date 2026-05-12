"""
Frequency Model — XGBoost Classifier for Annual Claim Probability

Predicts P(claim | property features) — the annual probability that a given
property generates a flood insurance claim. This becomes the RATE column in
the generated Event Loss Table.

Key design decisions:
    - Binary classification: claim (1) vs no-claim (0)
    - Class imbalance: ~10-15% positive rate; handled with scale_pos_weight
    - Calibration: Platt scaling applied post-training for reliable probabilities
    - Output: calibrated probability in [0, 1] = annual claim frequency

Usage:
    from frequency_model import FrequencyModel

    model = FrequencyModel()
    model.fit(X_train, y_train)

    # Predict annual claim probabilities
    annual_probs = model.predict_proba(X_new)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, log_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# sklearn 1.6+ removed cv='prefit' in favour of FrozenEstimator
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed. Run: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters for binary classification
DEFAULT_PARAMS = {
    'n_estimators':      1000,
    'max_depth':         5,
    'learning_rate':     0.05,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'min_child_weight':  20,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'objective':         'binary:logistic',
    'eval_metric':       'auc',
    'random_state':      42,
    'n_jobs':            4,
    'use_label_encoder': False,
    'early_stopping_rounds': 50,
}


class FrequencyModel:
    """
    XGBoost binary classifier for annual flood claim frequency.

    The output probability represents the expected annual claim rate for a
    given property. This maps directly to the RATE field in an Event Loss Table.

    Calibration:
        Raw XGBoost probabilities may be poorly calibrated for rare events.
        Platt scaling (sigmoid calibration) is applied by default to ensure
        the output probabilities are reliable estimates of true frequencies.

    Args:
        params:     XGBoost hyperparameters (None = use defaults)
        calibrate:  If True, apply Platt scaling calibration (recommended)
        pos_weight: Weight multiplier for positive class (None = auto-compute)
    """

    # Maximum auto-computed scale_pos_weight.
    # Values above this distort calibration; tune manually if needed.
    MAX_AUTO_SPW = 10.0

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        calibrate: bool = True,
        pos_weight: Optional[float] = None,
        max_pos_weight: float = 10.0,
    ):
        if not HAS_XGB:
            raise ImportError("xgboost is required. Run: pip install xgboost")

        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.calibrate = calibrate
        self.pos_weight = pos_weight
        self.max_pos_weight = max_pos_weight

        self._model = None              # raw XGBoost model
        self._calibrated_model = None   # calibrated wrapper (if enabled)
        self._feature_names: Optional[List[str]] = None
        self._positive_rate: float = 0.0
        self._shap_explainer = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_fraction: float = 0.15,
        cal_fraction: float = 0.15,
        verbose: bool = True,
    ) -> 'FrequencyModel':
        """
        Fit the frequency model with a proper 3-way split.

        Split strategy (avoids calibration double-dipping):
            1. Train set      (~70%) — XGBoost training
            2. Eval set        (~15%) — early stopping monitor
            3. Calibration set (~15%) — Platt scaling (never seen during training)

        Args:
            X:              Feature matrix (output of FeaturePipeline)
            y:              Binary target (1=claim occurred, 0=no claim)
            eval_fraction:  Fraction for early stopping eval set
            cal_fraction:   Fraction for calibration holdout (separate from eval)
            verbose:        Print training progress

        Returns:
            self (fitted model)
        """
        y = y.fillna(0).astype(int)
        self._feature_names = list(X.columns)
        self._positive_rate = float(y.mean())

        logger.info(
            f"Training frequency model on {len(X):,} records | "
            f"Positive rate: {self._positive_rate:.1%}"
        )

        # Auto-compute scale_pos_weight if not provided, with cap
        if self.pos_weight is None:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            raw_spw = n_neg / max(n_pos, 1)
            spw = min(raw_spw, self.max_pos_weight)
            if raw_spw > self.max_pos_weight:
                logger.warning(
                    f"scale_pos_weight capped: {raw_spw:.1f} → {spw:.1f} "
                    f"(max_pos_weight={self.max_pos_weight}). "
                    f"Excessive SPW distorts calibration."
                )
        else:
            spw = self.pos_weight

        logger.info(f"scale_pos_weight: {spw:.2f}")

        # --- 3-way split: train / eval (early stopping) / calibration ---
        holdout_fraction = eval_fraction + cal_fraction
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=holdout_fraction,
            stratify=y,
            random_state=42,
        )
        # Split holdout into eval and calibration sets
        cal_share = cal_fraction / holdout_fraction  # proportion of holdout for calibration
        X_eval, X_cal, y_eval, y_cal = train_test_split(
            X_holdout, y_holdout,
            test_size=cal_share,
            stratify=y_holdout,
            random_state=42,
        )

        logger.info(
            f"3-way split: train={len(X_train):,} | "
            f"eval={len(X_eval):,} | cal={len(X_cal):,}"
        )

        # XGBoost 2.x: early_stopping_rounds is a constructor arg, not fit() arg
        params = {**self.params, 'scale_pos_weight': spw}

        self._model = xgb.XGBClassifier(**params)
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=(100 if verbose else False),
        )

        # Evaluate on eval set (pre-calibration)
        raw_probs_eval = self._model.predict_proba(X_eval)[:, 1]
        auc = roc_auc_score(y_eval, raw_probs_eval)
        ap  = average_precision_score(y_eval, raw_probs_eval)
        logger.info(f"Pre-calibration: AUC={auc:.4f} | AP={ap:.4f} | Best iter={self._model.best_iteration}")

        # Calibrate with Platt scaling on SEPARATE calibration set
        if self.calibrate:
            logger.info("Applying Platt scaling on held-out calibration set...")
            if _HAS_FROZEN:
                # sklearn 1.6+: wrap the prefit estimator in FrozenEstimator
                self._calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(self._model),
                    method='sigmoid',
                    cv=2,
                )
            else:
                # sklearn < 1.6: legacy prefit API
                self._calibrated_model = CalibratedClassifierCV(
                    estimator=self._model,
                    method='sigmoid',
                    cv='prefit',
                )
            # Fit calibration on data the model has NEVER seen
            self._calibrated_model.fit(X_cal, y_cal)

            # Report calibration improvement on the cal set
            raw_probs_cal = self._model.predict_proba(X_cal)[:, 1]
            cal_probs = self._calibrated_model.predict_proba(X_cal)[:, 1]
            raw_brier = brier_score_loss(y_cal, raw_probs_cal)
            cal_brier = brier_score_loss(y_cal, cal_probs)
            logger.info(
                f"Post-calibration: Brier={cal_brier:.4f} (raw={raw_brier:.4f}) | "
                f"AUC={roc_auc_score(y_cal, cal_probs):.4f}"
            )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: pd.DataFrame,
        use_calibrated: bool = True,
    ) -> np.ndarray:
        """
        Predict annual claim probability for each property.

        Args:
            X:              Feature matrix
            use_calibrated: Use calibrated model if available (recommended)

        Returns:
            1-D array of probabilities in [0, 1]  (= annual claim rate)
        """
        self._check_fitted()
        X_aligned = self._align_features(X)

        if use_calibrated and self._calibrated_model is not None:
            return self._calibrated_model.predict_proba(X_aligned)[:, 1]
        return self._model.predict_proba(X_aligned)[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Binary class prediction.

        Args:
            X:         Feature matrix
            threshold: Classification threshold (default 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Calibration Assessment
    # ------------------------------------------------------------------

    def calibration_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compute calibration curve data for reliability diagram.

        Args:
            X:      Feature matrix
            y:      True binary labels
            n_bins: Number of probability bins

        Returns:
            DataFrame with columns: mean_predicted_prob, fraction_of_positives
        """
        self._check_fitted()
        probs = self.predict_proba(X)

        fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins)

        return pd.DataFrame({
            'mean_predicted_prob': mean_pred,
            'fraction_of_positives': fraction_pos,
        })

    # ------------------------------------------------------------------
    # SHAP Explainability (on base uncalibrated model)
    # ------------------------------------------------------------------

    def shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for the base classifier.

        Args:
            X:           Feature matrix to explain
            max_samples: Cap sample size for speed

        Returns:
            SHAP values array (n_samples × n_features)
        """
        if not HAS_SHAP:
            raise ImportError("shap is required. Run: pip install shap")

        self._check_fitted()

        X_sample = X.sample(min(max_samples, len(X)), random_state=42)
        X_aligned = self._align_features(X_sample)

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self._model)

        return self._shap_explainer.shap_values(X_aligned)

    def shap_summary(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> pd.DataFrame:
        """Return mean absolute SHAP values sorted by importance."""
        shap_vals = self.shap_values(X, max_samples=max_samples)
        mean_abs = np.abs(shap_vals).mean(axis=0)

        return pd.DataFrame({
            'feature': self._feature_names,
            'mean_abs_shap': mean_abs,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate frequency model performance on a hold-out test set.

        Returns:
            Dict with AUC, AP, Brier score, log-loss, and classification report
        """
        self._check_fitted()
        y_test = y_test.fillna(0).astype(int)
        probs = self.predict_proba(X_test)
        preds = self.predict(X_test)

        metrics = {
            'auc':           float(roc_auc_score(y_test, probs)),
            'avg_precision': float(average_precision_score(y_test, probs)),
            'brier_score':   float(brier_score_loss(y_test, probs)),
            'log_loss':      float(log_loss(y_test, probs)),
            'n_test':        int(len(y_test)),
            'positive_rate': float(y_test.mean()),
        }

        logger.info(
            f"Frequency evaluation | "
            f"AUC={metrics['auc']:.4f} | "
            f"AP={metrics['avg_precision']:.4f} | "
            f"Brier={metrics['brier_score']:.4f}"
        )

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Stratified K-fold cross-validation. Returns per-fold AUC/AP metrics."""
        if not HAS_XGB:
            raise ImportError("xgboost required")

        y = y.fillna(0).astype(int)
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        spw = n_neg / max(n_pos, 1)

        params = {**self.params, 'scale_pos_weight': spw}

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = xgb.XGBClassifier(**params)
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            probs = fold_model.predict_proba(X_val)[:, 1]
            fold_metrics.append({
                'fold': fold + 1,
                'auc':           float(roc_auc_score(y_val, probs)),
                'avg_precision': float(average_precision_score(y_val, probs)),
                'brier_score':   float(brier_score_loss(y_val, probs)),
            })

            if verbose:
                logger.info(f"  Fold {fold+1}: AUC={fold_metrics[-1]['auc']:.4f}, "
                            f"AP={fold_metrics[-1]['avg_precision']:.4f}")

        results = pd.DataFrame(fold_metrics)
        logger.info(f"\nCV Summary: AUC={results['auc'].mean():.4f}±{results['auc'].std():.4f}, "
                    f"AP={results['avg_precision'].mean():.4f}±{results['avg_precision'].std():.4f}")
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model':             self._model,
            'calibrated_model':  self._calibrated_model,
            'feature_names':     self._feature_names,
            'positive_rate':     self._positive_rate,
            'params':            self.params,
            'calibrate':         self.calibrate,
        }, path)
        logger.info(f"Frequency model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'FrequencyModel':
        """Load a previously saved model."""
        data = joblib.load(path)
        instance = cls(params=data['params'], calibrate=data.get('calibrate', True))
        instance._model            = data['model']
        instance._calibrated_model = data.get('calibrated_model')
        instance._feature_names    = data['feature_names']
        instance._positive_rate    = data.get('positive_rate', 0.0)
        instance._is_fitted        = True
        logger.info(f"Frequency model loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # Properties & helpers
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> pd.Series:
        """XGBoost gain-based feature importances."""
        self._check_fitted()
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    @property
    def baseline_rate(self) -> float:
        """Training set positive rate (baseline annual claim frequency)."""
        return self._positive_rate

    def _check_fitted(self):
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has the same columns as training data."""
        if self._feature_names is None:
            return X
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = 0.0
            logger.warning(f"Added missing features with 0: {missing}")
        return X[self._feature_names]


# =============================================================================
# MAIN (quick smoke test)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    np.random.seed(42)
    n = 5000

    from feature_engineering import FREQUENCY_FEATURES
    X = pd.DataFrame(np.random.rand(n, len(FREQUENCY_FEATURES)), columns=FREQUENCY_FEATURES)

    # Simulate ~12% positive rate (realistic for NFIP)
    # Make flood_zone_risk_score strongly predictive
    log_odds = (
        -3.0
        + 2.5 * X['flood_zone_risk_score']
        + 1.5 * X['is_coastal_high_risk']
        + 0.8 * X['hurricane_risk_score']
        - 0.5 * X['has_elevation_cert']
        + np.random.normal(0, 0.5, n)
    )
    probs_true = 1 / (1 + np.exp(-log_odds))
    y = pd.Series((np.random.random(n) < probs_true).astype(int))

    print(f"Positive rate: {y.mean():.1%}")

    model = FrequencyModel()
    model.fit(X, y, verbose=False)

    preds = model.predict_proba(X.head(10))
    print(f"\nSample predicted frequencies: {preds.round(4)}")
    print(f"Feature importances (top 5):\n{model.feature_importances.head()}")

    metrics = model.evaluate(X, y)
    print(f"\nMetrics: AUC={metrics['auc']:.4f}, AP={metrics['avg_precision']:.4f}")

    print("\n✅ Frequency model working")
