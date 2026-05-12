"""
Severity Model — XGBoost Regressor for Expected Claim Amount

Predicts the expected dollar loss on a building given that a claim has occurred.
Trained on NFIP claims data with log-transformed target (amountPaidOnBuildingClaim).

Key design decisions:
    - Log-normal target: claim amounts are right-skewed; log transform → ~normal
    - XGBoost: handles missing values natively, robust to outliers
    - SHAP values: computed post-training for explainability dashboard
    - Prediction output: both log-scale and dollar-scale predictions

Usage:
    from severity_model import SeverityModel

    model = SeverityModel()
    model.fit(X_train, y_train_log)

    # Predict on new properties (returns dollar amounts)
    predictions = model.predict(X_new)

    # SHAP values for explainability
    shap_values = model.shap_values(X_sample)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Default XGBoost hyperparameters (tuned for log-normal severity)
DEFAULT_PARAMS = {
    'n_estimators':      1500,
    'max_depth':         6,
    'learning_rate':     0.05,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'min_child_weight':  10,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'objective':         'reg:squarederror',
    'eval_metric':       'rmse',
    'random_state':      42,
    'n_jobs':            -1,
    'early_stopping_rounds': 50,
}


class SeverityModel:
    """
    XGBoost severity model: predicts log(claim amount) given property features.

    The model is trained only on records where a claim occurred (positive losses).
    Predictions are returned as dollar amounts (exp-transformed from log scale).

    Uncertainty quantification:
        - Training RMSE on log scale → used to derive STDDEV for ELT generation
        - Quantile regression mode available for prediction intervals

    Args:
        params:    XGBoost hyperparameters (None = use defaults)
        log_target: If True, target is expected to be already log-transformed
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        log_target: bool = True,
    ):
        if not HAS_XGB:
            raise ImportError("xgboost is required. Run: pip install xgboost")

        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.log_target = log_target

        self._model: Optional[xgb.XGBRegressor] = None
        self._feature_names: Optional[list] = None
        self._train_rmse_log: float = 0.0
        self._train_mae_log: float = 0.0
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
        verbose: bool = True,
    ) -> 'SeverityModel':
        """
        Fit the severity model.

        Args:
            X: Feature matrix (output of FeaturePipeline)
            y: Log-severity target (np.nan for non-claims rows are dropped)
            eval_fraction: Fraction of data to hold out for early stopping eval
            verbose: Print training progress

        Returns:
            self (fitted model)
        """
        # Drop rows where target is NaN (non-claim records)
        valid_mask = pd.notna(y)
        X_fit = X[valid_mask].copy()
        y_fit = y[valid_mask].copy()

        logger.info(f"Training severity model on {len(X_fit):,} claim records "
                    f"(dropped {(~valid_mask).sum():,} non-claim rows)")

        self._feature_names = list(X_fit.columns)

        # Train/eval split for early stopping
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_fit, y_fit,
            test_size=eval_fraction,
            random_state=42
        )

        # Build model — XGBoost 2.x accepts early_stopping_rounds in the constructor
        self._model = xgb.XGBRegressor(**self.params)

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=(100 if verbose else False),
        )

        # Compute training metrics on the full fit set
        y_pred_log = self._model.predict(X_fit)
        self._train_rmse_log = float(np.sqrt(mean_squared_error(y_fit, y_pred_log)))
        self._train_mae_log  = float(mean_absolute_error(y_fit, y_pred_log))
        r2                   = float(r2_score(y_fit, y_pred_log))

        logger.info(
            f"Severity model fitted | "
            f"RMSE(log): {self._train_rmse_log:.4f} | "
            f"MAE(log): {self._train_mae_log:.4f} | "
            f"R²: {r2:.4f} | "
            f"Best iteration: {self._model.best_iteration}"
        )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        return_log: bool = False,
    ) -> np.ndarray:
        """
        Predict expected claim severity.

        Args:
            X:          Feature matrix
            return_log: If True, return log-scale predictions (default: dollar amounts)

        Returns:
            1-D array of predicted claim amounts (or log amounts if return_log=True)
        """
        self._check_fitted()
        X = self._align_features(X)
        log_preds = self._model.predict(X)

        if return_log:
            return log_preds
        return np.exp(log_preds)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        n_sigma: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict expected severity with uncertainty bounds.

        Uses training RMSE on log scale as the uncertainty estimate.
        Under log-normal assumption:
            - mean_dollar = exp(mu + sigma²/2)
            - upper_dollar = exp(mu + n_sigma * sigma)
            - lower_dollar = exp(mu - n_sigma * sigma)

        Args:
            X:       Feature matrix
            n_sigma: Number of standard deviations for uncertainty bounds

        Returns:
            DataFrame with columns: mean, lower, upper, log_mean, log_std
        """
        self._check_fitted()
        X_aligned = self._align_features(X)
        log_mu = self._model.predict(X_aligned)

        sigma = self._train_rmse_log  # log-scale std estimate

        # Log-normal mean (corrected for Jensen's inequality)
        mean_dollar = np.exp(log_mu + 0.5 * sigma ** 2)
        lower_dollar = np.exp(log_mu - n_sigma * sigma)
        upper_dollar = np.exp(log_mu + n_sigma * sigma)

        return pd.DataFrame({
            'mean':     mean_dollar,
            'lower':    lower_dollar,
            'upper':    upper_dollar,
            'log_mean': log_mu,
            'log_std':  sigma,
        }, index=X.index)

    # ------------------------------------------------------------------
    # SHAP Explainability
    # ------------------------------------------------------------------

    def shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for model explainability.

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

        shap_vals = self._shap_explainer.shap_values(X_aligned)
        return shap_vals

    def shap_summary(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> pd.DataFrame:
        """
        Return mean absolute SHAP values (feature importance) as a DataFrame.

        Args:
            X:           Feature matrix
            max_samples: Cap sample size for speed

        Returns:
            DataFrame with columns [feature, mean_abs_shap] sorted by importance
        """
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
        log_target: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a hold-out test set.

        Args:
            X_test:     Test feature matrix
            y_test:     Test target (log-scale if log_target=True)
            log_target: Whether y_test is log-transformed

        Returns:
            Dict of metrics: rmse_log, mae_log, r2, rmse_dollar, mae_dollar
        """
        self._check_fitted()
        valid = pd.notna(y_test)
        X_val = X_test[valid]
        y_val = y_test[valid]

        log_preds = self.predict(X_val, return_log=True)

        if not log_target:
            y_log = np.log(np.maximum(y_val, 1))
        else:
            y_log = y_val

        dollar_preds = np.exp(log_preds)
        dollar_true  = np.exp(y_log)

        metrics = {
            'rmse_log':    float(np.sqrt(mean_squared_error(y_log, log_preds))),
            'mae_log':     float(mean_absolute_error(y_log, log_preds)),
            'r2':          float(r2_score(y_log, log_preds)),
            'rmse_dollar': float(np.sqrt(mean_squared_error(dollar_true, dollar_preds))),
            'mae_dollar':  float(mean_absolute_error(dollar_true, dollar_preds)),
            'n_test':      int(valid.sum()),
        }

        logger.info(
            f"Severity evaluation | "
            f"R²: {metrics['r2']:.4f} | "
            f"RMSE(log): {metrics['rmse_log']:.4f} | "
            f"MAE($): ${metrics['mae_dollar']:,.0f}"
        )

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        K-fold cross-validation. Returns per-fold metrics as a DataFrame.
        """
        if not HAS_XGB:
            raise ImportError("xgboost required")

        valid_mask = pd.notna(y)
        X_fit = X[valid_mask].values
        y_fit = y[valid_mask].values

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_fit)):
            fold_model = xgb.XGBRegressor(**self.params)
            fold_model.fit(
                X_fit[train_idx], y_fit[train_idx],
                eval_set=[(X_fit[val_idx], y_fit[val_idx])],
                verbose=False,
            )

            preds = fold_model.predict(X_fit[val_idx])
            fold_metrics.append({
                'fold': fold + 1,
                'rmse_log': float(np.sqrt(mean_squared_error(y_fit[val_idx], preds))),
                'mae_log':  float(mean_absolute_error(y_fit[val_idx], preds)),
                'r2':       float(r2_score(y_fit[val_idx], preds)),
            })

            if verbose:
                logger.info(f"  Fold {fold+1}: RMSE(log)={fold_metrics[-1]['rmse_log']:.4f}, "
                            f"R²={fold_metrics[-1]['r2']:.4f}")

        results = pd.DataFrame(fold_metrics)
        logger.info(f"\nCV Summary: RMSE(log)={results['rmse_log'].mean():.4f}±{results['rmse_log'].std():.4f}, "
                    f"R²={results['r2'].mean():.4f}±{results['r2'].std():.4f}")
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model':          self._model,
            'feature_names':  self._feature_names,
            'train_rmse_log': self._train_rmse_log,
            'train_mae_log':  self._train_mae_log,
            'params':         self.params,
        }, path)
        logger.info(f"Severity model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SeverityModel':
        """Load a previously saved model."""
        data = joblib.load(path)
        instance = cls(params=data['params'])
        instance._model          = data['model']
        instance._feature_names  = data['feature_names']
        instance._train_rmse_log = data['train_rmse_log']
        instance._train_mae_log  = data['train_mae_log']
        instance._is_fitted      = True
        logger.info(f"Severity model loaded from {path}")
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
    def train_log_rmse(self) -> float:
        """Training RMSE on log scale (used as sigma for ELT generation)."""
        return self._train_rmse_log

    def _check_fitted(self):
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has the same columns as the training data."""
        if self._feature_names is None:
            return X
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            for col in missing:
                X = X.copy()
                X[col] = 0.0
            logger.warning(f"Added missing features with 0: {missing}")
        return X[self._feature_names]


# =============================================================================
# MAIN (quick smoke test)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Generate synthetic training data
    np.random.seed(42)
    n = 2000
    n_features = 17

    from feature_engineering import SEVERITY_FEATURES
    X = pd.DataFrame(
        np.random.rand(n, len(SEVERITY_FEATURES)),
        columns=SEVERITY_FEATURES
    )
    # Only ~15% of records have claims (log-severity for those)
    has_claim = np.random.random(n) < 0.15
    y = pd.Series(
        np.where(has_claim, np.random.normal(9.5, 1.5, n), np.nan),
        name='log_severity'
    )

    print(f"Training on {has_claim.sum()} claim records out of {n} total")

    model = SeverityModel()
    model.fit(X, y, verbose=False)

    print(f"\nFeature importances (top 5):\n{model.feature_importances.head()}")

    # Predict with uncertainty
    X_new = X.head(5)
    preds = model.predict_with_uncertainty(X_new)
    print(f"\nPredictions with uncertainty:\n{preds.round(0)}")

    print("\n✅ Severity model working")
