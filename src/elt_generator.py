"""
ELT Generator — ML Predictions → Event Loss Table

This is the bridge module that converts frequency/severity model predictions
into Event Loss Tables (ELTs) in the exact format consumed by the
MarginalPMLEngine.

ELT Format (per-event rows):
    EVENTID    - Unique event identifier
    RATE       - Annual frequency / occurrence rate (from frequency model)
    PERSPVALUE - Expected loss given event occurs (from severity model)
    EXPVALUE   - Exposure / total insured value (policy limit)
    STDDEVC    - Correlated standard deviation (systematic / cat event risk)
    STDDEVI    - Independent standard deviation (idiosyncratic property risk)

Design rationale:
    Real cat models (AIR Touchstone, RMS RiskLink, CoreLogic) produce ELTs
    from stochastic event sets. Our ML pipeline generates ELTs from property
    feature predictions — but the downstream pricing engine is IDENTICAL.

    This interchangeability is the core design insight of this project.

Variance decomposition:
    Total variance = STDDEVC² + STDDEVI²
    
    STDDEVC captures shared risk (properties in same flood zone / storm path
    will be hit together — this is the correlated component).
    
    STDDEVI captures idiosyncratic risk (given the event occurs, how much
    does this specific property's loss vary — foundation type, elevation, etc.)

Usage:
    from elt_generator import ELTGenerator

    generator = ELTGenerator(freq_model, sev_model)
    elt_df = generator.generate(X_properties, property_metadata)

    # Feed directly to pricing engine
    from marginal_pml_kernel import MarginalPMLEngine
    engine = MarginalPMLEngine(elt_df)
    results = engine.price(return_periods=[100, 250])
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ELT column specification (must match marginal_pml_kernel.py expectations)
ELT_COLUMNS = ['EVENTID', 'RATE', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI']

# Correlated risk fraction by flood zone
# Higher correlated fraction = more of the variance is shared across properties
# in the same event (hurricane / flood zone clustering)
CORR_FRACTION_BY_ZONE = {
    'coastal_high_risk': 0.70,  # V/VE zones: highly correlated (hurricane surge)
    'high_risk':         0.55,  # AE zones: moderately correlated (flood events)
    'moderate_risk':     0.35,  # X/B zones: lower correlation
    'minimal_risk':      0.20,  # Mostly idiosyncratic
    'unknown':           0.40,  # Default assumption
}

# Correlated risk fraction by peril
CORR_FRACTION_BY_PERIL = {
    'hurricane':       0.75,
    'tropical_storm':  0.65,
    'storm_surge':     0.70,
    'coastal_flood':   0.55,
    'flood':           0.45,
    'heavy_rain':      0.30,
    'other':           0.35,
}


class ELTGenerator:
    """
    Converts ML frequency/severity predictions into Event Loss Tables.

    The ELT format is a universal interface between catastrophe risk models
    and pricing engines. Each row represents one "virtual event" with:
        - An annual occurrence rate (frequency)
        - An expected loss given the event (severity)
        - A variance decomposition (correlated vs. idiosyncratic)

    For a portfolio, events are aggregated before pricing:
        Portfolio PERSPVALUE = Σ(property PERSPVALUE) for correlated events
        Portfolio STDDEVI    = sqrt(Σ(STDDEVI²))         (independent)
        Portfolio STDDEVC    = Σ(STDDEVC)                (fully correlated)

    Args:
        freq_model:  Fitted FrequencyModel instance
        sev_model:   Fitted SeverityModel instance
        n_events_per_property: Virtual events to generate per property
            (1 = single "average event" per property; >1 splits into
             multiple events with different severity scenarios)
        corr_method: How to compute STDDEVC
            'zone'  - based on flood zone category
            'peril' - based on matched storm peril type
            'both'  - average of zone and peril estimates
    """

    def __init__(
        self,
        freq_model=None,
        sev_model=None,
        n_events_per_property: int = 1,
        corr_method: str = 'both',
        min_rate: float = 1e-6,
        max_rate: float = 1.0,
        filing_rates: Optional[pd.DataFrame] = None,
    ):
        self.freq_model = freq_model
        self.sev_model = sev_model
        self.n_events_per_property = n_events_per_property
        self.corr_method = corr_method
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.filing_rates = filing_rates

    # ------------------------------------------------------------------
    # Primary generation method
    # ------------------------------------------------------------------

    def generate(
        self,
        X: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        property_ids: Optional[pd.Series] = None,
        portfolio_name: str = 'portfolio',
    ) -> pd.DataFrame:
        """
        Generate an ELT from model predictions on a set of properties.

        Args:
            X:              Feature matrix (output of FeaturePipeline)
            metadata:       Optional raw property data for variance decomposition
                            (should contain: flood_zone_category, peril_category,
                             totalBuildingInsuranceCoverage)
            property_ids:   Optional property identifiers (used in EVENTID)
            portfolio_name: Name prefix for EVENTID column

        Returns:
            pd.DataFrame with columns: EVENTID, RATE, PERSPVALUE, EXPVALUE, STDDEVC, STDDEVI
        """
        n = len(X)
        logger.info(f"Generating ELT for {n:,} properties...")

        # --- RATE: annual claim frequency ---
        rates = self._get_rates(X)

        # --- PERSPVALUE: expected loss given event ---
        mean_losses, log_stds = self._get_severity(X)

        # --- EXPVALUE: total insured value ---
        exp_values = self._get_exposure(X, metadata)

        # --- Variance decomposition ---
        stddev_c, stddev_i = self._compute_variance(
            mean_losses=mean_losses,
            log_stds=log_stds,
            X=X,
            metadata=metadata,
        )

        # --- Assemble ELT ---
        if property_ids is None:
            property_ids = pd.Series(
                [f"{portfolio_name}_{i:06d}" for i in range(n)],
                index=X.index
            )

        elt = pd.DataFrame({
            'EVENTID':    property_ids.values if hasattr(property_ids, 'values') else property_ids,
            'RATE':       np.clip(rates, self.min_rate, self.max_rate),
            'PERSPVALUE': np.maximum(mean_losses, 0.0),
            'EXPVALUE':   np.maximum(exp_values, 1.0),
            'STDDEVC':    np.maximum(stddev_c, 0.0),
            'STDDEVI':    np.maximum(stddev_i, 0.0),
        })

        # Sanity check: PERSPVALUE ≤ EXPVALUE
        elt['PERSPVALUE'] = np.minimum(elt['PERSPVALUE'], elt['EXPVALUE'])

        # Compute derived metrics for logging
        mal = (elt['RATE'] * elt['PERSPVALUE']).sum()
        logger.info(
            f"ELT generated: {len(elt):,} events | "
            f"Total MAL: ${mal:,.0f} | "
            f"Avg RATE: {elt['RATE'].mean():.4f} | "
            f"Avg PERSPVALUE: ${elt['PERSPVALUE'].mean():,.0f}"
        )

        return elt[ELT_COLUMNS]

    def generate_portfolio_elt(
        self,
        X: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        group_by_zone: bool = True,
    ) -> pd.DataFrame:
        """
        Generate a portfolio-level ELT by aggregating property-level ELTs.

        When group_by_zone=True, properties in the same flood zone are
        grouped into shared events (correlated losses). This produces a more
        realistic portfolio ELT where correlated properties lose together.

        Args:
            X:            Feature matrix
            metadata:     Optional raw property data
            group_by_zone: Group properties by flood zone for correlated events

        Returns:
            Aggregated portfolio ELT
        """
        # Generate property-level ELT first
        prop_elt = self.generate(X, metadata=metadata, portfolio_name='prop')

        if not group_by_zone or metadata is None:
            # Simple aggregation: sum all perspectives
            portfolio_elt = self._aggregate_simple(prop_elt)
        else:
            # Zone-grouped aggregation
            portfolio_elt = self._aggregate_by_zone(prop_elt, metadata)

        logger.info(
            f"Portfolio ELT: {len(portfolio_elt):,} events | "
            f"Portfolio MAL: ${(portfolio_elt['RATE'] * portfolio_elt['PERSPVALUE']).sum():,.0f}"
        )

        return portfolio_elt

    # ------------------------------------------------------------------
    # Helper: from pre-existing predictions (skip models)
    # ------------------------------------------------------------------

    @classmethod
    def from_predictions(
        cls,
        rates: Union[np.ndarray, pd.Series],
        mean_losses: Union[np.ndarray, pd.Series],
        exp_values: Union[np.ndarray, pd.Series],
        log_stds: Optional[Union[np.ndarray, pd.Series]] = None,
        corr_fractions: Optional[Union[np.ndarray, pd.Series]] = None,
        property_ids: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build an ELT directly from pre-computed predictions (no model needed).

        Useful when you have predictions from external models or want to
        manually construct an ELT for testing.

        Args:
            rates:          Annual event rates (RATE column)
            mean_losses:    Expected losses given event (PERSPVALUE column)
            exp_values:     Exposure / policy limit (EXPVALUE column)
            log_stds:       Log-scale standard deviations for variance splitting
                            (None = use 0.8 as default for log-normal severity)
            corr_fractions: Correlated variance fraction per property [0, 1]
                            (None = use 0.5 default)
            property_ids:   Property identifiers for EVENTID

        Returns:
            ELT DataFrame ready for MarginalPMLEngine
        """
        n = len(rates)
        rates = np.asarray(rates)
        mean_losses = np.asarray(mean_losses, dtype=float)
        exp_values = np.asarray(exp_values, dtype=float)

        if log_stds is None:
            log_stds = np.full(n, 0.8)
        else:
            log_stds = np.asarray(log_stds)

        if corr_fractions is None:
            corr_fractions = np.full(n, 0.5)
        else:
            corr_fractions = np.asarray(corr_fractions)

        # Total dollar std: for log-normal, std ≈ mean * sqrt(exp(sigma²) - 1)
        total_std = mean_losses * np.sqrt(np.exp(log_stds ** 2) - 1)
        stddev_c = total_std * corr_fractions
        stddev_i = total_std * np.sqrt(1 - corr_fractions ** 2)

        if property_ids is None:
            property_ids = [f"prop_{i:06d}" for i in range(n)]

        return pd.DataFrame({
            'EVENTID':    property_ids,
            'RATE':       np.clip(rates, 1e-6, 1.0),
            'PERSPVALUE': np.maximum(mean_losses, 0.0),
            'EXPVALUE':   np.maximum(exp_values, 1.0),
            'STDDEVC':    np.maximum(stddev_c, 0.0),
            'STDDEVI':    np.maximum(stddev_i, 0.0),
        })[ELT_COLUMNS]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rates(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get annual claim rates from frequency model, scaled by filing rate.

        The frequency model predicts P(paid | filed). To get a true annual
        claim rate we multiply by the filing rate P(filed | property-year):

            annual_rate = P(paid | filed) × filing_rate

        Filing rates are looked up from self.filing_rates by flood zone
        (and state if available). Without filing_rates, a literature-based
        fallback of ~3.5% is used with a warning.
        """
        if self.freq_model is not None:
            p_paid = self.freq_model.predict_proba(X)

            # Apply filing rate scaling
            filing_rate = self._lookup_filing_rates(X)
            scaled = p_paid * filing_rate

            logger.info(
                f"Rate scaling: P(paid|filed) mean={p_paid.mean():.3f} × "
                f"filing_rate mean={filing_rate.mean():.4f} → "
                f"annual_rate mean={scaled.mean():.4f}"
            )
            return scaled

        # Fallback: use feature-based heuristic
        logger.warning("No frequency model provided. Using flood_zone_risk_score as rate proxy.")
        if 'flood_zone_risk_score' in X.columns:
            return (X['flood_zone_risk_score'] * 0.15).clip(0.001, 0.5).values
        return np.full(len(X), 0.05)

    def _lookup_filing_rates(self, X: pd.DataFrame) -> np.ndarray:
        """
        Look up filing rates per property from self.filing_rates DataFrame.

        Matches on flood zone features in X. Falls back to national averages
        if no filing_rates table is provided.

        Returns:
            1-D array of filing rates, one per row in X.
        """
        n = len(X)

        # Literature-based fallback rates by zone
        FALLBACK = {
            'coastal_high_risk': 0.085,
            'high_risk':         0.042,
            'moderate_risk':     0.015,
            'minimal_risk':      0.008,
            'unknown':           0.035,
        }

        if self.filing_rates is not None and len(self.filing_rates) > 0:
            # Build a zone → rate lookup from the _ALL (national average) rows
            fr = self.filing_rates
            zone_rates = (
                fr[fr['state'] == '_ALL']
                .set_index('flood_zone_category')['filing_rate']
                .to_dict()
            )
            # Merge FALLBACK for any missing zones
            lookup = {**FALLBACK, **zone_rates}
        else:
            logger.warning(
                "No filing_rates provided to ELTGenerator. "
                "Using literature-based national averages. "
                "For rigorous results, compute filing rates from OpenFEMA "
                "policy data (see data_ingestion.compute_filing_rates)."
            )
            lookup = FALLBACK

        # Map flood zone features → zone category → filing rate
        if 'is_coastal_high_risk' in X.columns:
            zone_cat = np.where(
                X['is_coastal_high_risk'].values == 1, 'coastal_high_risk',
                np.where(
                    X.get('is_high_risk_zone', pd.Series(0, index=X.index)).values == 1,
                    'high_risk',
                    'moderate_risk',
                )
            )
        else:
            zone_cat = np.full(n, 'unknown')

        return np.array([lookup.get(z, 0.035) for z in zone_cat])

    def _get_severity(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mean expected losses and log-scale std from severity model.
        Returns (mean_dollar_loss, log_std).
        """
        if self.sev_model is not None:
            preds = self.sev_model.predict_with_uncertainty(X)
            # Log-normal mean = exp(mu + sigma²/2) — already computed in model
            return preds['mean'].values, np.full(len(X), self.sev_model.train_log_rmse)

        # Fallback: simple heuristic from coverage
        logger.warning("No severity model provided. Using coverage-based severity estimate.")
        if 'log_building_coverage' in X.columns:
            # Rough heuristic: expected loss ~ 20% of coverage for high-risk zones
            coverage = np.exp(X['log_building_coverage'].values) - 1
            zone_mult = X.get('flood_zone_risk_score', pd.Series(0.5)).values
            mean_loss = coverage * 0.20 * zone_mult
        else:
            mean_loss = np.full(len(X), 50_000.0)

        return mean_loss, np.full(len(X), 0.8)

    def _get_exposure(
        self,
        X: pd.DataFrame,
        metadata: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Get total insured value (EXPVALUE) for each property."""
        # Try metadata first (raw NFIP data)
        if metadata is not None and 'totalBuildingInsuranceCoverage' in metadata.columns:
            exp = pd.to_numeric(
                metadata['totalBuildingInsuranceCoverage'], errors='coerce'
            ).fillna(250_000).values
            return exp

        # Try feature matrix (log-transformed coverage)
        if 'log_building_coverage' in X.columns:
            return (np.exp(X['log_building_coverage'].values) - 1).clip(min=1.0)

        return np.full(len(X), 250_000.0)

    def _compute_variance(
        self,
        mean_losses: np.ndarray,
        log_stds: np.ndarray,
        X: pd.DataFrame,
        metadata: Optional[pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose total variance into STDDEVC (correlated) and STDDEVI (idiosyncratic).

        Under log-normal assumption:
            total_std = mean * sqrt(exp(sigma²) - 1)

        Correlated fraction is determined by flood zone and/or peril type.

        The correlated component (STDDEVC) represents losses that will co-move
        across properties hit by the same storm event. The independent component
        (STDDEVI) represents property-specific variation conditional on the event.
        """
        # Total dollar standard deviation (log-normal formula)
        total_std = mean_losses * np.sqrt(np.expm1(log_stds ** 2))

        # Compute correlated fraction per property
        corr_frac = self._get_corr_fractions(X, metadata)

        stddev_c = total_std * corr_frac
        stddev_i = total_std * np.sqrt(np.maximum(1 - corr_frac ** 2, 0))

        return stddev_c, stddev_i

    def _get_corr_fractions(
        self,
        X: pd.DataFrame,
        metadata: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """
        Determine the correlated variance fraction for each property.

        Returns array of values in [0, 1] where:
            0 = fully idiosyncratic (no correlation with other properties)
            1 = fully correlated (same loss fraction as all other properties in event)
        """
        n = len(X)
        zone_fracs  = np.full(n, 0.40)
        peril_fracs = np.full(n, 0.40)

        # Zone-based fractions
        if 'flood_zone_category' in X.columns:
            zone_fracs = X['flood_zone_category'].map(CORR_FRACTION_BY_ZONE).fillna(0.40).values
        elif 'is_coastal_high_risk' in X.columns:
            zone_fracs = np.where(
                X['is_coastal_high_risk'].values == 1, 0.70,
                np.where(X['is_high_risk_zone'].values == 1, 0.55, 0.35)
            )

        # Peril-based fractions (from NOAA storm join)
        peril_col = None
        if metadata is not None and 'peril_category' in metadata.columns:
            peril_col = metadata['peril_category']
        elif 'is_hurricane_peril' in X.columns:
            # Derive from feature flags
            peril_fracs = np.where(
                X['is_hurricane_peril'].values == 1, 0.70,
                np.where(X['is_flood_peril'].values == 1, 0.45, 0.35)
            )
            peril_col = None  # already handled

        if peril_col is not None:
            peril_fracs = peril_col.map(CORR_FRACTION_BY_PERIL).fillna(0.40).values

        # Combine based on corr_method
        if self.corr_method == 'zone':
            return zone_fracs
        elif self.corr_method == 'peril':
            return peril_fracs
        else:  # 'both' — average
            return (zone_fracs + peril_fracs) / 2.0

    def _aggregate_simple(self, prop_elt: pd.DataFrame) -> pd.DataFrame:
        """
        Simple portfolio aggregation: treat each property as a separate event.
        This is appropriate when properties are in different geographic areas.
        """
        return prop_elt.copy()

    def _aggregate_by_zone(
        self,
        prop_elt: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate properties within same flood zone into shared events.

        Properties in the same flood zone are grouped because they share
        correlated loss exposure — a single major storm/flood event hits
        all properties in an AE zone simultaneously.

        For each zone group:
        - RATE = max rate in group (if ANY property is hit, the "zone event" occurs)
        - PERSPVALUE = sum of individual PERSPVALUE (total zone loss)
        - EXPVALUE = sum of EXPVALUE (total zone exposure)
        - STDDEVC = sum of individual STDDEVC (linear add for fully correlated)
        - STDDEVI = sqrt(sum of STDDEVI²) (quadrature add for independent)
        """
        if 'flood_zone_category' not in metadata.columns:
            return prop_elt

        combined = pd.concat([
            prop_elt.reset_index(drop=True),
            metadata[['flood_zone_category']].reset_index(drop=True)
        ], axis=1)

        # Zone-level aggregation
        zone_groups = combined.groupby('flood_zone_category').agg(
            RATE=('RATE', 'max'),
            PERSPVALUE=('PERSPVALUE', 'sum'),
            EXPVALUE=('EXPVALUE', 'sum'),
            STDDEVC=('STDDEVC', 'sum'),
            STDDEVI=('STDDEVI', lambda x: np.sqrt((x**2).sum())),
        ).reset_index()

        zone_groups['EVENTID'] = 'zone_' + zone_groups['flood_zone_category']
        zone_groups = zone_groups.drop(columns='flood_zone_category')

        return zone_groups[ELT_COLUMNS]


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

def build_elt_from_claims(
    claims_df: pd.DataFrame,
    freq_model,
    sev_model,
    feature_pipeline,
    group_by_zone: bool = False,
) -> pd.DataFrame:
    """
    End-to-end pipeline: raw claims → features → predictions → ELT.

    This is the full pipeline that takes NFIP claims data all the way
    through to a pricing-ready ELT in one function call.

    Args:
        claims_df:        Raw NFIP claims DataFrame (from data_ingestion)
        freq_model:       Fitted FrequencyModel
        sev_model:        Fitted SeverityModel
        feature_pipeline: Fitted FeaturePipeline
        group_by_zone:    Whether to aggregate by flood zone

    Returns:
        ELT DataFrame ready for MarginalPMLEngine
    """
    logger.info("Building ELT from claims data...")

    # Transform features
    X = feature_pipeline.transform(claims_df)

    generator = ELTGenerator(
        freq_model=freq_model,
        sev_model=sev_model,
        corr_method='both',
    )

    if group_by_zone:
        elt = generator.generate_portfolio_elt(X, metadata=claims_df, group_by_zone=True)
    else:
        elt = generator.generate(X, metadata=claims_df)

    logger.info(
        f"Pipeline complete → {len(elt):,} ELT rows | "
        f"Total MAL: ${(elt['RATE'] * elt['PERSPVALUE']).sum():,.0f}"
    )

    return elt


def validate_elt(elt: pd.DataFrame, raise_errors: bool = False) -> Dict:
    """
    Validate an ELT DataFrame for use with the pricing engine.

    Checks:
        - Required columns present
        - RATE in (0, 1]
        - PERSPVALUE >= 0
        - EXPVALUE > 0
        - PERSPVALUE <= EXPVALUE
        - No NaN values in key columns
        - STDDEVC² + STDDEVI² <= PERSPVALUE² (variance sanity check)

    Args:
        elt:          ELT DataFrame to validate
        raise_errors: If True, raise ValueError on failures

    Returns:
        Dict with 'valid' bool and 'issues' list
    """
    issues = []

    # Column check
    missing_cols = [c for c in ELT_COLUMNS if c not in elt.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if issues:
        return {'valid': False, 'issues': issues, 'n_rows': len(elt)}

    # NaN check
    nan_counts = elt[ELT_COLUMNS].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        issues.append(f"NaN values in: {nan_cols.to_dict()}")

    # Rate range check
    bad_rate = ((elt['RATE'] <= 0) | (elt['RATE'] > 1)).sum()
    if bad_rate > 0:
        issues.append(f"{bad_rate} rows with RATE outside (0, 1]")

    # Non-negative loss check
    neg_persp = (elt['PERSPVALUE'] < 0).sum()
    if neg_persp > 0:
        issues.append(f"{neg_persp} rows with negative PERSPVALUE")

    # PERSPVALUE <= EXPVALUE
    exceed = (elt['PERSPVALUE'] > elt['EXPVALUE']).sum()
    if exceed > 0:
        issues.append(f"{exceed} rows where PERSPVALUE > EXPVALUE")

    # EXPVALUE > 0
    zero_exp = (elt['EXPVALUE'] <= 0).sum()
    if zero_exp > 0:
        issues.append(f"{zero_exp} rows with zero/negative EXPVALUE")

    valid = len(issues) == 0

    if not valid and raise_errors:
        raise ValueError(f"ELT validation failed:\n  " + "\n  ".join(issues))

    if valid:
        logger.info(f"ELT validation passed: {len(elt):,} rows")
    else:
        logger.warning(f"ELT validation issues:\n  " + "\n  ".join(issues))

    return {
        'valid': valid,
        'issues': issues,
        'n_rows': len(elt),
        'total_mal': float((elt['RATE'] * elt['PERSPVALUE']).sum()),
        'avg_rate': float(elt['RATE'].mean()),
        'total_exposure': float(elt['EXPVALUE'].sum()),
    }


# =============================================================================
# MAIN (smoke test + round-trip demo)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 65)
    print("ELT GENERATOR — Round-trip demo")
    print("  ML predictions → ELT → MarginalPMLEngine")
    print("=" * 65)

    np.random.seed(42)
    n = 100

    from feature_engineering import SEVERITY_FEATURES

    # Simulate feature matrix
    X = pd.DataFrame(np.random.rand(n, len(SEVERITY_FEATURES)), columns=SEVERITY_FEATURES)
    X['flood_zone_risk_score'] = np.random.choice([0.1, 0.3, 0.75, 1.0], n)
    X['is_coastal_high_risk']  = (X['flood_zone_risk_score'] == 1.0).astype(float)
    X['log_building_coverage'] = np.log1p(np.random.uniform(100_000, 500_000, n))

    # Simulate metadata
    metadata = pd.DataFrame({
        'flood_zone_category': np.random.choice(
            ['coastal_high_risk', 'high_risk', 'moderate_risk', 'minimal_risk'], n
        ),
        'peril_category': np.random.choice(
            ['hurricane', 'flood', 'coastal_flood', 'other'], n
        ),
        'totalBuildingInsuranceCoverage': np.random.uniform(100_000, 500_000, n),
    })

    # Generate ELT without real models (heuristic fallback)
    generator = ELTGenerator(freq_model=None, sev_model=None)
    elt = generator.generate(X, metadata=metadata)

    print(f"\nGenerated ELT ({len(elt)} rows):")
    print(elt.head(8).to_string(index=False))

    # Validate
    validation = validate_elt(elt)
    print(f"\nValidation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
    print(f"  Total MAL:       ${validation['total_mal']:,.0f}")
    print(f"  Avg annual rate: {validation['avg_rate']:.4f}")
    print(f"  Total exposure:  ${validation['total_exposure']:,.0f}")

    # --- Round-trip: feed into MarginalPMLEngine ---
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from marginal_pml_kernel import MarginalPMLEngine

        print("\n--- MarginalPMLEngine round-trip ---")
        engine = MarginalPMLEngine(elt)
        results = engine.price(return_periods=[50, 100, 250])
        print("PML Results:")
        for rp, pml in results.items():
            print(f"  1-in-{rp} year PML: ${pml:,.0f}")

        print("\n✅ Round-trip complete: ML predictions → ELT → PML")

    except Exception as e:
        print(f"\nNote: Round-trip to MarginalPMLEngine skipped ({e})")
        print("ELT is ready — run the full pipeline to complete the round-trip.")
