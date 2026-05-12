"""
Feature Engineering Pipeline — NFIP Claims → ML-ready features

Transforms raw NFIP claims (enriched with NOAA storm event data) into
the feature matrix used to train and score the frequency and severity models.

Key features:
    - Flood zone risk category (ordinal encoding)
    - Building characteristics (occupancy, foundation, floors, age)
    - Insurance coverage ratios
    - Storm event features (hurricane category, peril type)
    - Geographic risk proxies (state, coastal indicator)
    - Post-FIRM construction indicator (regulatory compliance era)

Usage:
    from feature_engineering import FeaturePipeline

    pipeline = FeaturePipeline()
    X, y_sev, y_freq = pipeline.fit_transform(claims_df)

    # Score new properties
    X_new = pipeline.transform(new_properties_df)
"""

import logging
import warnings
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Ordinal flood zone risk ordering (low → high risk)
FLOOD_ZONE_ORDER = ['minimal_risk', 'moderate_risk', 'unknown', 'high_risk', 'coastal_high_risk']

# Occupancy type risk mapping (OpenFEMA codes — both standard and subsidized)
OCCUPANCY_TYPE_MAP = {
    1:  'single_family',
    2:  'two_to_four_family',
    3:  'other_residential',
    4:  'non_residential',
    6:  'residential_condominium',
    11: 'single_family',            # Subsidized
    12: 'two_to_four_family',       # Subsidized
    13: 'other_residential',        # Subsidized
    14: 'non_residential',          # Subsidized
    15: 'single_family',            # With detached structures
    16: 'two_to_four_family',       # With detached structures
    17: 'other_residential',        # With detached structures
    18: 'non_residential',          # With detached structures
    19: 'single_family',            # With LOMA
}

# Foundation / basement type mapping
BASEMENT_TYPE_MAP = {
    '0': 'no_basement',
    '1': 'finished_basement',
    '2': 'unfinished_basement',
    '3': 'enclosure_below_grade',
    '4': 'crawlspace',
}

# Hurricane category → numeric risk score
HURRICANE_CAT_RISK = {
    0: 0.0,   # Tropical storm / depression
    1: 0.2,
    2: 0.4,
    3: 0.65,
    4: 0.85,
    5: 1.0,
}

# High-risk coastal states
COASTAL_STATES = {'FL', 'TX', 'LA', 'MS', 'AL', 'GA', 'SC', 'NC', 'VA', 'MD',
                  'DE', 'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'ME', 'PR', 'VI'}

# Features used for modeling (after engineering)
SEVERITY_FEATURES = [
    'flood_zone_risk_score',
    'is_coastal_high_risk',
    'is_high_risk_zone',
    'occupancy_is_residential',
    'occupancy_is_nonresidential',
    'has_basement',
    'log_building_coverage',
    'log_contents_coverage',
    'has_contents_coverage',
    'coverage_to_value_ratio',
    'num_floors',
    'construction_age_years',
    'is_post_firm',
    'has_elevation_cert',
    'hurricane_risk_score',
    'is_hurricane_peril',
    'is_flood_peril',
    'latitude',
    'longitude',
    'log_storm_damage',
]

# Frequency features: EXCLUDE leaky features that wouldn't be available at
# scoring time or are too correlated with the claim outcome in a claims-only dataset.
#   - log_storm_damage:       direct leaker — NOAA damage is only known after the event
#   - coverage_to_value_ratio: in claims-only data, heavily correlated with payment outcome
FREQUENCY_FEATURES = [
    f for f in SEVERITY_FEATURES
    if f not in ('log_storm_damage', 'coverage_to_value_ratio')
]


# =============================================================================
# INDIVIDUAL TRANSFORMERS
# =============================================================================

class FloodZoneEncoder(BaseEstimator, TransformerMixin):
    """
    Encode flood zone into numeric risk scores.

    Produces:
        - flood_zone_risk_score:  0.0–1.0 ordinal risk
        - is_coastal_high_risk:   1 if VE/V zone
        - is_high_risk_zone:      1 if AE/A/etc. zone (SFHA)
    """

    ZONE_RISK_SCORES = {
        'minimal_risk':       0.1,
        'moderate_risk':      0.3,
        'unknown':            0.4,
        'high_risk':          0.75,
        'coastal_high_risk':  1.0,
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cat_col = 'flood_zone_category'

        if cat_col not in X.columns:
            # Try to derive from floodZone
            if 'floodZone' in X.columns:
                from data_ingestion import FLOOD_ZONE_CATEGORIES
                X[cat_col] = X['floodZone'].str.strip().str.upper().map(FLOOD_ZONE_CATEGORIES).fillna('unknown')
            else:
                X[cat_col] = 'unknown'

        X['flood_zone_risk_score'] = (
            X[cat_col].map(self.ZONE_RISK_SCORES).fillna(0.4)
        )
        X['is_coastal_high_risk'] = (X[cat_col] == 'coastal_high_risk').astype(int)
        X['is_high_risk_zone'] = (
            X[cat_col].isin(['high_risk', 'coastal_high_risk'])
        ).astype(int)

        return X


class BuildingFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Encode building characteristics.

    Produces:
        - occupancy_is_residential
        - occupancy_is_nonresidential
        - has_basement
        - num_floors (clipped 1–10)
        - construction_age_years
        - is_post_firm
        - has_elevation_cert
        - log_building_coverage
        - coverage_to_value_ratio
    """

    def __init__(self, reference_year: int = 2023):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Occupancy type — convert to int before mapping (API returns float64)
        occ_raw = pd.to_numeric(
            X.get('occupancyType', pd.Series([1] * len(X))), errors='coerce'
        ).fillna(1).astype(int)
        occ_mapped = occ_raw.map(OCCUPANCY_TYPE_MAP).fillna('single_family')
        X['occupancy_is_residential'] = (
            occ_mapped.isin(['single_family', 'two_to_four_family', 'other_residential',
                             'residential_condominium'])
        ).astype(int)
        X['occupancy_is_nonresidential'] = (occ_mapped == 'non_residential').astype(int)

        # Basement / foundation — convert to numeric; NaN = unknown (treat as no basement)
        bsmt_raw = pd.to_numeric(
            X.get('basementEnclosureCrawlspaceType', pd.Series([0] * len(X))), errors='coerce'
        ).fillna(0).astype(int)
        X['has_basement'] = (bsmt_raw > 0).astype(int)

        # Number of floors
        floors = pd.to_numeric(X.get('numberOfFloorsInInsuredBuilding', 1), errors='coerce').fillna(1)
        X['num_floors'] = floors.clip(1, 10)

        # Construction age
        const_year = pd.to_numeric(X.get('construction_year',
                                          X.get('originalConstructionDate', np.nan)),
                                   errors='coerce')
        if const_year.dtype == 'O':  # might be datetime string
            const_year = pd.to_datetime(const_year, errors='coerce').dt.year
        const_year = const_year.fillna(1980).clip(1800, self.reference_year)
        X['construction_age_years'] = self.reference_year - const_year

        # Post-FIRM indicator (built after flood maps established ~1978)
        post_firm = X.get('postFIRMConstructionIndicator', pd.Series([None] * len(X)))
        if post_firm.dtype == 'O':
            post_firm = post_firm.map({'true': 1, 'false': 0, True: 1, False: 0})
        X['is_post_firm'] = pd.to_numeric(post_firm, errors='coerce').fillna(
            (const_year >= 1978).astype(int)
        )

        # Elevation certificate
        elev = X.get('elevationCertificateIndicator', pd.Series([None] * len(X)))
        X['has_elevation_cert'] = (
            pd.to_numeric(elev, errors='coerce').fillna(0) > 0
        ).astype(int)

        # Building coverage (log transform)
        coverage = pd.to_numeric(
            X.get('totalBuildingInsuranceCoverage', 100_000), errors='coerce'
        ).fillna(100_000).clip(lower=1)
        X['log_building_coverage'] = np.log1p(coverage)

        # Contents coverage (log transform + binary flag)
        contents_cov = pd.to_numeric(
            X.get('totalContentsInsuranceCoverage', 0), errors='coerce'
        ).fillna(0).clip(lower=0)
        X['log_contents_coverage'] = np.log1p(contents_cov)
        X['has_contents_coverage'] = (contents_cov > 0).astype(int)

        # Coverage to property value ratio
        prop_value = pd.to_numeric(
            X.get('buildingPropertyValue', coverage), errors='coerce'
        ).fillna(coverage).clip(lower=1)
        X['coverage_to_value_ratio'] = (coverage / prop_value).clip(0, 2)

        return X


class GeographicEncoder(BaseEstimator, TransformerMixin):
    """
    Encode geographic location features from latitude/longitude.

    Produces:
        - latitude:   Property latitude (clipped to US range, NaN filled with median)
        - longitude:  Property longitude (clipped to US range, NaN filled with median)
    """

    # Continental US + territories bounding box
    LAT_RANGE = (17.0, 50.0)    # PR (~18°N) to northern Maine (~47°N)
    LON_RANGE = (-125.0, -65.0)  # West coast to east coast

    # Fallback medians (approximate US coastal property centroid)
    DEFAULT_LAT = 33.0
    DEFAULT_LON = -82.0

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Latitude — clean, clip, and fill
        lat = pd.to_numeric(X.get('latitude', self.DEFAULT_LAT), errors='coerce')
        # Flag clearly invalid values before clipping
        lat = lat.where(
            (lat >= self.LAT_RANGE[0]) & (lat <= self.LAT_RANGE[1]),
            other=np.nan
        )
        lat = lat.fillna(self.DEFAULT_LAT)
        X['latitude'] = lat

        # Longitude — clean, clip, and fill
        lon = pd.to_numeric(X.get('longitude', self.DEFAULT_LON), errors='coerce')
        lon = lon.where(
            (lon >= self.LON_RANGE[0]) & (lon <= self.LON_RANGE[1]),
            other=np.nan
        )
        lon = lon.fillna(self.DEFAULT_LON)
        X['longitude'] = lon

        return X


class StormEventEncoder(BaseEstimator, TransformerMixin):
    """
    Encode storm event features from NOAA join.

    Produces:
        - hurricane_risk_score  (0–1 based on category)
        - is_hurricane_peril
        - is_flood_peril
        - log_storm_damage
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        n = len(X)

        def _get_col(df, name, default):
            """Safely get a column as a Series even if missing."""
            if name in df.columns:
                return df[name]
            return pd.Series([default] * n, index=df.index)

        # Hurricane category risk score
        hur_cat = pd.to_numeric(_get_col(X, 'hurricane_category', np.nan), errors='coerce')
        X['hurricane_risk_score'] = hur_cat.map(HURRICANE_CAT_RISK).fillna(0.0)

        # Peril flags — accept either 'peril_category' or fall back to 'EVENT_TYPE'
        if 'peril_category' in X.columns:
            peril = X['peril_category'].fillna('other').astype(str).str.lower()
        elif 'EVENT_TYPE' in X.columns:
            # Map NOAA event types onto our peril categories
            event_map = {
                'hurricane (typhoon)': 'hurricane',
                'tropical storm': 'tropical_storm',
                'storm surge/tide': 'storm_surge',
                'coastal flood': 'coastal_flood',
                'flash flood': 'flood',
                'flood': 'flood',
                'heavy rain': 'heavy_rain',
            }
            peril = (
                X['EVENT_TYPE'].fillna('other').astype(str).str.lower().map(event_map).fillna('other')
            )
        else:
            peril = pd.Series(['other'] * n, index=X.index)
        X['is_hurricane_peril'] = peril.isin(['hurricane', 'tropical_storm', 'storm_surge']).astype(int)
        X['is_flood_peril'] = peril.isin(['flood', 'coastal_flood', 'heavy_rain']).astype(int)

        # Coastal state flag
        state_col = _get_col(X, 'reportedState', None)
        if state_col.isna().all() and 'source_state' in X.columns:
            state_col = X['source_state']
        state = state_col.fillna('XX').astype(str).str.upper()
        X['is_coastal_state'] = state.isin(COASTAL_STATES).astype(int)

        # Storm damage magnitude — accept either canonical or raw NOAA column
        if 'storm_property_damage_usd' in X.columns:
            storm_dmg_src = X['storm_property_damage_usd']
        elif 'DAMAGE_PROPERTY_NUM' in X.columns:
            storm_dmg_src = X['DAMAGE_PROPERTY_NUM']
        else:
            storm_dmg_src = pd.Series([0] * n, index=X.index)
        storm_dmg = pd.to_numeric(storm_dmg_src, errors='coerce').fillna(0).clip(lower=0)
        X['log_storm_damage'] = np.log1p(storm_dmg)

        return X


# =============================================================================
# FULL FEATURE PIPELINE
# =============================================================================

class FeaturePipeline:
    """
    End-to-end feature engineering pipeline for NFIP claims data.

    Applies all feature transformers in sequence and returns a clean
    feature matrix ready for model training.

    Args:
        reference_year: Year to use for construction age calculation
        severity_target: Column name for severity target
        frequency_target: Column name for frequency target
        min_loss_for_severity: Minimum paid claim amount to include in severity training
    """

    def __init__(
        self,
        reference_year: int = 2023,
        severity_target: str = 'amountPaidOnBuildingClaim',
        frequency_target: str = 'had_claim',
        min_loss_for_severity: float = 100.0,
    ):
        self.reference_year = reference_year
        self.severity_target = severity_target
        self.frequency_target = frequency_target
        self.min_loss_for_severity = min_loss_for_severity

        self._transformers = [
            FloodZoneEncoder(),
            BuildingFeatureEncoder(reference_year=reference_year),
            GeographicEncoder(),
            StormEventEncoder(),
        ]
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        return_targets: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        Fit pipeline on training data and return feature matrix + targets.

        Args:
            df: Raw claims DataFrame from data_ingestion
            return_targets: Whether to extract and return target variables

        Returns:
            (X, y_severity, y_frequency) — X is the feature matrix,
            y_severity is log(claim amount) for rows with claims,
            y_frequency is binary 0/1 claim flag for all rows.
        """
        logger.info(f"Fitting feature pipeline on {len(df):,} records...")

        # Apply all transformers
        transformed = df.copy()
        for transformer in self._transformers:
            transformed = transformer.fit(transformed).transform(transformed)

        self._fitted = True
        self._fitted_columns = SEVERITY_FEATURES

        X = self._select_features(transformed)

        if not return_targets:
            return X, None, None

        # Frequency target: binary claim flag
        y_freq = None
        if self.frequency_target in transformed.columns:
            y_freq = transformed[self.frequency_target].fillna(0).astype(int)
        elif 'had_claim' in transformed.columns:
            y_freq = transformed['had_claim'].fillna(0).astype(int)

        # Severity target: log(claim amount) for claims > min threshold
        y_sev = None
        if self.severity_target in transformed.columns:
            raw_sev = pd.to_numeric(transformed[self.severity_target], errors='coerce').fillna(0)
            # Log-transform; only positive losses used for severity model
            y_sev = np.where(raw_sev > self.min_loss_for_severity, np.log(raw_sev), np.nan)
            y_sev = pd.Series(y_sev, index=transformed.index, name='log_severity')

        logger.info(
            f"Feature matrix: {X.shape} | "
            f"Freq positives: {(y_freq == 1).sum() if y_freq is not None else 'N/A'} | "
            f"Sev records: {pd.notna(y_sev).sum() if y_sev is not None else 'N/A'}"
        )

        return X, y_sev, y_freq

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted pipeline (no target extraction).

        Args:
            df: Raw property/claims DataFrame

        Returns:
            Feature matrix (pd.DataFrame)
        """
        transformed = df.copy()
        for transformer in self._transformers:
            transformed = transformer.transform(transformed)
        return self._select_features(transformed)

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order the final feature columns."""
        available = [f for f in SEVERITY_FEATURES if f in df.columns]
        missing = [f for f in SEVERITY_FEATURES if f not in df.columns]

        if missing:
            logger.warning(f"Missing features (will be filled with 0): {missing}")
            for col in missing:
                df[col] = 0.0

        return df[SEVERITY_FEATURES].astype(float)

    @property
    def feature_names(self) -> List[str]:
        """Return the list of output feature names."""
        return SEVERITY_FEATURES.copy()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return human-readable descriptions for each feature."""
        return {
            'flood_zone_risk_score':      'Flood zone risk score (0=minimal, 1=coastal V-zone)',
            'is_coastal_high_risk':       '1 if property is in a coastal V/VE zone',
            'is_high_risk_zone':          '1 if property is in any SFHA (A or V zone)',
            'occupancy_is_residential':   '1 if building is residential occupancy',
            'occupancy_is_nonresidential':'1 if building is commercial/non-residential',
            'has_basement':               '1 if building has a basement or crawlspace',
            'log_building_coverage':      'Log of total building insurance coverage ($)',
            'log_contents_coverage':      'Log of total contents insurance coverage ($)',
            'has_contents_coverage':      '1 if property has contents insurance coverage',
            'coverage_to_value_ratio':    'Insurance coverage / building property value',
            'num_floors':                 'Number of floors in the building (clipped 1-10)',
            'construction_age_years':     'Years since original construction',
            'is_post_firm':               '1 if built after flood maps established (post-1978)',
            'has_elevation_cert':         '1 if an elevation certificate is on file',
            'hurricane_risk_score':       'Hurricane category risk score (0=TS, 1=Cat5)',
            'is_hurricane_peril':         '1 if loss event was hurricane/tropical/surge',
            'is_flood_peril':             '1 if loss event was riverine/coastal flood',
            'latitude':                   'Property latitude (clipped to US range)',
            'longitude':                  'Property longitude (clipped to US range)',
            'log_storm_damage':           'Log of total storm property damage (NOAA estimate)',
        }


# =============================================================================
# CONVENIENCE: build features from raw claims directly
# =============================================================================

def build_features(
    claims_df: pd.DataFrame,
    reference_year: int = 2023,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience wrapper: raw claims → feature matrix + targets.

    Returns:
        X:      Feature matrix (all rows)
        y_sev:  Log severity target (NaN for non-claims)
        y_freq: Binary frequency target (all rows)
    """
    pipeline = FeaturePipeline(reference_year=reference_year)
    return pipeline.fit_transform(claims_df, return_targets=True)


# =============================================================================
# MAIN (quick smoke test)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Generate synthetic test data that mimics NFIP claims structure
    np.random.seed(42)
    n = 500

    test_data = pd.DataFrame({
        'floodZone': np.random.choice(['AE', 'VE', 'X', 'AH', 'B'], n),
        'occupancyType': np.random.choice(['1', '2', '3', '4'], n),
        'basementEnclosureCrawlspaceType': np.random.choice(['0', '1', '2', '3', '4'], n),
        'numberOfFloorsInInsuredBuilding': np.random.randint(1, 4, n),
        'construction_year': np.random.randint(1950, 2020, n),
        'postFIRMConstructionIndicator': np.random.choice([True, False], n),
        'elevationCertificateIndicator': np.random.choice([0, 1], n),
        'totalBuildingInsuranceCoverage': np.random.uniform(50_000, 500_000, n),
        'buildingPropertyValue': np.random.uniform(100_000, 1_000_000, n),
        'amountPaidOnBuildingClaim': np.where(
            np.random.random(n) < 0.15,
            np.random.lognormal(9, 1.5, n),
            0
        ),
        'had_claim': np.random.randint(0, 2, n),
        'peril_category': np.random.choice(['hurricane', 'flood', 'coastal_flood', 'other'], n),
        'hurricane_category': np.random.choice([np.nan, 1, 2, 3, 4], n),
        'storm_property_damage_usd': np.random.uniform(0, 1e9, n),
        'totalContentsInsuranceCoverage': np.where(
            np.random.random(n) < 0.5,
            np.random.uniform(10_000, 200_000, n),
            0
        ),
        'latitude': np.random.uniform(25, 45, n),
        'longitude': np.random.uniform(-95, -70, n),
        'reportedState': np.random.choice(['FL', 'TX', 'LA', 'NC', 'NY'], n),
    })

    pipeline = FeaturePipeline()
    X, y_sev, y_freq = pipeline.fit_transform(test_data)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"\nFrequency positives: {y_freq.sum()} / {len(y_freq)}")
    print(f"Severity records (non-NaN): {y_sev.notna().sum()} / {len(y_sev)}")
    print(f"\nFeature stats:\n{X.describe().round(3)}")
    print("\n✅ Feature engineering pipeline working")
