"""
Data Ingestion Module — OpenFEMA NFIP + NOAA Storm Events

Pulls historical FEMA NFIP claims and policy data via the OpenFEMA API,
then joins to NOAA Storm Events to associate losses with named storm events.

Key outputs:
    - claims_df:   Raw NFIP claims with flood zone, building, and loss columns
    - policies_df: NFIP policies in force for loss ratio calculation
    - events_df:   NOAA storm events joined to claim dates/locations

Usage:
    from data_ingestion import NFIPClient, NOAAClient, ClaimsEventJoiner

    client = NFIPClient()
    claims = client.fetch_claims(state='TX', limit=50_000)

    noaa = NOAAClient()
    storms = noaa.fetch_storm_events(years=range(2010, 2024), event_type='Hurricane')

    joiner = ClaimsEventJoiner()
    enriched = joiner.join(claims, storms)
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

OPENFEMA_BASE_URL = "https://www.fema.gov/api/open/v2"

# OpenFEMA dataset identifiers
NFIP_CLAIMS_DATASET   = "FimaNfipClaims"
NFIP_POLICIES_DATASET = "FimaNfipPolicies"

# NOAA Storm Events FTP
NOAA_STORM_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"

# Flood zone categories used throughout the project
FLOOD_ZONE_CATEGORIES = {
    # High-risk zones (Special Flood Hazard Areas)
    'A': 'high_risk',
    'AE': 'high_risk',
    'AH': 'high_risk',
    'AO': 'high_risk',
    'AR': 'high_risk',
    'A99': 'high_risk',
    # Coastal high-risk zones (velocity/wave action)
    'V': 'coastal_high_risk',
    'VE': 'coastal_high_risk',
    # Moderate-risk zones
    'B': 'moderate_risk',
    'X': 'moderate_risk',
    'X_SHADED': 'moderate_risk',
    # Minimal-risk zones
    'C': 'minimal_risk',
    'X_UNSHADED': 'minimal_risk',
    # Unknown
    'D': 'unknown',
}

# NFIP claims columns we care about (API v2 field names)
CLAIMS_COLUMNS = [
    'dateOfLoss',
    'yearOfLoss',
    'reportedCity',
    'state',                                  # API v2 name (was 'reportedState' in v1)
    'countyCode',
    'floodZoneCurrent',                       # API v2 name (was 'floodZone' in v1)
    'occupancyType',
    'buildingDamageAmount',
    'contentsDamageAmount',
    'amountPaidOnBuildingClaim',
    'amountPaidOnContentsClaim',
    'amountPaidOnIncreasedCostOfComplianceClaim',
    'buildingPropertyValue',
    'totalBuildingInsuranceCoverage',
    'totalContentsInsuranceCoverage',
    'numberOfFloorsInTheInsuredBuilding',     # API v2 name (was 'numberOfFloorsInInsuredBuilding' in v1)
    'originalConstructionDate',
    'originalNBDate',
    'elevationCertificateIndicator',
    'basementEnclosureCrawlspaceType',
    'postFIRMConstructionIndicator',
    'latitude',
    'longitude',
]

# Rename API v2 field names → internal names used throughout the project
_CLAIMS_API_RENAMES = {
    'state':                                'reportedState',
    'floodZoneCurrent':                     'floodZone',
    'numberOfFloorsInTheInsuredBuilding':    'numberOfFloorsInInsuredBuilding',
}

# NFIP policies columns we care about (API v2 field names)
POLICIES_COLUMNS = [
    'policyEffectiveDate',
    'policyTerminationDate',
    'floodZoneCurrent',                       # API v2 name
    'occupancyType',
    'propertyState',                          # API v2 name (was 'reportedState')
    'countyCode',
    'totalBuildingInsuranceCoverage',
    'totalContentsInsuranceCoverage',
    'buildingReplacementCost',                # API v2 name (was 'buildingPropertyValue')
    'numberOfFloorsInInsuredBuilding',
    'originalConstructionDate',
    'elevationCertificateIndicator',
    'basementEnclosureCrawlspaceType',
    'latitude',
    'longitude',
    'totalInsurancePremiumOfThePolicy',       # API v2 name (was 'annualizedPremiumRate')
]

# Rename API v2 field names → internal names for policies
_POLICIES_API_RENAMES = {
    'propertyState':                        'reportedState',
    'floodZoneCurrent':                     'floodZone',
    'buildingReplacementCost':              'buildingPropertyValue',
    'totalInsurancePremiumOfThePolicy':     'annualizedPremiumRate',
}


# =============================================================================
# HTTP SESSION FACTORY (retry + backoff)
# =============================================================================

def _make_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Create a requests Session with automatic retry and exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        'User-Agent': 'MarginalPML-Research/1.0 (academic portfolio project)',
        'Accept': 'application/json',
    })
    return session


# =============================================================================
# OPENFEMA NFIP CLIENT
# =============================================================================

class NFIPClient:
    """
    Client for the OpenFEMA NFIP Claims and Policies API.

    The OpenFEMA API returns paginated JSON. This client handles:
    - Automatic pagination (up to the requested limit)
    - Column filtering to only pull what we need
    - State and year filtering
    - Rate limiting (respects 429 responses)

    OpenFEMA API docs: https://www.fema.gov/about/openfema/api

    Args:
        base_url: OpenFEMA API base URL
        page_size: Records per API page (max 10,000)
        request_delay: Seconds to wait between pages (be polite to the API)
    """

    def __init__(
        self,
        base_url: str = OPENFEMA_BASE_URL,
        page_size: int = 10_000,
        request_delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip('/')
        self.page_size = min(page_size, 10_000)
        self.request_delay = request_delay
        self._session = _make_session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_claims(
        self,
        state: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        flood_zones: Optional[List[str]] = None,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch NFIP claims from OpenFEMA.

        Args:
            state:       Two-letter state abbreviation (e.g., 'TX', 'FL')
            start_year:  Filter claims on or after this year
            end_year:    Filter claims on or before this year
            flood_zones: List of flood zones to include (e.g., ['AE', 'VE'])
            limit:       Maximum number of records to pull (None = all)
            columns:     Columns to return (None = standard claims columns)
            cache_path:  If provided, save/load from this CSV path

        Returns:
            pd.DataFrame of NFIP claims
        """
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading claims from cache: {cache_path}")
            return pd.read_csv(cache_path, low_memory=False)

        cols = columns or CLAIMS_COLUMNS
        filters = self._build_filters(
            state=state,
            start_year=start_year,
            end_year=end_year,
            flood_zones=flood_zones
        )

        df = self._paginate(
            dataset=NFIP_CLAIMS_DATASET,
            select=cols,
            filters=filters,
            limit=limit,
        )

        df = self._clean_claims(df)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(df):,} claims to {cache_path}")

        return df

    def fetch_policies(
        self,
        state: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch NFIP policies from OpenFEMA.

        Args:
            state:      Two-letter state abbreviation
            start_year: Filter policies effective on or after this year
            end_year:   Filter policies effective on or before this year
            limit:      Maximum number of records to pull
            columns:    Columns to return
            cache_path: If provided, save/load from this CSV path

        Returns:
            pd.DataFrame of NFIP policies
        """
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading policies from cache: {cache_path}")
            return pd.read_csv(cache_path, low_memory=False)

        cols = columns or POLICIES_COLUMNS
        filters = self._build_filters(
            state=state,
            start_year=start_year,
            end_year=end_year,
            state_field='propertyState',
        )

        df = self._paginate(
            dataset=NFIP_POLICIES_DATASET,
            select=cols,
            filters=filters,
            limit=limit,
        )

        df = self._clean_policies(df)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(df):,} policies to {cache_path}")

        return df

    def get_record_count(
        self,
        dataset: str,
        state: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> int:
        """Get total record count for a dataset with filters (no data download)."""
        filters = self._build_filters(
            state=state,
            start_year=start_year,
            end_year=end_year
        )

        url = f"{self.base_url}/{dataset}"
        params = {
            '$top': 1,
            '$inlinecount': 'allpages',
            '$select': 'id',
        }
        if filters:
            params['$filter'] = filters

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # OpenFEMA returns count in metadata
            meta = data.get('metadata', {})
            return int(meta.get('count', 0))
        except Exception as e:
            logger.warning(f"Could not get record count: {e}")
            return -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filters(
        self,
        state: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        flood_zones: Optional[List[str]] = None,
        state_field: str = 'state',
    ) -> str:
        """Build OpenFEMA OData-style filter string.

        Args:
            state_field: API field name for state filtering.
                         Claims use 'state', Policies use 'propertyState'.
        """
        parts = []

        if state:
            parts.append(f"{state_field} eq '{state.upper()}'")

        if start_year:
            parts.append(f"yearOfLoss ge {start_year}")

        if end_year:
            parts.append(f"yearOfLoss le {end_year}")

        if flood_zones:
            zone_filter = " or ".join(f"floodZoneCurrent eq '{z}'" for z in flood_zones)
            parts.append(f"({zone_filter})")

        return " and ".join(parts)

    def _paginate(
        self,
        dataset: str,
        select: List[str],
        filters: str,
        limit: Optional[int],
    ) -> pd.DataFrame:
        """
        Paginate through OpenFEMA API results.

        OpenFEMA uses $skip/$top pagination with $select and $filter.
        """
        url = f"{self.base_url}/{dataset}"
        all_records = []
        skip = 0
        fetched = 0

        logger.info(f"Fetching {dataset}...")
        if filters:
            logger.info(f"  Filters: {filters}")

        while True:
            # Determine page size for this request
            if limit is not None:
                remaining = limit - fetched
                if remaining <= 0:
                    break
                page_size = min(self.page_size, remaining)
            else:
                page_size = self.page_size

            params = {
                '$top': page_size,
                '$skip': skip,
                '$select': ','.join(select),
                '$format': 'json',
            }
            if filters:
                params['$filter'] = filters

            try:
                resp = self._session.get(url, params=params, timeout=60)

                if resp.status_code == 429:
                    logger.warning("Rate limited. Waiting 10s...")
                    time.sleep(10)
                    continue

                resp.raise_for_status()
                data = resp.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"API error at skip={skip}: {e}")
                break
            except ValueError as e:
                logger.error(f"JSON decode error at skip={skip}: {e}")
                break

            # Parse response — OpenFEMA wraps records in dataset-named key
            # Try both exact name and camelCase (API v2 uses PascalCase keys)
            records = (
                data.get(dataset)
                or data.get(dataset[0].lower() + dataset[1:])
                or data.get('data', [])
            )

            if not records:
                logger.info(f"  No more records at skip={skip}. Done.")
                break

            all_records.extend(records)
            fetched += len(records)
            skip += len(records)

            logger.info(f"  Fetched {fetched:,} records...")

            # If we got fewer records than requested, we've reached the end
            if len(records) < page_size:
                break

            # Polite delay
            if self.request_delay > 0:
                time.sleep(self.request_delay)

        if not all_records:
            logger.warning("No records returned from API")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info(f"Total fetched: {len(df):,} records")
        return df

    def _clean_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize claims DataFrame."""
        if df.empty:
            return df

        # Rename API v2 field names → internal names used by the rest of the project
        df = df.rename(columns=_CLAIMS_API_RENAMES)

        # Date parsing
        for date_col in ['dateOfLoss', 'originalConstructionDate', 'originalNBDate']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Numeric columns
        numeric_cols = [
            'amountPaidOnBuildingClaim',
            'amountPaidOnContentsClaim',
            'amountPaidOnIncreasedCostOfComplianceClaim',
            'buildingDamageAmount',
            'contentsDamageAmount',
            'buildingPropertyValue',
            'totalBuildingInsuranceCoverage',
            'totalContentsInsuranceCoverage',
            'numberOfFloorsInInsuredBuilding',
            'latitude',
            'longitude',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Year of loss
        if 'yearOfLoss' not in df.columns and 'dateOfLoss' in df.columns:
            df['yearOfLoss'] = df['dateOfLoss'].dt.year

        # Binary claim flag
        df['had_claim'] = (
            df.get('amountPaidOnBuildingClaim', pd.Series(0)).fillna(0) > 0
        ).astype(int)

        # Total paid
        df['total_paid'] = (
            df.get('amountPaidOnBuildingClaim', 0).fillna(0) +
            df.get('amountPaidOnContentsClaim', 0).fillna(0)
        )

        # Flood zone category
        if 'floodZone' in df.columns:
            df['floodZone'] = df['floodZone'].str.strip().str.upper()
            df['flood_zone_category'] = df['floodZone'].map(FLOOD_ZONE_CATEGORIES).fillna('unknown')

        # Construction era
        if 'originalConstructionDate' in df.columns:
            df['construction_year'] = df['originalConstructionDate'].dt.year
            df['construction_era'] = pd.cut(
                df['construction_year'].fillna(1980),
                bins=[0, 1970, 1980, 1990, 2000, 2010, 2024],
                labels=['pre_1970', '1970s', '1980s', '1990s', '2000s', '2010s'],
                right=True
            )

        # Loss ratio (building)
        if 'totalBuildingInsuranceCoverage' in df.columns and 'amountPaidOnBuildingClaim' in df.columns:
            coverage = df['totalBuildingInsuranceCoverage'].replace(0, np.nan)
            df['building_loss_ratio'] = df['amountPaidOnBuildingClaim'].fillna(0) / coverage

        logger.info(f"Claims cleaned: {len(df):,} records")
        return df

    def _clean_policies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize policies DataFrame."""
        if df.empty:
            return df

        # Rename API v2 field names → internal names used by the rest of the project
        df = df.rename(columns=_POLICIES_API_RENAMES)

        # Date parsing
        for date_col in ['policyEffectiveDate', 'policyTerminationDate', 'originalConstructionDate']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Numeric columns
        numeric_cols = [
            'totalBuildingInsuranceCoverage',
            'totalContentsInsuranceCoverage',
            'buildingPropertyValue',
            'annualizedPremiumRate',
            'numberOfFloorsInInsuredBuilding',
            'latitude',
            'longitude',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'floodZone' in df.columns:
            df['floodZone'] = df['floodZone'].str.strip().str.upper()
            df['flood_zone_category'] = df['floodZone'].map(FLOOD_ZONE_CATEGORIES).fillna('unknown')

        logger.info(f"Policies cleaned: {len(df):,} records")
        return df


# =============================================================================
# NOAA STORM EVENTS CLIENT
# =============================================================================

class NOAAClient:
    """
    Client for NOAA Storm Events data.

    Downloads CSV files from NOAA's public FTP/HTTP server. Storm Events data
    is organized by year with separate detail and fatalities files.

    NOAA Storm Events docs: https://www.ncdc.noaa.gov/stormevents/

    Args:
        base_url: NOAA Storm Events base URL
        cache_dir: Directory to cache downloaded files
    """

    # Column names from NOAA Storm Events detail CSV
    DETAIL_COLUMNS = [
        'YEAR', 'MONTH_NAME', 'EVENT_TYPE', 'STATE', 'STATE_FIPS',
        'CZ_TYPE', 'CZ_NAME', 'BEGIN_DATE_TIME', 'END_DATE_TIME',
        'INJURIES_DIRECT', 'INJURIES_INDIRECT',
        'DEATHS_DIRECT', 'DEATHS_INDIRECT',
        'DAMAGE_PROPERTY', 'DAMAGE_CROPS',
        'SOURCE', 'MAGNITUDE', 'MAGNITUDE_TYPE',
        'CATEGORY', 'TOR_F_SCALE',
        'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON',
        'EVENT_ID', 'EPISODE_ID',
    ]

    # Map NOAA event types to internal peril categories
    PERIL_MAPPING = {
        'Hurricane (Typhoon)':   'hurricane',
        'Hurricane':             'hurricane',
        'Tropical Storm':        'tropical_storm',
        'Storm Surge/Tide':      'storm_surge',
        'Coastal Flood':         'coastal_flood',
        'Flash Flood':           'flood',
        'Flood':                 'flood',
        'Lakeshore Flood':       'flood',
        'Heavy Rain':            'heavy_rain',
        'Tornado':               'tornado',
        'Hail':                  'hail',
        'Thunderstorm Wind':     'convective',
        'Marine Thunderstorm Wind': 'convective',
        'High Wind':             'wind',
        'Winter Storm':          'winter',
        'Blizzard':              'winter',
        'Ice Storm':             'winter',
        'Wildfire':              'wildfire',
    }

    def __init__(
        self,
        base_url: str = NOAA_STORM_BASE_URL,
        cache_dir: str = "data/raw/noaa",
    ):
        self.base_url = base_url.rstrip('/')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = _make_session()

    def fetch_storm_events(
        self,
        years: Optional[range] = None,
        event_types: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch NOAA Storm Events for specified years.

        Args:
            years:       Range of years (default: 2000-2023)
            event_types: List of event types to filter (e.g., ['Hurricane', 'Flood'])
            states:      List of state names to filter (e.g., ['FLORIDA', 'TEXAS'])
            cache_path:  If provided, save/load from this CSV path

        Returns:
            pd.DataFrame of storm events
        """
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading storm events from cache: {cache_path}")
            return pd.read_csv(cache_path, low_memory=False)

        if years is None:
            years = range(2000, 2024)

        all_events = []

        for year in years:
            df = self._fetch_year(year)
            if df is not None and not df.empty:
                all_events.append(df)

        if not all_events:
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        combined = self._clean_storm_events(combined)

        # Apply filters
        if event_types:
            combined = combined[combined['EVENT_TYPE'].isin(event_types)]
        if states:
            states_upper = [s.upper() for s in states]
            combined = combined[combined['STATE'].str.upper().isin(states_upper)]

        combined = combined.reset_index(drop=True)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(combined):,} storm events to {cache_path}")

        return combined

    def _fetch_year(self, year: int) -> Optional[pd.DataFrame]:
        """Fetch storm events for a single year."""
        # NOAA filename pattern: StormEvents_details-ftp_v1.0_dYYYY_cYYYY....csv.gz
        # We search for the matching file via a directory listing approach
        filename_pattern = f"StormEvents_details-ftp_v1.0_d{year}_c"

        # Check local cache first
        cache_files = list(self.cache_dir.glob(f"*d{year}*.csv.gz"))
        if cache_files:
            try:
                return pd.read_csv(cache_files[0], compression='gzip', low_memory=False)
            except Exception as e:
                logger.warning(f"Error reading cached file for {year}: {e}")

        # Try to find the file via NOAA directory listing
        try:
            resp = self._session.get(self.base_url + "/", timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Could not list NOAA directory for year {year}")
                return None

            # Parse the directory listing to find the right file
            import re
            matches = re.findall(
                rf'(StormEvents_details-ftp_v1\.0_d{year}_c\d{{8}}\.csv\.gz)',
                resp.text
            )

            if not matches:
                logger.warning(f"No storm events file found for year {year}")
                return None

            # Use the most recent version (largest creation date in filename)
            filename = sorted(matches)[-1]
            file_url = f"{self.base_url}/{filename}"

            logger.info(f"  Downloading {filename}...")
            file_resp = self._session.get(file_url, timeout=120, stream=True)
            file_resp.raise_for_status()

            # Save to cache
            local_path = self.cache_dir / filename
            with open(local_path, 'wb') as f:
                for chunk in file_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            return pd.read_csv(local_path, compression='gzip', low_memory=False)

        except Exception as e:
            logger.error(f"Error fetching storm events for {year}: {e}")
            return None

    def _clean_storm_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize storm events DataFrame."""
        if df.empty:
            return df

        # Date parsing
        for date_col in ['BEGIN_DATE_TIME', 'END_DATE_TIME']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce',
                                               format='%d-%b-%y %H:%M:%S')

        # Extract begin date components
        if 'BEGIN_DATE_TIME' in df.columns:
            df['event_date'] = df['BEGIN_DATE_TIME'].dt.date
            df['event_year'] = df['BEGIN_DATE_TIME'].dt.year
            df['event_month'] = df['BEGIN_DATE_TIME'].dt.month

        # Parse damage strings like "5.0K", "1.2M", "50B"
        for col in ['DAMAGE_PROPERTY', 'DAMAGE_CROPS']:
            if col in df.columns:
                df[col + '_USD'] = df[col].apply(self._parse_damage_string)

        # Add peril category
        if 'EVENT_TYPE' in df.columns:
            df['peril_category'] = df['EVENT_TYPE'].map(self.PERIL_MAPPING).fillna('other')

        # Hurricane category (from CATEGORY column)
        if 'CATEGORY' in df.columns:
            df['hurricane_category'] = pd.to_numeric(df['CATEGORY'], errors='coerce')

        # Numeric coordinates
        for col in ['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Storm events cleaned: {len(df):,} records across {df['YEAR'].nunique() if 'YEAR' in df.columns else '?'} years")
        return df

    @staticmethod
    def _parse_damage_string(val) -> float:
        """Parse NOAA damage strings: '5.0K' → 5000, '1.2M' → 1200000."""
        if pd.isna(val) or val == '0' or val == '':
            return 0.0
        val = str(val).strip().upper()
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
        if val[-1] in multipliers:
            try:
                return float(val[:-1]) * multipliers[val[-1]]
            except ValueError:
                return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0


# =============================================================================
# CLAIMS + STORM EVENT JOINER
# =============================================================================

class ClaimsEventJoiner:
    """
    Join NFIP claims to NOAA storm events.

    Associates individual flood claims with the named storm event that caused
    them, enabling hurricane category, storm intensity, and event-level features
    to be added to each claim record.

    Matching strategy:
        1. Date window: claim date within ±30 days of storm event
        2. Geographic: claim state matches storm state
        3. Priority: prefer most damaging storm event in window

    This enriched dataset is the primary input to the feature engineering pipeline.
    """

    def __init__(
        self,
        date_window_days: int = 30,
        min_storm_damage_usd: float = 0.0,
    ):
        """
        Args:
            date_window_days:    How many days either side of a storm to match claims
            min_storm_damage_usd: Minimum property damage to consider a storm
        """
        self.date_window_days = date_window_days
        self.min_storm_damage_usd = min_storm_damage_usd

    def join(
        self,
        claims: pd.DataFrame,
        storms: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Join claims to storm events.

        Args:
            claims: NFIP claims DataFrame (from NFIPClient.fetch_claims)
            storms: NOAA storm events DataFrame (from NOAAClient.fetch_storm_events)
            verbose: Print progress

        Returns:
            Claims DataFrame enriched with storm event columns:
            - event_type
            - peril_category
            - hurricane_category
            - storm_property_damage_usd
            - storm_begin_date
            - matched_storm_id
        """
        # Storm columns that will be added to claims
        storm_cols = [
            'matched_storm_id', 'EVENT_TYPE', 'peril_category',
            'hurricane_category', 'storm_property_damage_usd', 'storm_begin_date',
        ]

        if claims.empty or storms.empty:
            logger.warning("Empty input to ClaimsEventJoiner.join()")
            # Still add the expected columns (as NaN) so downstream code doesn't break
            for col in storm_cols:
                claims[col] = np.nan
            return claims

        # Ensure date columns exist
        claims = claims.copy()
        storms = storms.copy()

        if 'dateOfLoss' not in claims.columns:
            logger.error("Claims missing 'dateOfLoss' column")
            return claims

        claims['dateOfLoss'] = pd.to_datetime(claims['dateOfLoss'], errors='coerce')
        # Strip timezone info so both sides are tz-naive (avoids subtract error)
        if hasattr(claims['dateOfLoss'].dtype, 'tz') and claims['dateOfLoss'].dtype.tz is not None:
            claims['dateOfLoss'] = claims['dateOfLoss'].dt.tz_localize(None)

        if 'event_date' not in storms.columns:
            if 'BEGIN_DATE_TIME' in storms.columns:
                storms['event_date'] = pd.to_datetime(storms['BEGIN_DATE_TIME'], errors='coerce').dt.date
            else:
                logger.error("Storms missing date column")
                return claims

        storms['event_date'] = pd.to_datetime(storms['event_date'], errors='coerce')
        if hasattr(storms['event_date'].dtype, 'tz') and storms['event_date'].dtype.tz is not None:
            storms['event_date'] = storms['event_date'].dt.tz_localize(None)

        # Filter storms by minimum damage
        if self.min_storm_damage_usd > 0 and 'DAMAGE_PROPERTY_USD' in storms.columns:
            storms = storms[storms['DAMAGE_PROPERTY_USD'] >= self.min_storm_damage_usd]

        # Standardize state columns
        if 'reportedState' in claims.columns:
            claims['_state'] = claims['reportedState'].str.strip().str.upper()
        if 'STATE' in storms.columns:
            storms['_state'] = storms['STATE'].str.strip().str.upper()
            # NOAA uses full state names, convert to abbreviations
            storms['_state'] = storms['_state'].map(_NOAA_STATE_TO_ABBREV).fillna(storms['_state'])

        if verbose:
            logger.info(f"Joining {len(claims):,} claims to {len(storms):,} storm events...")

        # Build storm lookup: {(state, year): [storm records]}
        # This avoids O(n*m) date comparisons for all claims
        storm_lookup = self._build_storm_lookup(storms)

        # Match each claim to the nearest storm event
        matched_cols = [
            'matched_storm_id', 'EVENT_TYPE', 'peril_category',
            'hurricane_category', 'storm_property_damage_usd', 'storm_begin_date',
        ]

        claim_matches = claims.apply(
            lambda row: self._match_claim_to_storm(row, storm_lookup),
            axis=1,
            result_type='expand'
        )
        claim_matches.columns = matched_cols

        result = pd.concat([claims.reset_index(drop=True), claim_matches], axis=1)

        matched_count = result['matched_storm_id'].notna().sum()
        match_rate = matched_count / len(result) * 100

        if verbose:
            logger.info(f"Matched {matched_count:,} of {len(result):,} claims "
                        f"({match_rate:.1f}%) to storm events")

        return result

    def _build_storm_lookup(
        self, storms: pd.DataFrame
    ) -> Dict[Tuple[str, int], List[Dict]]:
        """Build a dict keyed by (state, year) for fast claim matching."""
        lookup: Dict[Tuple[str, int], List[Dict]] = {}

        for _, row in storms.iterrows():
            if pd.isna(row.get('event_date')):
                continue

            state = str(row.get('_state', ''))
            year = int(row['event_date'].year) if hasattr(row['event_date'], 'year') else int(row.get('YEAR', 0))
            key = (state, year)

            storm_record = {
                'storm_id': row.get('EVENT_ID', ''),
                'event_type': row.get('EVENT_TYPE', ''),
                'peril_category': row.get('peril_category', 'other'),
                'hurricane_category': row.get('hurricane_category', np.nan),
                'damage_property_usd': row.get('DAMAGE_PROPERTY_USD', 0.0),
                'event_date': row['event_date'],
            }

            lookup.setdefault(key, []).append(storm_record)

        return lookup

    def _match_claim_to_storm(
        self,
        claim: pd.Series,
        storm_lookup: Dict,
    ) -> Tuple:
        """Match a single claim to the nearest storm event. Returns a tuple of storm fields."""
        null_result = (None, None, None, np.nan, 0.0, None)

        loss_date = claim.get('dateOfLoss')
        if pd.isna(loss_date):
            return null_result

        state = str(claim.get('_state', ''))
        year = int(loss_date.year)

        # Check this year and the adjacent years (storms can straddle year boundaries)
        candidates = []
        for y in [year - 1, year, year + 1]:
            candidates.extend(storm_lookup.get((state, y), []))

        if not candidates:
            return null_result

        # Find storms within the date window
        window = pd.Timedelta(days=self.date_window_days)
        in_window = [
            s for s in candidates
            if abs((loss_date - s['event_date']).days) <= self.date_window_days
        ]

        if not in_window:
            return null_result

        # Pick the storm with the most damage in window (best proxy for the event)
        best = max(in_window, key=lambda s: s.get('damage_property_usd', 0))

        return (
            best['storm_id'],
            best['event_type'],
            best['peril_category'],
            best['hurricane_category'],
            best['damage_property_usd'],
            best['event_date'],
        )


# =============================================================================
# STATE NAME → ABBREVIATION MAP (NOAA uses full names)
# =============================================================================

_NOAA_STATE_TO_ABBREV = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC',
    # Territories
    'PUERTO RICO': 'PR', 'VIRGIN ISLANDS': 'VI', 'GUAM': 'GU',
    'AMERICAN SAMOA': 'AS', 'NORTHERN MARIANA ISLANDS': 'MP',
}

# Reverse mapping: abbreviation → full state name (for NOAA filtering)
_ABBREV_TO_NOAA_STATE = {v: k for k, v in _NOAA_STATE_TO_ABBREV.items()}


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def load_nfip_data(
    states: Optional[List[str]] = None,
    start_year: int = 2000,
    end_year: int = 2023,
    limit_per_state: Optional[int] = 50_000,
    cache_dir: str = "data/raw",
    join_storms: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    High-level convenience function: fetch + clean + join NFIP claims data.

    Args:
        states:           List of state abbreviations (None = all coastal states)
        start_year:       First year to pull
        end_year:         Last year to pull
        limit_per_state:  Max claims per state
        cache_dir:        Base directory for caching raw files
        join_storms:      Whether to join NOAA storm event data
        verbose:          Print progress

    Returns:
        Enriched claims DataFrame ready for feature engineering.
    """
    if states is None:
        # Focus on coastal states most relevant to flood/hurricane risk
        states = ['FL', 'TX', 'LA', 'NC', 'SC', 'GA', 'VA', 'NJ', 'NY', 'MS']

    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger.info(f"Loading NFIP data for states: {states}, years: {start_year}–{end_year}")

    client = NFIPClient()
    all_claims = []

    for state in states:
        cache_path = f"{cache_dir}/nfip_claims_{state}_{start_year}_{end_year}.csv"

        claims = client.fetch_claims(
            state=state,
            start_year=start_year,
            end_year=end_year,
            limit=limit_per_state,
            cache_path=cache_path,
        )

        if not claims.empty:
            claims['source_state'] = state
            all_claims.append(claims)
            logger.info(f"  {state}: {len(claims):,} claims")

    if not all_claims:
        logger.warning("No claims data fetched")
        return pd.DataFrame()

    combined_claims = pd.concat(all_claims, ignore_index=True)
    logger.info(f"\nTotal claims: {len(combined_claims):,}")

    if join_storms:
        noaa = NOAAClient(cache_dir=f"{cache_dir}/noaa")
        storms = noaa.fetch_storm_events(
            years=range(start_year, end_year + 1),
            event_types=['Hurricane (Typhoon)', 'Hurricane', 'Tropical Storm',
                         'Storm Surge/Tide', 'Coastal Flood', 'Flash Flood', 'Flood'],
            states=[_ABBREV_TO_NOAA_STATE.get(s.upper(), s) for s in states],
            cache_path=f"{cache_dir}/noaa_storms_{start_year}_{end_year}.csv",
        )

        joiner = ClaimsEventJoiner(date_window_days=30, min_storm_damage_usd=1_000)
        combined_claims = joiner.join(combined_claims, storms, verbose=verbose)

    return combined_claims


# =============================================================================
# FILING RATE COMPUTATION
# =============================================================================

def compute_filing_rates(
    claims: pd.DataFrame,
    client: Optional[NFIPClient] = None,
    states: Optional[List[str]] = None,
    start_year: int = 2000,
    end_year: int = 2023,
    policy_limit_per_state: int = 100_000,
    cache_dir: str = "data/raw",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute annual filing rates from OpenFEMA claims and policy counts.

    Filing rate = claims filed per year / policies in force per year,
    stratified by state and flood zone category. This converts the
    frequency model's P(paid | filed) into a true annual claim rate:

        annual_rate = P(paid | filed) × filing_rate

    The function pulls policy counts from OpenFEMA, counts claims per
    (state, flood_zone, year) group, and returns average filing rates.

    Args:
        claims:                  NFIP claims DataFrame (from fetch_claims or load_nfip_data)
        client:                  NFIPClient instance (created if None)
        states:                  States to compute rates for (None = infer from claims)
        start_year:              First year to include
        end_year:                Last year to include
        policy_limit_per_state:  Max policy records to pull per state (for speed)
        cache_dir:               Directory for caching policy data
        cache_path:              If provided, save/load filing rates from this path

    Returns:
        DataFrame with columns: state, flood_zone_category, filing_rate, claims_count, policy_years
    """
    # Check cache first
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading filing rates from cache: {cache_path}")
        return pd.read_parquet(cache_path) if cache_path.endswith('.parquet') else pd.read_csv(cache_path)

    if client is None:
        client = NFIPClient()

    if states is None:
        if 'reportedState' in claims.columns:
            states = claims['reportedState'].dropna().unique().tolist()
        else:
            states = ['FL', 'TX', 'LA', 'NC', 'SC', 'GA', 'NJ', 'NY']

    logger.info(f"Computing filing rates for {len(states)} states, {start_year}–{end_year}...")

    # --- Count claims per (state, flood_zone_category, year) ---
    claims_copy = claims.copy()
    if 'yearOfLoss' not in claims_copy.columns and 'dateOfLoss' in claims_copy.columns:
        claims_copy['yearOfLoss'] = pd.to_datetime(claims_copy['dateOfLoss'], errors='coerce').dt.year

    # Ensure flood_zone_category exists
    if 'flood_zone_category' not in claims_copy.columns and 'floodZone' in claims_copy.columns:
        claims_copy['flood_zone_category'] = (
            claims_copy['floodZone'].str.strip().str.upper()
            .map(FLOOD_ZONE_CATEGORIES).fillna('unknown')
        )

    claims_filtered = claims_copy[
        (claims_copy['yearOfLoss'] >= start_year) &
        (claims_copy['yearOfLoss'] <= end_year)
    ].copy()

    if 'reportedState' in claims_filtered.columns:
        claims_filtered['_state'] = claims_filtered['reportedState'].str.strip().str.upper()
    else:
        claims_filtered['_state'] = 'XX'

    claim_counts = (
        claims_filtered
        .groupby(['_state', 'flood_zone_category'])
        .size()
        .reset_index(name='claims_count')
    )
    n_years = end_year - start_year + 1
    claim_counts['claims_per_year'] = claim_counts['claims_count'] / n_years

    # --- Pull policy counts per (state, flood_zone_category) ---
    logger.info("Pulling NFIP policy data for filing rate computation...")
    all_policies = []

    for state in states:
        state_upper = state.upper()
        pol_cache = f"{cache_dir}/nfip_policies_{state_upper}_{start_year}_{end_year}.csv"

        policies = client.fetch_policies(
            state=state_upper,
            start_year=start_year,
            end_year=end_year,
            limit=policy_limit_per_state,
            cache_path=pol_cache,
        )

        if not policies.empty:
            policies['_state'] = state_upper

            # Get total count from API for extrapolation
            total_count = client.get_record_count(
                NFIP_POLICIES_DATASET,
                state=state_upper,
                start_year=start_year,
                end_year=end_year,
            )
            sample_count = len(policies)

            # Extrapolation factor: if we sampled 100K of 500K, multiply by 5
            if total_count > 0 and sample_count > 0:
                extrap_factor = total_count / sample_count
            else:
                extrap_factor = 1.0

            policies['_extrap_factor'] = extrap_factor
            all_policies.append(policies)
            logger.info(f"  {state_upper}: {sample_count:,} policies sampled "
                        f"(total: {total_count:,}, extrap: {extrap_factor:.1f}x)")

    if not all_policies:
        logger.warning("No policy data retrieved. Using literature-based fallback rates.")
        return _fallback_filing_rates()

    combined_policies = pd.concat(all_policies, ignore_index=True)

    # Ensure flood_zone_category in policies
    if 'flood_zone_category' not in combined_policies.columns and 'floodZone' in combined_policies.columns:
        combined_policies['flood_zone_category'] = (
            combined_policies['floodZone'].str.strip().str.upper()
            .map(FLOOD_ZONE_CATEGORIES).fillna('unknown')
        )
    elif 'flood_zone_category' not in combined_policies.columns:
        combined_policies['flood_zone_category'] = 'unknown'

    # Count policies per (state, zone), apply extrapolation factor
    policy_counts = (
        combined_policies
        .groupby(['_state', 'flood_zone_category'])
        .agg(
            sampled_count=('_state', 'size'),
            extrap_factor=('_extrap_factor', 'first'),
        )
        .reset_index()
    )
    policy_counts['estimated_total_policies'] = (
        policy_counts['sampled_count'] * policy_counts['extrap_factor']
    )
    policy_counts['policies_per_year'] = policy_counts['estimated_total_policies'] / n_years

    # --- Join and compute filing rates ---
    merged = pd.merge(
        claim_counts, policy_counts,
        on=['_state', 'flood_zone_category'],
        how='outer',
    ).fillna(0)

    # Filing rate = claims per year / policies per year
    merged['filing_rate'] = np.where(
        merged['policies_per_year'] > 0,
        (merged['claims_per_year'] / merged['policies_per_year']).clip(0.001, 0.30),
        0.035,  # National average fallback
    )

    result = merged[['_state', 'flood_zone_category', 'filing_rate',
                      'claims_count', 'policies_per_year']].copy()
    result = result.rename(columns={'_state': 'state'})

    # Add overall averages by flood zone (for properties without state match)
    zone_avg = (
        result.groupby('flood_zone_category')
        .agg(filing_rate=('filing_rate', 'mean'))
        .reset_index()
    )
    zone_avg['state'] = '_ALL'
    zone_avg['claims_count'] = 0
    zone_avg['policies_per_year'] = 0
    result = pd.concat([result, zone_avg], ignore_index=True)

    logger.info(f"\nFiling rates computed ({len(result)} rows):")
    for _, row in result[result['state'] == '_ALL'].iterrows():
        logger.info(f"  {row['flood_zone_category']:25s} → {row['filing_rate']:.3f}")

    # Cache result
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        if cache_path.endswith('.parquet'):
            result.to_parquet(cache_path, index=False)
        else:
            result.to_csv(cache_path, index=False)
        logger.info(f"Filing rates saved to {cache_path}")

    return result


def _fallback_filing_rates() -> pd.DataFrame:
    """
    Literature-based filing rates when OpenFEMA policy data is unavailable.

    These are approximate NFIP filing rates based on published data:
    - FEMA actuarial reports (2000–2023 averages)
    - Congressional Research Service NFIP reports
    """
    rates = [
        ('_ALL', 'coastal_high_risk', 0.085),
        ('_ALL', 'high_risk',         0.042),
        ('_ALL', 'moderate_risk',     0.015),
        ('_ALL', 'minimal_risk',      0.008),
        ('_ALL', 'unknown',           0.035),
    ]
    return pd.DataFrame(rates, columns=['state', 'flood_zone_category', 'filing_rate'])


def load_filing_rates(path: str = "data/raw/filing_rates.parquet") -> pd.DataFrame:
    """
    Load cached filing rates, or return literature-based fallback.

    Args:
        path: Path to cached filing rates file

    Returns:
        DataFrame with state, flood_zone_category, filing_rate columns
    """
    p = Path(path)
    if p.exists():
        logger.info(f"Loading filing rates from {path}")
        return pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    else:
        logger.warning(f"Filing rates not found at {path}. Using literature-based fallback.")
        return _fallback_filing_rates()


# =============================================================================
# MAIN (quick smoke test)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    print("=" * 60)
    print("NFIP DATA INGESTION - Quick Test")
    print("=" * 60)

    # Test record count (no download)
    client = NFIPClient()
    count = client.get_record_count('FimaNfipClaims', state='FL', start_year=2017, end_year=2020)
    print(f"\nFlorida claims 2017-2020: {count:,} records")

    # Fetch a small sample
    print("\nFetching 100 FL claims as sample...")
    sample = client.fetch_claims(state='FL', start_year=2020, end_year=2022, limit=100)
    print(f"Sample shape: {sample.shape}")
    if not sample.empty:
        print(f"Columns: {list(sample.columns)}")
        print(f"Loss range: ${sample['amountPaidOnBuildingClaim'].min():,.0f} – "
              f"${sample['amountPaidOnBuildingClaim'].max():,.0f}")

    print("\n✅ Data ingestion module working")
