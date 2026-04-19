"""
Marginal PML Pricing Engine - Vectorized Math Kernel

This module contains the core mathematical logic for calculating Marginal PML
using the Method of Moments fitting to a Beta Distribution.

No SQL or database dependencies - pure NumPy/Pandas operations.

Domain Knowledge:
- Data Source: Event Loss Tables (ELT)
- Primary Math: Aggregate Portfolio and New Account by EventID
- Aggregation Rules:
    - Means are additive
    - Max Exposure is additive
    - Correlated Standard Deviation (SD_C) is additive
    - Independent Standard Deviation (SD_I) is root of sum of squares: sqrt(sum(SD_I^2))
- Goal: Calculate Marginal Impact = PML_{Portfolio+Account} - PML_{Portfolio}
- Distribution: Beta Distribution with Alpha/Beta from Method of Moments
- Frequency: Poisson frequency assumption for AEP calculation
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import math
import warnings
from scipy.stats import beta as beta_dist
from scipy.optimize import brentq


@dataclass
class MomentsConfig:
    """Configuration for moment calculations and PML solving."""
    use_poisson_aep: bool = True  # True = AEP = 1-exp(-λ); False = AEP = λ
    rp_tolerance: float = 0.5  # Convergence tolerance in years
    brent_xtol: float = 100.0  # Absolute tolerance on PML ($)
    brent_rtol: float = 1e-6  # Relative tolerance for Brent's method
    max_iterations: int = 10000  # Maximum iterations for root finding


@dataclass
class CorrelationConfig:
    """Configuration for correlated PML simulation via Gaussian copula.

    The single-factor model uses each event's SD_C / (SD_C + SD_I) ratio
    as the loading on a shared "catastrophe" factor.  When a bad year is
    drawn (high Z_common), all events with high correlated-fraction move
    together — capturing portfolio concentration risk.

    Optionally supply *event_groups* (array of integer group labels, one per
    event) to use a multi-factor model with one shared factor per group.
    """
    n_trials: int = 10_000          # number of Monte-Carlo simulation years
    seed: Optional[int] = 42        # RNG seed (set None for non-deterministic)
    event_groups: Optional[np.ndarray] = None  # multi-factor peril/region groups


def calculate_combined_moments(
    base_elt: pd.DataFrame,
    account_elt: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine a baseline (portfolio) ELT with an account ELT using the Method of Moments.
    
    This function performs an outer join on EventID and applies insurance industry
    aggregation rules to derive combined moment statistics and Beta distribution
    parameters for Marginal PML calculation.
    
    Aggregation Rules:
        - Mean Loss (PERSPVALUE): Additive → Combined = Base + Account
        - Max Exposure (EXPVALUE): Additive → Combined = Base + Account
        - Correlated Std Dev (STDDEVC): Additive → Combined = Base + Account
        - Independent Std Dev (STDDEVI): Root sum of squares → sqrt(Base² + Account²)
    
    Method of Moments for Beta Distribution:
        Given mean μ and std σ on [0, 1]:
        - μ_normalized = Mean / MaxExposure
        - σ_normalized = (SD_I + SD_C) / MaxExposure
        - α = μ²(1-μ)/σ² - μ
        - β = α(1-μ)/μ
    
    Args:
        base_elt: Baseline/Portfolio Event Loss Table with columns:
            - EVENTID: Event identifier
            - RATE: Annual occurrence rate (Poisson λ per event)
            - PERSPVALUE: Expected loss (mean)
            - EXPVALUE: Maximum possible loss
            - STDDEVC: Correlated standard deviation
            - STDDEVI: Independent standard deviation
        
        account_elt: Account Event Loss Table with same columns.
            May contain events not in baseline ("new-to-portfolio" events).
    
    Returns:
        pd.DataFrame ready for BrentQ root-finder with columns:
            - EVENTID: Event identifier
            - Port_Rate: Annual occurrence rate
            - MaxExposure: Combined maximum exposure
            - Alpha: Beta distribution α parameter
            - Beta: Beta distribution β parameter
            - Comb_MeanLoss: Combined mean loss
            - Comb_StdDev: Combined total standard deviation
    
    Notes:
        - Events with invalid parameters (Alpha/Beta <= 0, NaN) are filtered out
        - New-to-portfolio events (in account but not baseline) are included
        - Events where combined exposure would be <= 0 are excluded
    """
    # Validate input columns
    required_base_cols = ['EVENTID', 'RATE', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI']
    required_acct_cols = ['EVENTID', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI']
    
    missing_base = [c for c in required_base_cols if c not in base_elt.columns]
    missing_acct = [c for c in required_acct_cols if c not in account_elt.columns]
    
    if missing_base:
        raise ValueError(f"base_elt missing required columns: {missing_base}")
    if missing_acct:
        raise ValueError(f"account_elt missing required columns: {missing_acct}")
    
    # Handle empty DataFrames
    if base_elt.empty:
        warnings.warn("base_elt is empty - returning empty DataFrame")
        return pd.DataFrame(columns=[
            'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta',
            'Comb_MeanLoss', 'Comb_StdDev'
        ])
    
    # Perform OUTER JOIN on EVENTID to capture:
    # 1. Events in both baseline and account
    # 2. Events only in baseline (account has no exposure)
    # 3. Events only in account (new-to-portfolio)
    combined = base_elt.merge(
        account_elt,
        on='EVENTID',
        how='outer',
        suffixes=('_base', '_acct')
    )
    
    # Fill NaN with 0 for events missing from one side
    # Base columns
    combined['PERSPVALUE_base'] = combined['PERSPVALUE_base'].fillna(0)
    combined['EXPVALUE_base'] = combined['EXPVALUE_base'].fillna(0)
    combined['STDDEVC_base'] = combined['STDDEVC_base'].fillna(0)
    combined['STDDEVI_base'] = combined['STDDEVI_base'].fillna(0)
    combined['RATE'] = combined['RATE'].fillna(0)
    
    # Account columns
    combined['PERSPVALUE_acct'] = combined['PERSPVALUE_acct'].fillna(0)
    combined['EXPVALUE_acct'] = combined['EXPVALUE_acct'].fillna(0)
    combined['STDDEVC_acct'] = combined['STDDEVC_acct'].fillna(0)
    combined['STDDEVI_acct'] = combined['STDDEVI_acct'].fillna(0)
    
    # If account has RATE column, use it for new-to-portfolio events
    if 'RATE_acct' in combined.columns:
        combined['RATE'] = combined['RATE'].where(
            combined['RATE'] > 0,
            combined['RATE_acct'].fillna(0)
        )
    
    # Aggregation Rules (vectorized)
    # 1. Mean Loss: Additive
    combined['Comb_MeanLoss'] = (
        combined['PERSPVALUE_base'] + combined['PERSPVALUE_acct']
    )
    
    # 2. Max Exposure: Additive
    combined['Comb_ExpValue'] = (
        combined['EXPVALUE_base'] + combined['EXPVALUE_acct']
    )
    
    # 3. Correlated Std Dev: Additive
    combined['Comb_StdDevC'] = (
        combined['STDDEVC_base'] + combined['STDDEVC_acct']
    )
    
    # 4. Independent Std Dev: Root of sum of squares
    combined['Comb_StdDevI'] = np.sqrt(
        np.square(combined['STDDEVI_base']) + 
        np.square(combined['STDDEVI_acct'])
    )
    
    # Total standard deviation
    combined['Comb_StdDev'] = combined['Comb_StdDevI'] + combined['Comb_StdDevC']
    
    # Filter: Combined exposure and mean must be positive
    # This removes degenerate events
    valid_mask = (
        (combined['Comb_ExpValue'] > 0) &
        (combined['Comb_MeanLoss'] > 0) &
        (combined['Comb_StdDev'] > 0)
    )
    combined = combined[valid_mask].copy()
    
    if combined.empty:
        warnings.warn("No valid events after filtering - returning empty DataFrame")
        return pd.DataFrame(columns=[
            'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta',
            'Comb_MeanLoss', 'Comb_StdDev'
        ])
    
    # Method of Moments: Derive Beta Distribution Parameters
    # 
    # For a Beta distribution on [0, 1] with mean μ and variance σ²:
    #   α = μ²(1-μ)/σ² - μ
    #   β = α(1-μ)/μ
    #
    # We normalize to [0, MaxExposure]:
    #   μ_norm = MeanLoss / MaxExposure
    #   σ_norm = TotalStdDev / MaxExposure
    
    # Normalized mean (loss ratio)
    combined['BetaDistMeanU'] = (
        combined['Comb_MeanLoss'] / combined['Comb_ExpValue']
    )
    
    # Normalized standard deviation
    combined['BetaDistStdDevO'] = (
        combined['Comb_StdDev'] / combined['Comb_ExpValue']
    )
    
    # Calculate Alpha parameter
    # α = μ²(1-μ)/σ² - μ
    mu = combined['BetaDistMeanU'].values
    sigma = combined['BetaDistStdDevO'].values
    
    # Validity conditions for Method of Moments:
    # - 0 < μ < 1 (mean must be within bounds)
    # - σ > 0 (must have variance)
    # - σ² < μ(1-μ) (variance constraint for valid Beta distribution)
    valid_params = (
        (mu > 0) & (mu < 1) &
        (sigma > 0) &
        (np.square(sigma) < mu * (1 - mu))  # Variance constraint
    )
    
    # Initialize with NaN for invalid cases
    alpha = np.full_like(mu, np.nan)
    beta_param = np.full_like(mu, np.nan)
    
    # Calculate only for valid cases (vectorized)
    if valid_params.any():
        mu_v = mu[valid_params]
        sigma_v = sigma[valid_params]
        sigma_sq = np.square(sigma_v)
        
        # α = μ²(1-μ)/σ² - μ
        alpha[valid_params] = (
            np.square(mu_v) * (1 - mu_v) / sigma_sq - mu_v
        )
        
        # β = α(1-μ)/μ
        alpha_v = alpha[valid_params]
        beta_param[valid_params] = alpha_v * (1 - mu_v) / mu_v
    
    combined['Alpha'] = alpha
    combined['Beta'] = beta_param
    
    # Filter out invalid Alpha/Beta
    valid_ab = (combined['Alpha'] > 0) & (combined['Beta'] > 0)
    combined = combined[valid_ab].copy()
    
    # Prepare output DataFrame for root-finder
    result = pd.DataFrame({
        'EVENTID': combined['EVENTID'],
        'Port_Rate': combined['RATE'],
        'MaxExposure': combined['Comb_ExpValue'],
        'Alpha': combined['Alpha'],
        'Beta': combined['Beta'],
        'Comb_MeanLoss': combined['Comb_MeanLoss'],
        'Comb_StdDev': combined['Comb_StdDev']
    })
    
    # Sort by MaxExposure descending (matches SQL ORDER BY)
    result = result.sort_values('MaxExposure', ascending=False).reset_index(drop=True)
    
    return result


def calculate_portfolio_minus_account(
    portfolio_elt: pd.DataFrame,
    account_elt: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the "Portfolio minus Account" moments for Marginal PML.
    
    This is used to determine PML(Portfolio - Account) which is needed for
    Marginal Impact = PML(Portfolio) - PML(Portfolio - Account).
    
    Subtraction Rules:
        - Mean Loss: Portfolio - Account
        - Max Exposure: Portfolio - Account
        - Correlated Std Dev: Portfolio - Account
        - Independent Std Dev: sqrt(Portfolio² - Account²) if Portfolio² >= Account²
    
    Args:
        portfolio_elt: Full portfolio ELT
        account_elt: Account to subtract from portfolio
    
    Returns:
        DataFrame ready for BrentQ root-finder (same format as calculate_combined_moments)
    """
    # Validate input columns
    required_port_cols = ['EVENTID', 'RATE', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI']
    required_acct_cols = ['EVENTID', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI']
    
    missing_port = [c for c in required_port_cols if c not in portfolio_elt.columns]
    missing_acct = [c for c in required_acct_cols if c not in account_elt.columns]
    
    if missing_port:
        raise ValueError(f"portfolio_elt missing required columns: {missing_port}")
    if missing_acct:
        raise ValueError(f"account_elt missing required columns: {missing_acct}")
    
    if portfolio_elt.empty:
        warnings.warn("portfolio_elt is empty - returning empty DataFrame")
        return pd.DataFrame(columns=[
            'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta',
            'Comb_MeanLoss', 'Comb_StdDev'
        ])
    
    # LEFT JOIN: Portfolio events, with account data where available
    combined = portfolio_elt.merge(
        account_elt,
        on='EVENTID',
        how='left',
        suffixes=('_port', '_acct')
    )
    
    # Fill NaN with 0 for events where account has no exposure
    combined['PERSPVALUE_acct'] = combined['PERSPVALUE_acct'].fillna(0)
    combined['EXPVALUE_acct'] = combined['EXPVALUE_acct'].fillna(0)
    combined['STDDEVC_acct'] = combined['STDDEVC_acct'].fillna(0)
    combined['STDDEVI_acct'] = combined['STDDEVI_acct'].fillna(0)
    
    # Handle column naming based on merge result
    if 'RATE_port' in combined.columns:
        combined['RATE'] = combined['RATE_port']
    
    # Subtraction Rules (vectorized)
    # 1. Mean Loss: Portfolio - Account
    combined['Comb_MeanLoss'] = (
        combined['PERSPVALUE_port'] - combined['PERSPVALUE_acct']
    )
    
    # 2. Max Exposure: Portfolio - Account
    combined['Comb_ExpValue'] = (
        combined['EXPVALUE_port'] - combined['EXPVALUE_acct']
    )
    
    # 3. Correlated Std Dev: Portfolio - Account
    combined['Comb_StdDevC'] = (
        combined['STDDEVC_port'] - combined['STDDEVC_acct']
    )
    
    # 4. Independent Std Dev: sqrt(Portfolio² - Account²) if positive
    # Use np.maximum to prevent negative sqrt
    port_stddevi_sq = np.square(combined['STDDEVI_port'])
    acct_stddevi_sq = np.square(combined['STDDEVI_acct'])
    diff_sq = np.maximum(0, port_stddevi_sq - acct_stddevi_sq)
    combined['Comb_StdDevI'] = np.sqrt(diff_sq)
    
    # Total standard deviation
    combined['Comb_StdDev'] = combined['Comb_StdDevI'] + combined['Comb_StdDevC']
    
    # Filter: Combined exposure and mean must be positive
    # Remove events where account exposure >= portfolio exposure
    valid_mask = (
        (combined['Comb_ExpValue'] > 1) &  # Minimum exposure threshold
        (combined['Comb_MeanLoss'] > 1) &  # Minimum mean loss threshold
        (combined['Comb_StdDev'] > 0)
    )
    combined = combined[valid_mask].copy()
    
    if combined.empty:
        warnings.warn("No valid events after filtering - returning empty DataFrame")
        return pd.DataFrame(columns=[
            'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta',
            'Comb_MeanLoss', 'Comb_StdDev'
        ])
    
    # Method of Moments: Derive Beta Distribution Parameters
    combined['BetaDistMeanU'] = (
        combined['Comb_MeanLoss'] / combined['Comb_ExpValue']
    )
    combined['BetaDistStdDevO'] = (
        combined['Comb_StdDev'] / combined['Comb_ExpValue']
    )
    
    mu = combined['BetaDistMeanU'].values
    sigma = combined['BetaDistStdDevO'].values
    
    # Validity conditions
    valid_params = (
        (mu > 0) & (mu < 1) &
        (sigma > 0) &
        (np.square(sigma) < mu * (1 - mu))
    )
    
    alpha = np.full_like(mu, np.nan)
    beta_param = np.full_like(mu, np.nan)
    
    if valid_params.any():
        mu_v = mu[valid_params]
        sigma_v = sigma[valid_params]
        sigma_sq = np.square(sigma_v)
        
        alpha[valid_params] = np.square(mu_v) * (1 - mu_v) / sigma_sq - mu_v
        alpha_v = alpha[valid_params]
        beta_param[valid_params] = alpha_v * (1 - mu_v) / mu_v
    
    combined['Alpha'] = alpha
    combined['Beta'] = beta_param
    
    # Filter out invalid Alpha/Beta
    valid_ab = (combined['Alpha'] > 0) & (combined['Beta'] > 0)
    combined = combined[valid_ab].copy()
    
    # Prepare output
    result = pd.DataFrame({
        'EVENTID': combined['EVENTID'],
        'Port_Rate': combined['RATE'],
        'MaxExposure': combined['Comb_ExpValue'],
        'Alpha': combined['Alpha'],
        'Beta': combined['Beta'],
        'Comb_MeanLoss': combined['Comb_MeanLoss'],
        'Comb_StdDev': combined['Comb_StdDev']
    })
    
    result = result.sort_values('MaxExposure', ascending=False).reset_index(drop=True)
    
    return result


class PMLCalculator:
    """
    High-performance PML calculator using precomputed NumPy arrays.
    
    This class caches all static computations at initialization time,
    allowing repeated PML/Return Period calculations with minimal overhead.
    
    Usage:
        calc = PMLCalculator(events_df, use_poisson=True)
        rp = calc.implied_return_period(pml=1_000_000)
        pml, iters = calc.find_pml_for_rp(target_rp=250)
    """
    __slots__ = ('_port_rates', '_max_exposures', '_alphas', '_betas', 
                 '_valid_mask', '_use_poisson', '_n_events', '_config')
    
    def __init__(
        self,
        events: pd.DataFrame,
        use_poisson: bool = True,
        config: Optional[MomentsConfig] = None
    ):
        """
        Initialize calculator with precomputed arrays.
        
        Args:
            events: DataFrame with columns [Port_Rate, MaxExposure, Alpha, Beta]
            use_poisson: If True, use Poisson AEP formula
            config: Optional configuration for solver parameters
        """
        self._use_poisson = use_poisson
        self._config = config or MomentsConfig(use_poisson_aep=use_poisson)
        
        if events.empty:
            self._n_events = 0
            self._port_rates = np.array([])
            self._max_exposures = np.array([])
            self._alphas = np.array([])
            self._betas = np.array([])
            self._valid_mask = np.array([], dtype=bool)
            return
        
        # Extract and cache NumPy arrays (one-time cost)
        self._port_rates = events['Port_Rate'].values.astype(np.float64)
        self._max_exposures = events['MaxExposure'].values.astype(np.float64)
        self._alphas = events['Alpha'].values.astype(np.float64)
        self._betas = events['Beta'].values.astype(np.float64)
        
        # Precompute static validity mask
        self._valid_mask = (
            (self._max_exposures > 0) &
            (self._alphas > 0) &
            (self._betas > 0) &
            np.isfinite(self._alphas) &
            np.isfinite(self._betas)
        )
        
        self._n_events = len(events)
    
    @property
    def n_events(self) -> int:
        """Number of events in this calculator."""
        return self._n_events
    
    @property
    def max_exposure(self) -> float:
        """Maximum exposure across all events."""
        if self._n_events == 0:
            return 0.0
        return float(np.max(self._max_exposures[self._valid_mask]))
    
    def implied_return_period(self, pml: float) -> float:
        """
        Calculate implied return period for a given PML threshold.
        
        Uses the Poisson frequency model with Beta severity distribution:
        
        For each event i where MaxExposure_i >= PML:
            CEP_i = 1 - F_Beta(PML/MaxExposure_i | α_i, β_i)
            OEP_i = Rate_i × CEP_i
        
        λ = Σ OEP_i (Poisson parameter)
        
        If use_poisson:
            AEP = 1 - exp(-λ)
        Else:
            AEP = λ
        
        RP = 1 / AEP
        
        Args:
            pml: Candidate PML value (positive float)
        
        Returns:
            Return period in years. Returns inf if AEP is zero or no events.
        """
        if self._n_events == 0:
            return math.inf
        
        # Handle PML <= 0: all events can exceed
        if pml <= 0:
            lambda_param = float(np.sum(self._port_rates))
            if lambda_param <= 0:
                return math.inf
            aep = -np.expm1(-lambda_param) if self._use_poisson else lambda_param
            return 1.0 / aep if aep > 0 else math.inf
        
        # Normalize PML by max exposure
        u = pml / self._max_exposures
        
        # Filter: events where MaxExposure < PML cannot exceed
        can_exceed = self._max_exposures >= pml
        valid = self._valid_mask & can_exceed
        
        # Initialize exceedance probabilities
        prob_exceed = np.zeros_like(u)
        
        # Fast path: handle trivial regions without Beta CDF
        lt0 = valid & (u <= 0.0)  # Always exceed
        ge1 = valid & (u >= 1.0)  # Never exceed
        mid = valid & (u > 0.0) & (u < 1.0)  # Need CDF
        
        prob_exceed[lt0] = 1.0
        prob_exceed[ge1] = 0.0
        
        # Beta CDF only for middle region
        if mid.any():
            cdf_vals = beta_dist.cdf(
                u[mid],
                self._alphas[mid],
                self._betas[mid]
            )
            prob_exceed[mid] = 1.0 - cdf_vals
        
        # Calculate λ
        oep_values = self._port_rates * prob_exceed
        lambda_param = float(np.sum(oep_values))
        
        if lambda_param <= 0:
            return math.inf
        
        # AEP calculation
        if self._use_poisson:
            aep = -np.expm1(-lambda_param)
        else:
            aep = lambda_param
        
        return 1.0 / aep if aep > 0 else math.inf
    
    def find_pml_for_rp(
        self,
        target_rp: int,
        seed: Optional[float] = None,
        seed_increment: Optional[float] = None
    ) -> Tuple[float, int]:
        """
        Find PML for a target return period using Brent's method.
        
        Args:
            target_rp: Target return period in years (e.g., 100, 250, 500)
            seed: Initial PML guess. If None, uses MaxExposure / 10
            seed_increment: Bracket expansion step. If None, uses seed * 0.01
        
        Returns:
            Tuple[float, int]: (converged_pml, num_evaluations)
        """
        if self._n_events == 0:
            return 0.0, 0
        
        config = self._config
        
        # Default seed
        if seed is None:
            seed = self.max_exposure / 10
        if seed <= 0:
            seed = 1e6  # Fallback
        
        if seed_increment is None:
            seed_increment = seed * 0.01
        
        eval_count = [0]
        
        def objective(pml: float) -> float:
            eval_count[0] += 1
            return self.implied_return_period(pml) - target_rp
        
        # Evaluate seed
        seed_obj = objective(seed)
        
        if abs(seed_obj) <= config.rp_tolerance:
            return seed, eval_count[0]
        
        # Exponential bracketing
        min_pml = -1.0
        max_pml = -1.0
        curr_pml = seed
        
        multipliers_up = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.3, 16.0, 22.6, 32.0, 
                        45.3, 64.0, 90.5, 128.0, 181.0, 256.0, 362.0, 512.0, 724.0, 1024.0]
        multipliers_down = [0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.09, 0.06, 0.04, 
                          0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
        
        if seed_obj < 0:
            # RP < target, search upward
            min_pml = seed
            for mult in multipliers_up:
                test_pml = seed * mult
                test_obj = objective(test_pml)
                if test_obj >= 0:
                    max_pml = test_pml
                    break
                min_pml = test_pml
                curr_pml = test_pml
        else:
            # RP > target, search downward
            max_pml = seed
            for mult in multipliers_down:
                test_pml = max(0, seed * mult)
                test_obj = objective(test_pml)
                if test_obj <= 0:
                    min_pml = test_pml
                    break
                max_pml = test_pml
                curr_pml = test_pml
        
        # Fallback: linear search if exponential failed
        if min_pml < 0 or max_pml < 0:
            curr_pml = seed
            min_pml = -1.0
            max_pml = -1.0
            
            for _ in range(500):
                if curr_pml < 0:
                    curr_pml = 0
                
                obj_val = objective(curr_pml)
                
                if abs(obj_val) <= config.rp_tolerance:
                    return curr_pml, eval_count[0]
                
                if obj_val < 0:
                    min_pml = curr_pml
                    if max_pml >= 0:
                        break
                    curr_pml += seed_increment
                else:
                    max_pml = curr_pml
                    if min_pml >= 0:
                        break
                    curr_pml = max(0, curr_pml - seed_increment)
        
        # Brent's method
        if min_pml >= 0 and max_pml >= 0 and min_pml != max_pml:
            try:
                converged_pml = brentq(
                    objective,
                    min_pml,
                    max_pml,
                    xtol=config.brent_xtol,
                    rtol=config.brent_rtol,
                    maxiter=config.max_iterations - eval_count[0]
                )
                return converged_pml, eval_count[0]
            except ValueError as e:
                warnings.warn(f"Brent's method failed for RP={target_rp}: {e}")
                return (min_pml + max_pml) / 2.0, eval_count[0]
        
        warnings.warn(f"Could not bracket for RP={target_rp}")
        return curr_pml, eval_count[0]


def calculate_marginal_impact(
    portfolio_elt: pd.DataFrame,
    account_elt: pd.DataFrame,
    return_periods: Tuple[int, ...] = (50, 100, 250, 500, 1000),
    config: Optional[MomentsConfig] = None
) -> Dict[int, float]:
    """
    Calculate Marginal Impact for an account across multiple return periods.
    
    Marginal Impact = PML(Portfolio + Account) - PML(Portfolio)
    
    For the "Portfolio minus Account" scenario (used in RI pricing):
    Marginal Impact = PML(Portfolio) - PML(Portfolio - Account)
    
    Args:
        portfolio_elt: Full portfolio Event Loss Table
        account_elt: Account Event Loss Table
        return_periods: Return periods to calculate (years)
        config: Solver configuration
    
    Returns:
        Dict mapping return period to marginal impact: {RP: impact}
    """
    if config is None:
        config = MomentsConfig()
    
    # Calculate portfolio-minus-account moments
    combined_df = calculate_portfolio_minus_account(portfolio_elt, account_elt)
    
    if combined_df.empty:
        warnings.warn("No valid combined events - returning zeros")
        return {rp: 0.0 for rp in return_periods}
    
    # Create calculator for combined events
    combined_calc = PMLCalculator(combined_df, use_poisson=config.use_poisson_aep, config=config)
    
    # Create calculator for portfolio
    portfolio_df = _portfolio_to_events_df(portfolio_elt)
    portfolio_calc = PMLCalculator(portfolio_df, use_poisson=config.use_poisson_aep, config=config)
    
    results = {}
    
    for rp in return_periods:
        # Get portfolio PML
        portfolio_pml, _ = portfolio_calc.find_pml_for_rp(rp)
        
        # Get combined (portfolio - account) PML
        combined_pml, _ = combined_calc.find_pml_for_rp(rp, seed=portfolio_pml * 0.95)
        
        # Marginal Impact = Portfolio PML - (Portfolio - Account) PML
        marginal = portfolio_pml - combined_pml
        results[rp] = marginal
    
    return results


def _portfolio_to_events_df(portfolio_elt: pd.DataFrame) -> pd.DataFrame:
    """
    Convert portfolio ELT to events DataFrame format for PMLCalculator.
    
    Applies Method of Moments to derive Alpha/Beta parameters.
    """
    if portfolio_elt.empty:
        return pd.DataFrame(columns=[
            'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta'
        ])
    
    df = portfolio_elt.copy()
    
    # Ensure required columns exist
    df['MaxExposure'] = df['EXPVALUE']
    df['Port_Rate'] = df['RATE']
    
    # Total std dev
    df['TotalStdDev'] = df['STDDEVI'] + df['STDDEVC']
    
    # Filter valid events
    valid = (
        (df['EXPVALUE'] > 0) &
        (df['PERSPVALUE'] > 0) &
        (df['TotalStdDev'] > 0)
    )
    df = df[valid].copy()
    
    if df.empty:
        return pd.DataFrame(columns=[
            'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta'
        ])
    
    # Method of Moments
    mu = df['PERSPVALUE'].values / df['EXPVALUE'].values
    sigma = df['TotalStdDev'].values / df['EXPVALUE'].values
    
    valid_params = (
        (mu > 0) & (mu < 1) &
        (sigma > 0) &
        (np.square(sigma) < mu * (1 - mu))
    )
    
    alpha = np.full_like(mu, np.nan)
    beta_param = np.full_like(mu, np.nan)
    
    if valid_params.any():
        mu_v = mu[valid_params]
        sigma_v = sigma[valid_params]
        sigma_sq = np.square(sigma_v)
        
        alpha[valid_params] = np.square(mu_v) * (1 - mu_v) / sigma_sq - mu_v
        alpha_v = alpha[valid_params]
        beta_param[valid_params] = alpha_v * (1 - mu_v) / mu_v
    
    df['Alpha'] = alpha
    df['Beta'] = beta_param
    
    # Filter invalid
    valid_ab = (df['Alpha'] > 0) & (df['Beta'] > 0)
    df = df[valid_ab].copy()
    
    result = pd.DataFrame({
        'EVENTID': df['EVENTID'],
        'Port_Rate': df['Port_Rate'],
        'MaxExposure': df['MaxExposure'],
        'Alpha': df['Alpha'],
        'Beta': df['Beta'],
        'Corr_Fraction': (df['STDDEVC'].values / df['TotalStdDev'].values).clip(0, 0.999),
    })
    
    return result.sort_values('MaxExposure', ascending=False).reset_index(drop=True)


# =============================================================================
# CORRELATED PML SIMULATOR — GAUSSIAN COPULA + BETA MARGINALS
# =============================================================================
#
# Uses each event's SD_C/(SD_C+SD_I) ratio as the loading on a shared
# factor (Gaussian copula).  High-SD_C events move together in tail
# scenarios, capturing portfolio concentration risk that the analytical
# (independent) path misses.
# =============================================================================

class CorrelatedSimulator:
    """Monte Carlo PML simulator with Gaussian copula correlation.

    Given an events DataFrame (with Alpha, Beta, Port_Rate, MaxExposure,
    Corr_Fraction), generates *n_trials* simulated years of correlated
    aggregate losses.

    Correlation model (single-factor or multi-factor):
        Z_i = ρ_i · Z_group[g_i] + √(1 − ρ_i²) · Z_ind_i
        U_i = Φ(Z_i)                           → correlated uniforms
        LR_i = F⁻¹_Beta(U_i, α_i, β_i)       → correlated loss ratios
        Loss_i = LR_i × MaxExposure_i           → dollar loss
        AnnualLoss = Σ (occurrence_i × Loss_i)

    ρ_i = Corr_Fraction = SD_C / (SD_C + SD_I).
    """

    __slots__ = ('_events', '_config', '_annual_losses')

    def __init__(
        self,
        events_df: pd.DataFrame,
        config: Optional[CorrelationConfig] = None,
    ):
        self._config = config or CorrelationConfig()
        self._events = events_df
        self._annual_losses: Optional[np.ndarray] = None

    def simulate(self) -> np.ndarray:
        """Run Monte Carlo, return **sorted** annual-aggregate-loss array."""
        if self._annual_losses is not None:
            return self._annual_losses

        cfg = self._config
        rng = np.random.default_rng(cfg.seed)
        T = cfg.n_trials
        ev = self._events
        n = len(ev)

        if n == 0:
            self._annual_losses = np.zeros(T)
            return self._annual_losses

        rates   = ev['Port_Rate'].values
        max_exp = ev['MaxExposure'].values
        alphas  = ev['Alpha'].values
        betas   = ev['Beta'].values

        # Correlation fractions (ρ per event)
        if 'Corr_Fraction' in ev.columns:
            rho = np.asarray(ev['Corr_Fraction'].values, dtype=np.float64).clip(0, 0.999)
        else:
            rho = np.zeros(n)

        # Groups for shared factors
        if cfg.event_groups is not None and len(cfg.event_groups) == n:
            groups = np.asarray(cfg.event_groups, dtype=int)
            n_groups = int(groups.max()) + 1
        else:
            groups = np.zeros(n, dtype=int)
            n_groups = 1

        # --- Draw random numbers ------------------------------------------
        Z_common = rng.standard_normal((T, n_groups))   # shared factors
        U_occur  = rng.random((T, n))                   # occurrence draw

        # --- Sparse occurrence mask (huge speed-up) -----------------------
        # Most events have low rates, so only a small fraction occur.
        # We compute the expensive beta.ppf ONLY for occurring events.
        occurs = (U_occur < rates[np.newaxis, :])                       # bool (T, n)
        trial_idx, event_idx = np.nonzero(occurs)
        n_occ = len(trial_idx)

        if n_occ == 0:
            self._annual_losses = np.zeros(T)
            return self._annual_losses

        # Independent normals only for occurring (trial, event) pairs
        Z_ind_occ = rng.standard_normal(n_occ)

        # Correlated normal per occurring pair
        rho_occ = rho[event_idx]
        Z_occ = (rho_occ * Z_common[trial_idx, groups[event_idx]]
                 + np.sqrt(1.0 - rho_occ ** 2) * Z_ind_occ)

        from scipy.stats import norm as _norm
        U_sev = _norm.cdf(Z_occ)                                       # (n_occ,)

        lr = beta_dist.ppf(U_sev, alphas[event_idx], betas[event_idx]) # (n_occ,)
        np.nan_to_num(lr, copy=False, nan=0.0)
        np.clip(lr, 0.0, 1.0, out=lr)

        # --- Annual aggregate via scatter-add -----------------------------
        dollar_losses = lr * max_exp[event_idx]
        annual = np.bincount(trial_idx, weights=dollar_losses,
                             minlength=T).astype(np.float64)

        self._annual_losses = np.sort(annual)
        return self._annual_losses

    def pml_at_rp(self, rp: int) -> float:
        """Empirical PML for a given return period."""
        annual = self.simulate()
        return float(np.percentile(annual, (1.0 - 1.0 / rp) * 100.0))

    def aal(self) -> float:
        return float(np.mean(self.simulate()))

    def ep_curve(self, rps: Optional[Tuple[int, ...]] = None) -> Dict[int, float]:
        if rps is None:
            rps = (10, 25, 50, 100, 250, 500, 1000)
        return {rp: self.pml_at_rp(rp) for rp in rps}


# =============================================================================
# HIGH-PERFORMANCE MARGINAL PML ENGINE
# =============================================================================
# 
# This engine is designed for "Consumer-Fast" pricing scenarios where:
# 1. A baseline (portfolio) is loaded ONCE
# 2. Multiple accounts are "hot-swapped" for rapid-fire pricing
# 3. Marginal PML is computed in milliseconds per account
#
# Key optimizations:
# - Baseline PML solved once and cached
# - Portfolio arrays indexed by EVENTID for O(1) account lookup
# - Vectorized subtraction math with NumPy views
# - Numba-compatible array layouts (future optimization)
# =============================================================================


# Valid pricing modes
PRICING_MODE_SUBTRACT = 'subtract'  # Marginal = PML(Portfolio) - PML(Portfolio - Account)
PRICING_MODE_ADD = 'add'            # Marginal = PML(Portfolio + Account) - PML(Portfolio)
VALID_PRICING_MODES = {PRICING_MODE_SUBTRACT, PRICING_MODE_ADD}


class MarginalPMLEngine:
    """
    High-Performance Marginal PML Pricing Engine with Hot-Swap capability.
    
    This engine pre-computes baseline (portfolio) PMLs once at initialization,
    then allows rapid-fire pricing of multiple accounts by "hot-swapping"
    account ELTs without recomputing the baseline.
    
    Pricing Modes:
        - 'subtract': Marginal = PML(Portfolio) - PML(Portfolio - Account)
                      Used for RI pricing (existing account removal impact)
        - 'add':      Marginal = PML(Portfolio + Account) - PML(Portfolio)
                      Used for new business pricing (prospective account impact)
    
    Performance Characteristics:
        - Baseline initialization: O(n_events × n_return_periods)
        - Account pricing: O(n_account_events) per account
        - Memory: O(n_events) for indexed portfolio arrays
    
    Usage:
        # Initialize once with portfolio (subtract mode - default)
        engine = MarginalPMLEngine(portfolio_elt, mode='subtract')
        
        # Or use add mode for new business pricing
        engine = MarginalPMLEngine(portfolio_elt, mode='add')
        
        # Price multiple accounts rapidly
        for account_elt in account_elts:
            result = engine.price_account(account_elt)
            print(f"Account marginal PMLs: {result}")
    
    The Poisson AEP formula is preserved:
        AEP = 1 - exp(-Σ(Rate × ProbExceed))
    """
    
    __slots__ = (
        '_portfolio_elt', '_portfolio_indexed', '_portfolio_event_ids',
        '_baseline_pmls', '_baseline_calc', '_config', '_return_periods',
        '_n_portfolio_events', '_is_initialized', '_mode',
        '_correlation', '_portfolio_events_df',
    )
    
    def __init__(
        self,
        portfolio_elt: pd.DataFrame,
        return_periods: Tuple[int, ...] = (50, 100, 250),
        config: Optional[MomentsConfig] = None,
        mode: str = PRICING_MODE_SUBTRACT,
        correlation: Optional[CorrelationConfig] = None,
    ):
        """
        Initialize engine with portfolio baseline.
        
        This performs all expensive one-time computations:
        1. Index portfolio by EVENTID for O(1) lookup
        2. Compute portfolio Alpha/Beta parameters
        3. Solve baseline PMLs for all return periods
        
        Args:
            portfolio_elt: Full portfolio Event Loss Table with columns:
                [EVENTID, RATE, PERSPVALUE, EXPVALUE, STDDEVC, STDDEVI]
            return_periods: Return periods to compute (default: 50, 100, 250)
            config: Solver configuration
            mode: Pricing mode - 'subtract' or 'add'
                - 'subtract': Marginal = PML(Portfolio) - PML(Portfolio - Account)
                - 'add': Marginal = PML(Portfolio + Account) - PML(Portfolio)
            correlation: Optional CorrelationConfig to enable Gaussian copula
                simulation. When set, PMLs are computed via Monte Carlo
                instead of analytical Brent's method.
        """
        if mode not in VALID_PRICING_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {VALID_PRICING_MODES}")
        
        self._mode = mode
        self._config = config or MomentsConfig()
        self._correlation = correlation
        self._return_periods = return_periods
        self._is_initialized = False
        
        if portfolio_elt.empty:
            raise ValueError("Portfolio ELT cannot be empty")
        
        # Store original portfolio
        self._portfolio_elt = portfolio_elt.copy()
        self._n_portfolio_events = len(portfolio_elt)
        
        # Index portfolio by EVENTID for O(1) account matching
        # This is the key to hot-swap performance
        self._portfolio_indexed = portfolio_elt.set_index('EVENTID')
        self._portfolio_event_ids = set(portfolio_elt['EVENTID'].values)
        
        # Compute portfolio events DataFrame with Alpha/Beta (+ Corr_Fraction)
        portfolio_events_df = _portfolio_to_events_df(portfolio_elt)
        self._portfolio_events_df = portfolio_events_df
        
        if portfolio_events_df.empty:
            raise ValueError("No valid portfolio events after Method of Moments filtering")
        
        # Create baseline calculator (always available for EP curves etc.)
        self._baseline_calc = PMLCalculator(
            portfolio_events_df,
            use_poisson=self._config.use_poisson_aep,
            config=self._config
        )
        
        # Solve baseline PMLs
        self._baseline_pmls = {}
        if correlation is not None:
            # Correlated path — Monte Carlo simulation
            sim = CorrelatedSimulator(portfolio_events_df, correlation)
            for rp in return_periods:
                self._baseline_pmls[rp] = sim.pml_at_rp(rp)
        else:
            # Analytical path — Brent's method (original)
            for rp in return_periods:
                pml, _ = self._baseline_calc.find_pml_for_rp(rp)
                self._baseline_pmls[rp] = pml
        
        self._is_initialized = True
    
    @property
    def baseline_pmls(self) -> Dict[int, float]:
        """Get baseline (portfolio) PMLs for all return periods."""
        return self._baseline_pmls.copy()
    
    @property
    def n_portfolio_events(self) -> int:
        """Number of events in portfolio."""
        return self._n_portfolio_events
    
    @property
    def return_periods(self) -> Tuple[int, ...]:
        """Return periods being computed."""
        return self._return_periods
    
    @property
    def mode(self) -> str:
        """Pricing mode: 'subtract' or 'add'."""
        return self._mode
    
    def price_account(
        self,
        account_elt: pd.DataFrame,
        return_dict: bool = True,
        include_combined_pml: bool = False
    ) -> Dict[int, float]:
        """
        Compute Marginal PML for an account using hot-swap mode.
        
        This is the fast path - uses pre-computed baseline and indexed
        portfolio arrays for rapid account pricing.
        
        Modes:
            - 'subtract': Marginal = PML(Portfolio) - PML(Portfolio - Account)
            - 'add':      Marginal = PML(Portfolio + Account) - PML(Portfolio)
        
        Args:
            account_elt: Account Event Loss Table with columns:
                [EVENTID, PERSPVALUE, EXPVALUE, STDDEVC, STDDEVI]
                For 'add' mode, RATE column is optional (uses portfolio rate if missing)
            return_dict: If True, return dict; if False, return tuple
            include_combined_pml: If True, return dict with both marginal and combined PMls
        
        Returns:
            If include_combined_pml=False (default):
                Dict mapping return period to marginal impact: {50: val, 100: val, 250: val}
            
            If include_combined_pml=True:
                Dict with structure: {
                    'marginal': {50: val, 100: val, 250: val},
                    'combined': {50: val, 100: val, 250: val},
                    'baseline': {50: val, 100: val, 250: val}
                }
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized")
        
        if account_elt.empty:
            # No account events = no marginal impact
            if include_combined_pml:
                return {
                    'marginal': {rp: 0.0 for rp in self._return_periods},
                    'combined': self._baseline_pmls.copy(),
                    'baseline': self._baseline_pmls.copy()
                }
            return {rp: 0.0 for rp in self._return_periods}
        
        # Choose path based on mode
        if self._mode == PRICING_MODE_SUBTRACT:
            # Fast-path subtraction using indexed portfolio
            combined_df = self._fast_subtract_account(account_elt)
        else:
            # Fast-path addition using indexed portfolio
            combined_df = self._fast_add_account(account_elt)
        
        if combined_df.empty:
            # All events filtered out
            if include_combined_pml:
                return {
                    'marginal': {rp: 0.0 for rp in self._return_periods},
                    'combined': self._baseline_pmls.copy(),
                    'baseline': self._baseline_pmls.copy()
                }
            return {rp: 0.0 for rp in self._return_periods}
        
        # Compute marginal impacts
        marginal_results = {}
        combined_results = {}

        if self._correlation is not None:
            # Correlated path — Monte Carlo
            combined_sim = CorrelatedSimulator(combined_df, self._correlation)
            for rp in self._return_periods:
                baseline_pml = self._baseline_pmls[rp]
                combined_pml = combined_sim.pml_at_rp(rp)
                combined_results[rp] = combined_pml
                if self._mode == PRICING_MODE_SUBTRACT:
                    marginal_results[rp] = baseline_pml - combined_pml
                else:
                    marginal_results[rp] = combined_pml - baseline_pml
        else:
            # Analytical path — Brent's method (original)
            combined_calc = PMLCalculator(
                combined_df,
                use_poisson=self._config.use_poisson_aep,
                config=self._config
            )
            for rp in self._return_periods:
                baseline_pml = self._baseline_pmls[rp]
                if self._mode == PRICING_MODE_SUBTRACT:
                    seed = baseline_pml * 0.95
                else:
                    seed = baseline_pml * 1.05
                combined_pml, _ = combined_calc.find_pml_for_rp(rp, seed=seed)
                combined_results[rp] = combined_pml
                if self._mode == PRICING_MODE_SUBTRACT:
                    marginal_results[rp] = baseline_pml - combined_pml
                else:
                    marginal_results[rp] = combined_pml - baseline_pml
        
        if include_combined_pml:
            return {
                'marginal': marginal_results,
                'combined': combined_results,
                'baseline': self._baseline_pmls.copy()
            }
        
        return marginal_results
    
    def price_account_batch(
        self,
        account_elts: Dict[str, pd.DataFrame],
        progress_callback: Optional[callable] = None,
        include_combined_pml: bool = False
    ) -> pd.DataFrame:
        """
        Price multiple accounts in batch for maximum throughput.
        
        Args:
            account_elts: Dict mapping account_id to account ELT DataFrame
            progress_callback: Optional callback(account_id, idx, total)
            include_combined_pml: If True, include baseline and combined PML columns
        
        Returns:
            DataFrame with columns:
            - AccountID
            - RI_50, RI_100, RI_250 (marginal impacts)
            - Baseline_50, Baseline_100, Baseline_250 (if include_combined_pml=True)
            - Combined_50, Combined_100, Combined_250 (if include_combined_pml=True)
        """
        results = []
        total = len(account_elts)
        
        for idx, (account_id, account_elt) in enumerate(account_elts.items()):
            if progress_callback:
                progress_callback(account_id, idx, total)
            
            try:
                pricing_result = self.price_account(account_elt, include_combined_pml=include_combined_pml)
                row = {'AccountID': account_id}
                
                if include_combined_pml:
                    # Add baseline columns
                    for rp in self._return_periods:
                        row[f'Baseline_{rp}'] = pricing_result['baseline'][rp]
                    
                    # Add combined columns
                    for rp in self._return_periods:
                        row[f'Combined_{rp}'] = pricing_result['combined'][rp]
                    
                    # Add marginal columns (labeled RI for Reinsurance Impact)
                    for rp in self._return_periods:
                        row[f'RI_{rp}'] = pricing_result['marginal'][rp]
                else:
                    # Only marginal columns
                    for rp, value in pricing_result.items():
                        row[f'RI_{rp}'] = value
                
                results.append(row)
            except Exception as e:
                warnings.warn(f"Error pricing account {account_id}: {e}")
                row = {'AccountID': account_id}
                for rp in self._return_periods:
                    row[f'RI_{rp}'] = np.nan
                    if include_combined_pml:
                        row[f'Baseline_{rp}'] = np.nan
                        row[f'Combined_{rp}'] = np.nan
                results.append(row)
        
        return pd.DataFrame(results)
    
    def _fast_subtract_account(self, account_elt: pd.DataFrame) -> pd.DataFrame:
        """
        Fast-path portfolio-minus-account using indexed arrays.
        
        This is optimized for the hot-swap scenario where the portfolio
        is already indexed and we just need to subtract account values.
        
        Returns:
            DataFrame ready for PMLCalculator with columns:
            [EVENTID, Port_Rate, MaxExposure, Alpha, Beta]
        """
        # Get portfolio values only for events in account (fast lookup)
        account_event_ids = account_elt['EVENTID'].values
        
        # Separate events into: in-portfolio vs new-to-portfolio
        in_portfolio_mask = np.isin(account_event_ids, list(self._portfolio_event_ids))
        
        # For events IN portfolio: subtract account from portfolio
        acct_in_port = account_elt[in_portfolio_mask].copy()
        
        if acct_in_port.empty:
            # Account has no overlap with portfolio - just use full portfolio
            return _portfolio_to_events_df(self._portfolio_elt)
        
        # Get corresponding portfolio rows (aligned by EVENTID)
        port_subset = self._portfolio_indexed.loc[acct_in_port['EVENTID'].values].reset_index()
        
        # Perform subtraction
        combined = port_subset.copy()
        combined['PERSPVALUE'] = port_subset['PERSPVALUE'] - acct_in_port['PERSPVALUE'].values
        combined['EXPVALUE'] = port_subset['EXPVALUE'] - acct_in_port['EXPVALUE'].values
        combined['STDDEVC'] = port_subset['STDDEVC'] - acct_in_port['STDDEVC'].values
        
        # Independent StdDev: sqrt(port² - acct²)
        port_stddevi_sq = np.square(port_subset['STDDEVI'].values)
        acct_stddevi_sq = np.square(acct_in_port['STDDEVI'].values)
        diff_sq = np.maximum(0, port_stddevi_sq - acct_stddevi_sq)
        combined['STDDEVI'] = np.sqrt(diff_sq)
        
        # Get portfolio events NOT affected by this account
        affected_event_ids = set(acct_in_port['EVENTID'].values)
        unaffected_mask = ~self._portfolio_elt['EVENTID'].isin(affected_event_ids)
        unaffected = self._portfolio_elt[unaffected_mask].copy()
        
        # Combine: modified events + unaffected events
        full_combined = pd.concat([combined, unaffected], ignore_index=True)
        
        # Filter validity and compute Alpha/Beta
        return self._compute_alpha_beta(full_combined)
    
    def _fast_add_account(self, account_elt: pd.DataFrame) -> pd.DataFrame:
        """
        Fast-path portfolio-plus-account using indexed arrays.
        
        This is optimized for the hot-swap scenario where the portfolio
        is already indexed and we just need to add account values.
        
        Aggregation Rules:
            - Mean Loss (PERSPVALUE): Additive
            - Max Exposure (EXPVALUE): Additive
            - Correlated Std Dev (STDDEVC): Additive
            - Independent Std Dev (STDDEVI): Root sum of squares sqrt(a² + b²)
        
        Returns:
            DataFrame ready for PMLCalculator with columns:
            [EVENTID, Port_Rate, MaxExposure, Alpha, Beta]
        """
        account_event_ids = account_elt['EVENTID'].values
        
        # Separate events into: in-portfolio vs new-to-portfolio
        in_portfolio_mask = np.isin(account_event_ids, list(self._portfolio_event_ids))
        
        # Events that exist in both portfolio and account
        acct_in_port = account_elt[in_portfolio_mask].copy()
        
        # Events only in account (new-to-portfolio)
        acct_new = account_elt[~in_portfolio_mask].copy()
        
        combined_parts = []
        
        # Part 1: Portfolio events affected by account (add values)
        if not acct_in_port.empty:
            port_subset = self._portfolio_indexed.loc[acct_in_port['EVENTID'].values].reset_index()
            
            combined = port_subset.copy()
            combined['PERSPVALUE'] = port_subset['PERSPVALUE'] + acct_in_port['PERSPVALUE'].values
            combined['EXPVALUE'] = port_subset['EXPVALUE'] + acct_in_port['EXPVALUE'].values
            combined['STDDEVC'] = port_subset['STDDEVC'] + acct_in_port['STDDEVC'].values
            
            # Independent StdDev: sqrt(port² + acct²)
            combined['STDDEVI'] = np.sqrt(
                np.square(port_subset['STDDEVI'].values) + 
                np.square(acct_in_port['STDDEVI'].values)
            )
            
            combined_parts.append(combined)
        
        # Part 2: Portfolio events NOT affected by this account
        if not acct_in_port.empty:
            affected_event_ids = set(acct_in_port['EVENTID'].values)
            unaffected_mask = ~self._portfolio_elt['EVENTID'].isin(affected_event_ids)
            unaffected = self._portfolio_elt[unaffected_mask].copy()
        else:
            unaffected = self._portfolio_elt.copy()
        combined_parts.append(unaffected)
        
        # Part 3: New-to-portfolio events from account
        # These need RATE - use portfolio average rate if not provided
        if not acct_new.empty:
            new_events = acct_new.copy()
            
            if 'RATE' not in new_events.columns:
                # Use average portfolio rate for new events
                avg_rate = self._portfolio_elt['RATE'].mean()
                new_events['RATE'] = avg_rate
            
            combined_parts.append(new_events)
        
        # Combine all parts
        full_combined = pd.concat(combined_parts, ignore_index=True)
        
        # Filter validity and compute Alpha/Beta
        return self._compute_alpha_beta(full_combined)
    
    def _compute_alpha_beta(self, elt: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Alpha/Beta parameters using Method of Moments.
        
        Vectorized implementation for maximum speed.
        """
        if elt.empty:
            return pd.DataFrame(columns=[
                'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta'
            ])
        
        # Filter: positive exposure and mean
        valid = (
            (elt['EXPVALUE'] > 1) &
            (elt['PERSPVALUE'] > 1)
        )
        df = elt[valid].copy()
        
        if df.empty:
            return pd.DataFrame(columns=[
                'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta'
            ])
        
        # Total std dev
        df['TotalStdDev'] = df['STDDEVI'] + df['STDDEVC']
        
        # Filter: positive std dev
        df = df[df['TotalStdDev'] > 0].copy()
        
        if df.empty:
            return pd.DataFrame(columns=[
                'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta'
            ])
        
        # Method of Moments (vectorized)
        mu = df['PERSPVALUE'].values / df['EXPVALUE'].values
        sigma = df['TotalStdDev'].values / df['EXPVALUE'].values
        
        # Validity conditions
        valid_params = (
            (mu > 0) & (mu < 1) &
            (sigma > 0) &
            (np.square(sigma) < mu * (1 - mu))
        )
        
        alpha = np.full_like(mu, np.nan)
        beta_param = np.full_like(mu, np.nan)
        
        if valid_params.any():
            mu_v = mu[valid_params]
            sigma_v = sigma[valid_params]
            sigma_sq = np.square(sigma_v)
            
            alpha[valid_params] = np.square(mu_v) * (1 - mu_v) / sigma_sq - mu_v
            alpha_v = alpha[valid_params]
            beta_param[valid_params] = alpha_v * (1 - mu_v) / mu_v
        
        df['Alpha'] = alpha
        df['Beta'] = beta_param
        
        # Filter invalid Alpha/Beta
        valid_ab = (df['Alpha'] > 0) & (df['Beta'] > 0)
        df = df[valid_ab].copy()
        
        if df.empty:
            return pd.DataFrame(columns=[
                'EVENTID', 'Port_Rate', 'MaxExposure', 'Alpha', 'Beta'
            ])
        
        result = pd.DataFrame({
            'EVENTID': df['EVENTID'],
            'Port_Rate': df['RATE'],
            'MaxExposure': df['EXPVALUE'],
            'Alpha': df['Alpha'],
            'Beta': df['Beta'],
            'Corr_Fraction': (df['STDDEVC'].values / df['TotalStdDev'].values).clip(0, 0.999),
        })
        
        return result.sort_values('MaxExposure', ascending=False).reset_index(drop=True)
    
    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the engine state.
        
        Useful for debugging and performance monitoring.
        """
        return {
            'mode': self._mode,
            'n_portfolio_events': self._n_portfolio_events,
            'n_valid_events': self._baseline_calc.n_events,
            'max_exposure': self._baseline_calc.max_exposure,
            'baseline_pmls': self._baseline_pmls.copy(),
            'return_periods': self._return_periods,
            'config': {
                'use_poisson_aep': self._config.use_poisson_aep,
                'rp_tolerance': self._config.rp_tolerance,
                'brent_xtol': self._config.brent_xtol,
            }
        }


# =============================================================================
# YELT (Year Event Loss Table) ENGINE
# =============================================================================
#
# The YELT path provides exact empirical PML calculation from full stochastic
# simulation output.  Correlation between events is already embedded in the
# trial structure — no Beta fitting or copula needed.
#
# Expected YELT schema:
#   TrialID  (int)  – simulation year / trial number  (1 … N)
#   EventID  (int)  – catastrophe event identifier
#   Loss     (float) – ground-up or gross loss for that event in that trial
#
# The engine aggregates losses per trial to build an empirical annual-
# aggregate-loss distribution, then reads PML directly as a percentile.
#
# Marginal PML works the same way as the ELT engine:
#   subtract: Marginal = PML(Portfolio) − PML(Portfolio − Account)
#   add:      Marginal = PML(Portfolio + Account) − PML(Portfolio)
# =============================================================================

# Column name aliases the YELT loader will accept
_YELT_TRIAL_ALIASES = {
    'TRIALID', 'TRIAL_ID', 'TRIAL', 'YEARID', 'YEAR_ID', 'YEAR',
    'SIMID', 'SIM_ID', 'SIMULATION', 'SAMPLE', 'SAMPLEID', 'SAMPLE_ID',
    'ITERATIONID', 'ITERATION_ID', 'ITERATION',
}
_YELT_EVENT_ALIASES = {
    'EVENTID', 'EVENT_ID', 'EVENT', 'EVENTNUM', 'CATALOGNUMBER',
    'CATALOG_NUMBER', 'PERILID', 'PERIL_ID',
}
_YELT_LOSS_ALIASES = {
    'LOSS', 'GROUNDUPLOSS', 'GROUND_UP_LOSS', 'GU_LOSS', 'GULOSS',
    'GROSSLOSS', 'GROSS_LOSS', 'PERSPVALUE', 'MEAN_LOSS', 'EXPECTED_LOSS',
}


def _resolve_yelt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a YELT DataFrame to canonical columns: TrialID, EventID, Loss.

    Accepts many common vendor column names (case-insensitive).  Raises
    ValueError if any required column cannot be resolved.
    """
    upper_map = {c.upper(): c for c in df.columns}
    resolved = {}

    for target, aliases in [
        ('TrialID', _YELT_TRIAL_ALIASES),
        ('EventID', _YELT_EVENT_ALIASES),
        ('Loss',    _YELT_LOSS_ALIASES),
    ]:
        for alias in aliases:
            if alias in upper_map:
                resolved[target] = upper_map[alias]
                break
        if target not in resolved:
            raise ValueError(
                f"Cannot find YELT column '{target}'. "
                f"Expected one of: {sorted(aliases)}.  "
                f"Found columns: {list(df.columns)}"
            )

    out = df.rename(columns={
        resolved['TrialID']: 'TrialID',
        resolved['EventID']: 'EventID',
        resolved['Loss']:    'Loss',
    })
    return out[['TrialID', 'EventID', 'Loss']].copy()


def is_yelt(df: pd.DataFrame) -> bool:
    """Return True if *df* looks like a YELT (has Trial + Event + Loss)."""
    upper = {c.upper() for c in df.columns}
    has_trial = bool(upper & _YELT_TRIAL_ALIASES)
    has_event = bool(upper & _YELT_EVENT_ALIASES)
    has_loss  = bool(upper & _YELT_LOSS_ALIASES)
    # Must have trial column AND NOT look like an ELT (no RATE / STDDEV columns)
    elt_markers = {'RATE', 'STDDEVC', 'STDDEVI', 'SD_C', 'SD_I', 'EXPVALUE',
                   'MAXEXPOSURE', 'MAX_EXPOSURE'}
    looks_like_elt = bool(upper & elt_markers)
    return has_trial and has_event and has_loss and not looks_like_elt


class YELTAggregator:
    """
    Aggregate raw YELT rows into a sorted annual-aggregate-loss array.

    After construction, ``annual_losses`` is a 1-D NumPy array of length
    ``n_trials`` sorted in ascending order — ready for fast percentile lookups.
    """

    __slots__ = ('_annual_losses', '_n_trials', '_yelt')

    def __init__(self, yelt: pd.DataFrame):
        """
        Args:
            yelt: DataFrame with columns [TrialID, EventID, Loss].
        """
        yelt = _resolve_yelt_columns(yelt)
        self._yelt = yelt

        # Sum losses per trial (annual aggregate)
        trial_sums = yelt.groupby('TrialID', sort=False)['Loss'].sum()

        # Some trials may have zero events (no rows) — we need to know the
        # full trial count.  Assume TrialIDs are 1…N with no gaps.
        max_trial = int(yelt['TrialID'].max())
        min_trial = int(yelt['TrialID'].min())
        n_trials = max_trial - min_trial + 1

        # Build full array (trials with no loss → 0)
        annual = np.zeros(n_trials, dtype=np.float64)
        idx = trial_sums.index.values - min_trial  # zero-based offsets
        annual[idx] = trial_sums.values

        self._annual_losses = np.sort(annual)
        self._n_trials = n_trials

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def annual_losses(self) -> np.ndarray:
        """Sorted annual-aggregate-loss array (ascending)."""
        return self._annual_losses

    @property
    def n_trials(self) -> int:
        return self._n_trials

    @property
    def yelt(self) -> pd.DataFrame:
        """Canonical YELT (TrialID, EventID, Loss)."""
        return self._yelt

    def pml_at_rp(self, rp: int) -> float:
        """
        Return the empirical PML for a given return period.

        PML at RP *r* = the loss at the (1 − 1/r) quantile of the annual-
        aggregate-loss distribution.

        E.g. 250-year → 99.6th percentile.
        """
        if self._n_trials == 0:
            return 0.0
        percentile = (1.0 - 1.0 / rp) * 100.0
        return float(np.percentile(self._annual_losses, percentile))

    def aep_at_loss(self, loss: float) -> float:
        """Return empirical annual exceedance probability for a loss threshold."""
        if self._n_trials == 0:
            return 0.0
        return float(np.mean(self._annual_losses > loss))

    def implied_return_period(self, loss: float) -> float:
        """Return implied return period for a loss threshold."""
        aep = self.aep_at_loss(loss)
        return 1.0 / aep if aep > 0 else math.inf

    def aal(self) -> float:
        """Average Annual Loss (mean of annual aggregates)."""
        return float(np.mean(self._annual_losses))

    def ep_curve(self, rps: Optional[Tuple[int, ...]] = None) -> Dict[int, float]:
        """Return {RP: PML} for a set of return periods."""
        if rps is None:
            rps = (10, 25, 50, 100, 250, 500, 1000)
        return {rp: self.pml_at_rp(rp) for rp in rps}


class YELTMarginalEngine:
    """
    Marginal PML engine using full YELT stochastic simulation data.

    Produces **identical output formats** as ``MarginalPMLEngine`` so that
    consumer code (pml_tool.py, workbench.py) works without changes.

    The marginal calculation is exact — no distribution fitting:
        subtract: Marginal = PML(Portfolio) − PML(Portfolio − Account)
        add:      Marginal = PML(Portfolio + Account) − PML(Portfolio)

    Account losses are matched into the portfolio by (TrialID, EventID).
    """

    __slots__ = (
        '_portfolio_yelt', '_portfolio_agg', '_baseline_pmls',
        '_return_periods', '_mode', '_is_initialized',
    )

    def __init__(
        self,
        portfolio_yelt: pd.DataFrame,
        return_periods: Tuple[int, ...] = (50, 100, 250),
        mode: str = PRICING_MODE_SUBTRACT,
    ):
        if mode not in VALID_PRICING_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {VALID_PRICING_MODES}")

        self._mode = mode
        self._return_periods = return_periods
        self._is_initialized = False

        # Normalise and store
        self._portfolio_yelt = _resolve_yelt_columns(portfolio_yelt)

        # Pre-compute baseline
        self._portfolio_agg = YELTAggregator(self._portfolio_yelt)

        self._baseline_pmls: Dict[int, float] = {}
        for rp in return_periods:
            self._baseline_pmls[rp] = self._portfolio_agg.pml_at_rp(rp)

        self._is_initialized = True

    # ------------------------------------------------------------------
    # Properties (mirror MarginalPMLEngine interface)
    # ------------------------------------------------------------------

    @property
    def baseline_pmls(self) -> Dict[int, float]:
        return self._baseline_pmls.copy()

    @property
    def n_portfolio_events(self) -> int:
        return int(self._portfolio_yelt['EventID'].nunique())

    @property
    def return_periods(self) -> Tuple[int, ...]:
        return self._return_periods

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Core pricing
    # ------------------------------------------------------------------

    def price_account(
        self,
        account_yelt: pd.DataFrame,
        return_dict: bool = True,
        include_combined_pml: bool = False,
    ) -> Dict:
        """
        Compute marginal PML for an account YELT.

        Output format is identical to ``MarginalPMLEngine.price_account``.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialised")

        account_yelt = _resolve_yelt_columns(account_yelt)

        if account_yelt.empty:
            zeros = {rp: 0.0 for rp in self._return_periods}
            if include_combined_pml:
                return {
                    'marginal': zeros,
                    'combined': self._baseline_pmls.copy(),
                    'baseline': self._baseline_pmls.copy(),
                }
            return zeros

        # Build combined YELT (portfolio ± account)
        combined_yelt = self._build_combined_yelt(account_yelt)
        combined_agg = YELTAggregator(combined_yelt)

        marginal_results: Dict[int, float] = {}
        combined_results: Dict[int, float] = {}

        for rp in self._return_periods:
            baseline_pml = self._baseline_pmls[rp]
            combined_pml = combined_agg.pml_at_rp(rp)
            combined_results[rp] = combined_pml

            if self._mode == PRICING_MODE_SUBTRACT:
                marginal_results[rp] = baseline_pml - combined_pml
            else:
                marginal_results[rp] = combined_pml - baseline_pml

        if include_combined_pml:
            return {
                'marginal': marginal_results,
                'combined': combined_results,
                'baseline': self._baseline_pmls.copy(),
            }
        return marginal_results

    def price_account_batch(
        self,
        account_yelts: Dict[str, pd.DataFrame],
        progress_callback: Optional[callable] = None,
        include_combined_pml: bool = False,
    ) -> pd.DataFrame:
        """
        Price multiple accounts in batch.

        Output format is identical to ``MarginalPMLEngine.price_account_batch``.
        """
        results = []
        total = len(account_yelts)

        for idx, (account_id, acct_yelt) in enumerate(account_yelts.items()):
            if progress_callback:
                progress_callback(account_id, idx, total)

            try:
                pricing = self.price_account(
                    acct_yelt, include_combined_pml=include_combined_pml,
                )
                row: Dict = {'AccountID': account_id}

                if include_combined_pml:
                    for rp in self._return_periods:
                        row[f'Baseline_{rp}'] = pricing['baseline'][rp]
                    for rp in self._return_periods:
                        row[f'Combined_{rp}'] = pricing['combined'][rp]
                    for rp in self._return_periods:
                        row[f'RI_{rp}'] = pricing['marginal'][rp]
                else:
                    for rp, value in pricing.items():
                        row[f'RI_{rp}'] = value

                results.append(row)
            except Exception as e:
                warnings.warn(f"Error pricing account {account_id}: {e}")
                row = {'AccountID': account_id}
                for rp in self._return_periods:
                    row[f'RI_{rp}'] = np.nan
                    if include_combined_pml:
                        row[f'Baseline_{rp}'] = np.nan
                        row[f'Combined_{rp}'] = np.nan
                results.append(row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_combined_yelt(self, account_yelt: pd.DataFrame) -> pd.DataFrame:
        """
        Combine portfolio YELT with account YELT.

        subtract mode → portfolio losses minus account losses (per trial/event)
        add mode      → portfolio losses plus account losses
        """
        port = self._portfolio_yelt
        acct = account_yelt

        if self._mode == PRICING_MODE_SUBTRACT:
            # Merge on (TrialID, EventID); keep all portfolio rows
            merged = port.merge(
                acct, on=['TrialID', 'EventID'], how='left', suffixes=('_port', '_acct'),
            )
            merged['Loss_acct'] = merged['Loss_acct'].fillna(0.0)
            merged['Loss'] = np.maximum(0.0, merged['Loss_port'] - merged['Loss_acct'])
            return merged[['TrialID', 'EventID', 'Loss']]
        else:
            # Add mode — outer join so new-to-portfolio events appear
            merged = port.merge(
                acct, on=['TrialID', 'EventID'], how='outer', suffixes=('_port', '_acct'),
            )
            merged['Loss_port'] = merged['Loss_port'].fillna(0.0)
            merged['Loss_acct'] = merged['Loss_acct'].fillna(0.0)
            merged['Loss'] = merged['Loss_port'] + merged['Loss_acct']
            return merged[['TrialID', 'EventID', 'Loss']]

    def get_diagnostics(self) -> Dict:
        return {
            'mode': self._mode,
            'data_type': 'YELT',
            'n_trials': self._portfolio_agg.n_trials,
            'n_portfolio_events': self.n_portfolio_events,
            'baseline_pmls': self._baseline_pmls.copy(),
            'baseline_aal': self._portfolio_agg.aal(),
            'return_periods': self._return_periods,
        }


def create_pricing_engine(
    portfolio_elt: pd.DataFrame,
    return_periods: Tuple[int, ...] = (50, 100, 250),
    use_poisson: bool = True,
    mode: str = PRICING_MODE_SUBTRACT,
    correlation: Optional[CorrelationConfig] = None,
):
    """
    Factory function to create a configured pricing engine.

    Auto-detects whether *portfolio_elt* is an ELT or a YELT:
        - ELT columns (RATE, STDDEVC, …) → MarginalPMLEngine (Method of Moments)
        - YELT columns (TrialID, EventID, Loss) → YELTMarginalEngine (empirical)

    Both engine types expose the same API:
        engine.price_account(account_df)
        engine.price_account_batch(accounts_dict)
        engine.baseline_pmls
        engine.return_periods
        engine.mode

    Args:
        portfolio_elt: Portfolio data — either ELT or YELT format.
        return_periods: Return periods to compute.
        use_poisson: Poisson AEP flag (ELT path only; ignored for YELT).
        mode: 'subtract' or 'add'.
        correlation: Optional CorrelationConfig to enable Gaussian copula
            simulation on the ELT path.  Ignored for YELT (correlation is
            already embedded in the trial structure).

    Returns:
        MarginalPMLEngine (ELT) or YELTMarginalEngine (YELT).
    """
    if is_yelt(portfolio_elt):
        return YELTMarginalEngine(portfolio_elt, return_periods, mode=mode)

    config = MomentsConfig(use_poisson_aep=use_poisson)
    return MarginalPMLEngine(portfolio_elt, return_periods, config, mode=mode,
                             correlation=correlation)


# =============================================================================
# EXAMPLE USAGE AND BENCHMARKS
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("MARGINAL PML PRICING ENGINE - PERFORMANCE DEMO")
    print("=" * 70)
    
    # Create sample portfolio data
    np.random.seed(42)
    n_events = 5000  # Larger portfolio for realistic benchmark
    
    portfolio_elt = pd.DataFrame({
        'EVENTID': range(1, n_events + 1),
        'RATE': np.random.uniform(0.001, 0.01, n_events),
        'PERSPVALUE': np.random.exponential(1e6, n_events),
        'EXPVALUE': np.random.exponential(5e6, n_events),
        'STDDEVC': np.random.exponential(2e5, n_events),
        'STDDEVI': np.random.exponential(3e5, n_events),
    })
    portfolio_elt['EXPVALUE'] = portfolio_elt[['PERSPVALUE', 'EXPVALUE']].max(axis=1) * 1.5
    
    print(f"\n📊 Portfolio: {len(portfolio_elt):,} events")
    
    # Generate sample accounts
    n_accounts = 20
    accounts = {}
    
    print(f"\n📋 Generating {n_accounts} sample accounts...")
    for i in range(n_accounts):
        n_account_events = np.random.randint(50, 300)
        account_events = np.random.choice(n_events, n_account_events, replace=False) + 1
        
        accounts[f"ACCT_{i+1:04d}"] = pd.DataFrame({
            'EVENTID': account_events,
            'PERSPVALUE': np.random.exponential(1e5, n_account_events),
            'EXPVALUE': np.random.exponential(5e5, n_account_events),
            'STDDEVC': np.random.exponential(2e4, n_account_events),
            'STDDEVI': np.random.exponential(3e4, n_account_events),
        })
    
    # ==========================================================================
    # TEST SUBTRACT MODE (RI Pricing - existing accounts)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODE: SUBTRACT (RI Pricing)")
    print("Marginal = PML(Portfolio) - PML(Portfolio - Account)")
    print("=" * 70)
    
    t0 = time.perf_counter()
    engine_subtract = create_pricing_engine(
        portfolio_elt,
        return_periods=(50, 100, 250),
        use_poisson=True,
        mode='subtract'
    )
    init_time = time.perf_counter() - t0
    
    print(f"\n🔧 Engine initialized in {init_time:.3f}s")
    print(f"   Mode: {engine_subtract.mode}")
    print(f"   Valid events: {engine_subtract._baseline_calc.n_events:,}")
    print(f"   Baseline PMLs:")
    for rp, pml in engine_subtract.baseline_pmls.items():
        print(f"      {rp:>4}yr: ${pml:>15,.0f}")
    
    # Price accounts
    t0 = time.perf_counter()
    results_subtract = []
    for acct_id, acct_elt in accounts.items():
        result = engine_subtract.price_account(acct_elt)
        results_subtract.append({'AccountID': acct_id, **{f'RI_{rp}': v for rp, v in result.items()}})
    subtract_time = time.perf_counter() - t0
    
    results_subtract_df = pd.DataFrame(results_subtract)
    print(f"\n⚡ Priced {n_accounts} accounts in {subtract_time:.3f}s ({subtract_time/n_accounts*1000:.1f}ms/account)")
    print(f"\n📊 Subtract Mode Results:")
    for rp in [50, 100, 250]:
        col = f'RI_{rp}'
        print(f"   {rp}yr: min=${results_subtract_df[col].min():,.0f}, "
              f"mean=${results_subtract_df[col].mean():,.0f}, "
              f"max=${results_subtract_df[col].max():,.0f}")
    
    # ==========================================================================
    # TEST ADD MODE (New Business Pricing)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODE: ADD (New Business Pricing)")
    print("Marginal = PML(Portfolio + Account) - PML(Portfolio)")
    print("=" * 70)
    
    t0 = time.perf_counter()
    engine_add = create_pricing_engine(
        portfolio_elt,
        return_periods=(50, 100, 250),
        use_poisson=True,
        mode='add'
    )
    init_time = time.perf_counter() - t0
    
    print(f"\n🔧 Engine initialized in {init_time:.3f}s")
    print(f"   Mode: {engine_add.mode}")
    
    # Price accounts
    t0 = time.perf_counter()
    results_add = []
    for acct_id, acct_elt in accounts.items():
        result = engine_add.price_account(acct_elt)
        results_add.append({'AccountID': acct_id, **{f'RI_{rp}': v for rp, v in result.items()}})
    add_time = time.perf_counter() - t0
    
    results_add_df = pd.DataFrame(results_add)
    print(f"\n⚡ Priced {n_accounts} accounts in {add_time:.3f}s ({add_time/n_accounts*1000:.1f}ms/account)")
    print(f"\n📊 Add Mode Results:")
    for rp in [50, 100, 250]:
        col = f'RI_{rp}'
        print(f"   {rp}yr: min=${results_add_df[col].min():,.0f}, "
              f"mean=${results_add_df[col].mean():,.0f}, "
              f"max=${results_add_df[col].max():,.0f}")
    
    # ==========================================================================
    # COMPARE MODES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODE COMPARISON")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'AccountID': results_subtract_df['AccountID'],
        'Subtract_100': results_subtract_df['RI_100'],
        'Add_100': results_add_df['RI_100'],
    })
    comparison['Difference'] = comparison['Add_100'] - comparison['Subtract_100']
    
    print("\nFirst 10 accounts (100yr RP):")
    print(comparison.head(10).to_string(index=False))
    
    print(f"\n📈 Average difference (Add - Subtract): ${comparison['Difference'].mean():,.0f}")
    print(f"   This difference reflects how new-to-portfolio events")
    print(f"   and aggregation asymmetry affect marginal pricing.")
    
    # Verify Poisson AEP formula
    print(f"\n🔬 Verifying Poisson AEP Formula:")
    print(f"   AEP = 1 - exp(-Σ(Rate × ProbExceed))")
    test_pml = engine_subtract.baseline_pmls[100]
    test_rp = engine_subtract._baseline_calc.implied_return_period(test_pml)
    print(f"   For PML=${test_pml:,.0f}: RP={test_rp:.2f}yr (target=100yr)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
