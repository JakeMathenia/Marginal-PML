#!/usr/bin/env python
"""
Marginal PML Pricing Tool - Consumer CLI Wrapper

A user-friendly command-line interface for the Marginal PML Pricing Engine.
Designed for underwriters and pricing analysts who need fast, accurate
marginal PML calculations without coding expertise.

Usage:
    pml-tool --portfolio base.csv --quote new_deal.xlsx
    pml-tool --portfolio portfolio.xlsx --quote account.csv --output results.xlsx
    pml-tool --portfolio base.csv --quote-folder ./accounts/ --batch

Features:
    - Automatic file format detection (Excel, CSV)
    - Smart column mapping for common insurance headers
    - Batch processing for multiple accounts
    - Summary report with price recommendations
    - 15% Return on Marginal Capital pricing
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import yaml

# Import the core pricing engine
from marginal_pml_kernel import (
    create_pricing_engine,
    MarginalPMLEngine,
    MomentsConfig,
    calculate_portfolio_minus_account
)


# =============================================================================
# COLUMN MAPPING CONFIGURATION
# =============================================================================

# Standard column names expected by the engine
STANDARD_COLUMNS = {
    'EVENTID': 'EVENTID',
    'RATE': 'RATE',
    'PERSPVALUE': 'PERSPVALUE',  # Mean/Expected Loss
    'EXPVALUE': 'EXPVALUE',      # Max Exposure
    'STDDEVC': 'STDDEVC',        # Correlated Std Dev
    'STDDEVI': 'STDDEVI',        # Independent Std Dev
}

# Common alternative names used in the insurance industry
# Maps from common names → standard name
COLUMN_ALIASES = {
    # Event ID variations
    'EVENTID': 'EVENTID',
    'EVENT_ID': 'EVENTID',
    'EVENT ID': 'EVENTID',
    'EVENT': 'EVENTID',
    'ID': 'EVENTID',
    'EVENTNUM': 'EVENTID',
    'EVENT_NUM': 'EVENTID',
    'EVENT_NUMBER': 'EVENTID',
    
    # Rate variations
    'RATE': 'RATE',
    'PORT_RATE': 'RATE',
    'PORTRATE': 'RATE',
    'ANNUAL_RATE': 'RATE',
    'ANNUALRATE': 'RATE',
    'FREQUENCY': 'RATE',
    'FREQ': 'RATE',
    'LAMBDA': 'RATE',
    'OCCURRENCE_RATE': 'RATE',
    'OCC_RATE': 'RATE',
    
    # Mean/Expected Loss variations
    'PERSPVALUE': 'PERSPVALUE',
    'MEAN': 'PERSPVALUE',
    'MEANLOSS': 'PERSPVALUE',
    'MEAN_LOSS': 'PERSPVALUE',
    'EXPECTED_LOSS': 'PERSPVALUE',
    'EXPECTEDLOSS': 'PERSPVALUE',
    'EXP_LOSS': 'PERSPVALUE',
    'EXPLOSS': 'PERSPVALUE',
    'AVG_LOSS': 'PERSPVALUE',
    'AVGLOSS': 'PERSPVALUE',
    'AVERAGE_LOSS': 'PERSPVALUE',
    'MU': 'PERSPVALUE',
    'GROUND_UP_LOSS': 'PERSPVALUE',
    'GU_LOSS': 'PERSPVALUE',
    'GULOSS': 'PERSPVALUE',
    
    # Max Exposure variations
    'EXPVALUE': 'EXPVALUE',
    'EXPOSURE': 'EXPVALUE',
    'MAX_EXPOSURE': 'EXPVALUE',
    'MAXEXPOSURE': 'EXPVALUE',
    'MAX_EXP': 'EXPVALUE',
    'MAXEXP': 'EXPVALUE',
    'TIV': 'EXPVALUE',
    'TOTAL_INSURED_VALUE': 'EXPVALUE',
    'SUM_INSURED': 'EXPVALUE',
    'SUMINSURED': 'EXPVALUE',
    'LIMIT': 'EXPVALUE',
    'POLICY_LIMIT': 'EXPVALUE',
    'MAX_LOSS': 'EXPVALUE',
    'MAXLOSS': 'EXPVALUE',
    
    # Correlated Std Dev variations
    'STDDEVC': 'STDDEVC',
    'STDEV_C': 'STDDEVC',
    'STDEVC': 'STDDEVC',
    'SD_C': 'STDDEVC',
    'SDC': 'STDDEVC',
    'CORR_STD': 'STDDEVC',
    'CORRELATED_STD': 'STDDEVC',
    'CORRELATED_STDEV': 'STDDEVC',
    'STD_CORR': 'STDDEVC',
    'SIGMA_C': 'STDDEVC',
    'SIGMAC': 'STDDEVC',
    
    # Independent Std Dev variations
    'STDDEVI': 'STDDEVI',
    'STDEV_I': 'STDDEVI',
    'STDEVI': 'STDDEVI',
    'SD_I': 'STDDEVI',
    'SDI': 'STDDEVI',
    'IND_STD': 'STDDEVI',
    'INDEPENDENT_STD': 'STDDEVI',
    'INDEPENDENT_STDEV': 'STDDEVI',
    'STD_IND': 'STDDEVI',
    'SIGMA_I': 'STDDEVI',
    'SIGMAI': 'STDDEVI',
    'IDIO_STD': 'STDDEVI',
    'IDIOSYNCRATIC_STD': 'STDDEVI',
}


# =============================================================================
# FILE INGESTOR
# =============================================================================

class FileIngestor:
    """
    Smart file ingestor that handles Excel and CSV files with automatic
    format detection and column mapping.
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._column_mapper = ColumnMapper(verbose=verbose)
    
    def ingest(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        is_portfolio: bool = False
    ) -> pd.DataFrame:
        """
        Ingest a file and return a standardized DataFrame.
        
        Args:
            file_path: Path to CSV or Excel file
            sheet_name: Sheet name for Excel files (optional)
            is_portfolio: If True, require RATE column
        
        Returns:
            DataFrame with standardized column names
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Read file based on extension
        if ext == '.csv':
            df = self._read_csv(path)
        else:
            df = self._read_excel(path, sheet_name)
        
        if df.empty:
            raise ValueError(f"File is empty: {file_path}")
        
        if self.verbose:
            print(f"  Loaded {len(df):,} rows from {path.name}")
        
        # Map columns to standard names
        df = self._column_mapper.map_columns(df, is_portfolio=is_portfolio)
        
        # Validate required columns
        self._validate_columns(df, is_portfolio)
        
        return df
    
    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read CSV with automatic encoding and delimiter detection."""
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # Try to detect delimiter
                with open(path, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                
                if '\t' in first_line:
                    delimiter = '\t'
                elif ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = ','
                
                return pd.read_csv(path, encoding=encoding, delimiter=delimiter)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ValueError(f"Error reading CSV: {e}")
        
        raise ValueError(f"Could not determine encoding for: {path}")
    
    def _read_excel(
        self,
        path: Path,
        sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Read Excel file, optionally from specific sheet."""
        try:
            if sheet_name:
                return pd.read_excel(path, sheet_name=sheet_name)
            else:
                # Read first sheet
                return pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")
    
    def _validate_columns(self, df: pd.DataFrame, is_portfolio: bool) -> None:
        """Validate that required columns are present."""
        required = ['EVENTID', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI']
        
        if is_portfolio:
            required.append('RATE')
        
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            available = ', '.join(df.columns.tolist())
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available columns: {available}\n"
                f"Use --map-columns to see column mapping options."
            )


class ColumnMapper:
    """
    Smart column mapper that auto-detects common insurance column names
    and maps them to the standard format expected by the engine.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Build reverse lookup: uppercase alias → standard name
        self._alias_map = {
            alias.upper(): standard
            for alias, standard in COLUMN_ALIASES.items()
        }
    
    def map_columns(
        self,
        df: pd.DataFrame,
        is_portfolio: bool = False,
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Map DataFrame columns to standard names.
        
        Args:
            df: Input DataFrame
            is_portfolio: If True, require RATE column
            custom_mapping: Optional custom column name mapping
        
        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        
        # Apply custom mapping first if provided
        if custom_mapping:
            df = df.rename(columns=custom_mapping)
        
        # Auto-detect and map columns
        mapping = {}
        matched_standards = set()
        
        for col in df.columns:
            col_upper = col.upper().strip()
            
            # Check if already standard
            if col_upper in STANDARD_COLUMNS:
                if col != col_upper:
                    mapping[col] = col_upper
                matched_standards.add(col_upper)
                continue
            
            # Check aliases
            if col_upper in self._alias_map:
                standard = self._alias_map[col_upper]
                if standard not in matched_standards:
                    mapping[col] = standard
                    matched_standards.add(standard)
                    if self.verbose:
                        print(f"  Column mapping: '{col}' → '{standard}'")
        
        if mapping:
            df = df.rename(columns=mapping)
        
        return df
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect which columns can be auto-mapped.
        
        Returns dict of {original_name: standard_name}
        """
        detected = {}
        
        for col in df.columns:
            col_upper = col.upper().strip()
            
            if col_upper in STANDARD_COLUMNS:
                detected[col] = col_upper
            elif col_upper in self._alias_map:
                detected[col] = self._alias_map[col_upper]
        
        return detected
    
    def get_unmapped_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of columns that couldn't be auto-mapped."""
        detected = self.detect_columns(df)
        return [col for col in df.columns if col not in detected]


# =============================================================================
# SUMMARY REPORT GENERATOR
# =============================================================================

class SummaryReportGenerator:
    """
    Generate professional summary reports with price recommendations.
    
    Pricing Formula:
        Technical Price = AAL + (Marginal PML × Capital Rate × ROC)
        
    Where:
        - AAL = Average Annual Loss (account-level)
        - Marginal PML = Marginal impact at specified RP
        - Capital Rate = Capital required per $ of marginal PML (e.g., 0.25 for 250yr)
        - ROC = Return on Capital target (default 15%)
    """
    
    # Default pricing parameters
    DEFAULT_ROC = 0.15  # 15% Return on Capital
    DEFAULT_CAPITAL_RATES = {
        50: 0.20,   # 20% capital at 1-in-50
        100: 0.25,  # 25% capital at 1-in-100
        250: 0.35,  # 35% capital at 1-in-250
    }
    
    def __init__(
        self,
        roc: float = DEFAULT_ROC,
        capital_rates: Optional[Dict[int, float]] = None
    ):
        self.roc = roc
        self.capital_rates = capital_rates or self.DEFAULT_CAPITAL_RATES
    
    def calculate_price_recommendation(
        self,
        marginal_pmls: Dict[int, float],
        account_aal: float = 0.0,
        pricing_rp: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate price recommendation based on marginal capital.
        
        Technical Price = AAL + (Marginal PML × Capital Rate × ROC)
        
        Args:
            marginal_pmls: Dict of {RP: marginal_pml}
            account_aal: Account's Average Annual Loss
            pricing_rp: Return period to use for pricing (default: 100yr)
        
        Returns:
            Dict with pricing components and recommendation
        """
        if pricing_rp not in marginal_pmls:
            raise ValueError(f"RP {pricing_rp} not in marginal PMls")
        
        marginal_pml = marginal_pmls.get(pricing_rp, 0)
        capital_rate = self.capital_rates.get(pricing_rp, 0.25)
        
        # Marginal capital required
        marginal_capital = max(0, marginal_pml * capital_rate)
        
        # Return on marginal capital
        capital_charge = marginal_capital * self.roc
        
        # Technical price components
        technical_price = account_aal + capital_charge
        
        return {
            'account_aal': account_aal,
            'pricing_rp': pricing_rp,
            'marginal_pml': marginal_pml,
            'capital_rate': capital_rate,
            'marginal_capital': marginal_capital,
            'roc': self.roc,
            'capital_charge': capital_charge,
            'technical_price': technical_price,
        }
    
    def generate_report(
        self,
        results_df: pd.DataFrame,
        portfolio_name: str,
        baseline_pmls: Dict[int, float],
        output_path: str,
        pricing_rp: int = 100
    ) -> str:
        """
        Generate comprehensive Excel report with multiple sheets.
        
        Sheets:
            1. Summary - High-level overview and totals
            2. Account Details - Per-account pricing
            3. Methodology - Explanation of calculations
        
        Args:
            results_df: DataFrame with columns [AccountID, RI_50, RI_100, RI_250, ...]
            portfolio_name: Name of portfolio for report header
            baseline_pmls: Dict of baseline portfolio PMLs
            output_path: Path for output Excel file
            pricing_rp: Return period used for pricing
        
        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        
        # Prepare results with price recommendations
        results = results_df.copy()
        
        # Calculate price recommendations for each account
        recommendations = []
        for _, row in results.iterrows():
            marginals = {
                rp: row.get(f'RI_{rp}', 0)
                for rp in [50, 100, 250]
                if f'RI_{rp}' in row
            }
            
            # Use AAL if available, otherwise estimate from marginal
            aal = row.get('AAL', row.get('AccountAAL', 0)) or 0
            
            try:
                rec = self.calculate_price_recommendation(
                    marginals, 
                    account_aal=aal,
                    pricing_rp=pricing_rp
                )
                recommendations.append(rec)
            except Exception:
                recommendations.append({
                    'account_aal': aal,
                    'pricing_rp': pricing_rp,
                    'marginal_pml': 0,
                    'capital_rate': 0,
                    'marginal_capital': 0,
                    'roc': self.roc,
                    'capital_charge': 0,
                    'technical_price': aal,
                })
        
        rec_df = pd.DataFrame(recommendations)
        results = pd.concat([results.reset_index(drop=True), rec_df], axis=1)
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self._write_summary_sheet(
                writer, results, portfolio_name, baseline_pmls, pricing_rp
            )
            self._write_details_sheet(writer, results)
            self._write_methodology_sheet(writer, pricing_rp)
        
        return str(output_path)
    
    def _write_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        results: pd.DataFrame,
        portfolio_name: str,
        baseline_pmls: Dict[int, float],
        pricing_rp: int
    ) -> None:
        """Write summary sheet with high-level metrics."""
        summary_data = [
            ['MARGINAL PML PRICING REPORT', ''],
            ['', ''],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Portfolio:', portfolio_name],
            ['Pricing RP:', f'{pricing_rp}-year'],
            ['Return on Capital:', f'{self.roc:.0%}'],
            ['', ''],
            ['BASELINE PORTFOLIO PMLs', ''],
        ]
        
        for rp, pml in sorted(baseline_pmls.items()):
            summary_data.append([f'  {rp}-year PML:', f'${pml:,.0f}'])
        
        summary_data.extend([
            ['', ''],
            ['AGGREGATE RESULTS', ''],
            ['Total Accounts Priced:', len(results)],
            ['Total Marginal Capital:', f'${results["marginal_capital"].sum():,.0f}'],
            ['Total Capital Charge:', f'${results["capital_charge"].sum():,.0f}'],
            ['Total Technical Price:', f'${results["technical_price"].sum():,.0f}'],
            ['', ''],
            ['TOP 10 ACCOUNTS BY MARGINAL CAPITAL', ''],
        ])
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False, header=False)
        
        # Add top accounts table
        top_accounts = results.nlargest(10, 'marginal_capital')[
            ['AccountID', 'marginal_pml', 'marginal_capital', 'capital_charge', 'technical_price']
        ].copy()
        top_accounts.columns = ['Account ID', 'Marginal PML', 'Marginal Capital', 
                               'Capital Charge', 'Technical Price']
        
        # Write starting after summary
        start_row = len(summary_data) + 2
        top_accounts.to_excel(
            writer, sheet_name='Summary', 
            index=False, startrow=start_row
        )
    
    def _write_details_sheet(
        self,
        writer: pd.ExcelWriter,
        results: pd.DataFrame
    ) -> None:
        """Write detailed account-level results."""
        # Select and rename columns for clarity
        detail_cols = ['AccountID']
        
        # Add baseline, combined, and marginal columns if present (in that order)
        for rp in [50, 100, 250]:
            baseline_col = f'Baseline_{rp}'
            combined_col = f'Combined_{rp}'
            marginal_col = f'RI_{rp}'
            
            if baseline_col in results.columns:
                detail_cols.append(baseline_col)
            if combined_col in results.columns:
                detail_cols.append(combined_col)
            if marginal_col in results.columns:
                detail_cols.append(marginal_col)
        
        # Add pricing columns
        pricing_cols = [
            'account_aal', 'marginal_pml', 'capital_rate',
            'marginal_capital', 'capital_charge', 'technical_price'
        ]
        detail_cols.extend([c for c in pricing_cols if c in results.columns])
        
        details = results[detail_cols].copy()
        
        # Rename for readability
        rename_map = {
            'AccountID': 'Account ID',
            'Baseline_50': 'Portfolio PML (50yr)',
            'Combined_50': 'Portfolio±Account PML (50yr)',
            'RI_50': 'Marginal PML (50yr)',
            'Baseline_100': 'Portfolio PML (100yr)',
            'Combined_100': 'Portfolio±Account PML (100yr)',
            'RI_100': 'Marginal PML (100yr)',
            'Baseline_250': 'Portfolio PML (250yr)',
            'Combined_250': 'Portfolio±Account PML (250yr)',
            'RI_250': 'Marginal PML (250yr)',
            'account_aal': 'Account AAL',
            'marginal_pml': 'Pricing Marginal PML',
            'capital_rate': 'Capital Rate',
            'marginal_capital': 'Marginal Capital',
            'capital_charge': 'Capital Charge',
            'technical_price': 'Technical Price',
        }
        details = details.rename(columns=rename_map)
        
        details.to_excel(writer, sheet_name='Account Details', index=False)
    
    def _write_methodology_sheet(
        self,
        writer: pd.ExcelWriter,
        pricing_rp: int
    ) -> None:
        """Write methodology explanation sheet."""
        methodology = [
            ['METHODOLOGY', ''],
            ['', ''],
            ['Overview', ''],
            ['This report calculates the marginal impact on portfolio PML from each account,', ''],
            ['then derives a price recommendation based on Return on Marginal Capital.', ''],
            ['', ''],
            ['Key Formulas', ''],
            ['', ''],
            ['1. Marginal PML', ''],
            ['   Marginal PML = PML(Portfolio) - PML(Portfolio - Account)', ''],
            ['   This measures how much the account contributes to portfolio tail risk.', ''],
            ['', ''],
            ['2. Marginal Capital', ''],
            [f'   Marginal Capital = Marginal PML × Capital Rate', ''],
            [f'   Capital Rate at {pricing_rp}yr = {self.capital_rates.get(pricing_rp, 0.25):.0%}', ''],
            ['', ''],
            ['3. Capital Charge', ''],
            [f'   Capital Charge = Marginal Capital × ROC', ''],
            [f'   Return on Capital (ROC) = {self.roc:.0%}', ''],
            ['', ''],
            ['4. Technical Price', ''],
            ['   Technical Price = Account AAL + Capital Charge', ''],
            ['', ''],
            ['Distribution Assumptions', ''],
            ['- Loss severity follows a Beta distribution', ''],
            ['- Event frequency follows a Poisson process', ''],
            ['- AEP = 1 - exp(-λ) where λ = Σ(Rate × Prob Exceed)', ''],
            ['', ''],
            ['Aggregation Rules', ''],
            ['- Mean Loss: Additive', ''],
            ['- Max Exposure: Additive', ''],
            ['- Correlated Std Dev: Additive', ''],
            ['- Independent Std Dev: Root Sum of Squares √(Σ SD²)', ''],
        ]
        
        meth_df = pd.DataFrame(methodology, columns=['Description', ''])
        meth_df.to_excel(writer, sheet_name='Methodology', index=False, header=False)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='pml-tool',
        description='Marginal PML Pricing Tool - Fast marginal PML calculations for insurance pricing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pml-tool --portfolio base.csv --quote new_deal.xlsx
  pml-tool --portfolio portfolio.xlsx --quote account.csv --output results.xlsx
  pml-tool --portfolio base.csv --quote-folder ./accounts/ --batch
  pml-tool --portfolio base.csv --quote deal.csv --roc 0.20 --pricing-rp 250  
  # Use 'add' mode for new business pricing
  pml-tool --portfolio base.csv --quote new_deal.xlsx --mode add
  
  # Use 'subtract' mode for RI pricing (default)
  pml-tool --portfolio base.csv --quote existing_account.csv --mode subtract
For more information, visit: https://github.com/your-org/marginal-pml
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--portfolio', '-p',
        required=True,
        help='Path to portfolio/baseline ELT file (CSV or Excel)'
    )
    
    # Quote arguments (mutually exclusive)
    quote_group = parser.add_mutually_exclusive_group(required=True)
    quote_group.add_argument(
        '--quote', '-q',
        help='Path to single account/quote ELT file'
    )
    quote_group.add_argument(
        '--quote-folder', '-Q',
        help='Path to folder containing multiple account files (batch mode)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: results_YYYYMMDD_HHMMSS.xlsx)'
    )
    
    # Pricing parameters
    parser.add_argument(
        '--roc',
        type=float,
        default=0.15,
        help='Return on Capital target (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--pricing-rp',
        type=int,
        default=100,
        choices=[50, 100, 250],
        help='Return period for pricing (default: 100)'
    )
    parser.add_argument(
        '--return-periods',
        type=str,
        default='50,100,250',
        help='Comma-separated return periods to calculate (default: 50,100,250)'
    )
    
    # Processing options
    parser.add_argument(
        '--sheet',
        help='Sheet name for Excel files (uses first sheet if not specified)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode for --quote-folder'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['subtract', 'add'],
        default='subtract',
        help="Pricing mode: 'subtract' for RI pricing (existing accounts), "
             "'add' for new business pricing (default: subtract)"
    )
    parser.add_argument(
        '--show-combined-pml',
        action='store_true',
        help='Include baseline and combined PML values in results (in addition to marginal)'
    )
    
    # Column mapping
    parser.add_argument(
        '--map-columns',
        action='store_true',
        help='Show detected column mappings and exit'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser


# =============================================================================
# CONFIGURATION FILE SUPPORT
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / 'pml_config.yaml'


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Optional[Dict]:
    """
    Load settings from a YAML config file.
    
    Returns:
        Dict of settings, or None if file not found.
    """
    if not config_path.exists():
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    return cfg


def config_to_args(cfg: Dict) -> List[str]:
    """
    Convert a config dict into CLI-style args list.
    
    This lets us reuse the existing argparse/main logic without
    duplicating all the validation and processing code.
    """
    args = []
    
    # Required: portfolio
    portfolio = cfg.get('portfolio', '')
    if portfolio:
        args.extend(['--portfolio', str(portfolio)])
    
    # Quote: single file or folder
    quote = cfg.get('quote', '')
    quote_folder = cfg.get('quote_folder', '')
    
    if quote:
        args.extend(['--quote', str(quote)])
    elif quote_folder:
        args.extend(['--quote-folder', str(quote_folder)])
    
    # Output
    output = cfg.get('output', '')
    if output:
        args.extend(['--output', str(output)])
    
    # Sheet
    sheet = cfg.get('sheet', '')
    if sheet:
        args.extend(['--sheet', str(sheet)])
    
    # Mode
    mode = cfg.get('mode', 'subtract')
    if mode:
        args.extend(['--mode', str(mode)])
    
    # Return periods
    rps = cfg.get('return_periods', '50,100,250')
    if rps:
        # Handle both "50, 100, 250" and [50, 100, 250]
        if isinstance(rps, list):
            rps = ','.join(str(r) for r in rps)
        args.extend(['--return-periods', str(rps).replace(' ', '')])
    
    # ROC
    roc = cfg.get('roc')
    if roc is not None:
        args.extend(['--roc', str(roc)])
    
    # Pricing RP
    pricing_rp = cfg.get('pricing_rp')
    if pricing_rp is not None:
        args.extend(['--pricing-rp', str(pricing_rp)])
    
    # Booleans
    if cfg.get('show_combined_pml', False):
        args.append('--show-combined-pml')
    
    if cfg.get('verbose', False):
        args.append('--verbose')
    
    if cfg.get('quiet', False):
        args.append('--quiet')
    
    if cfg.get('batch', False):
        args.append('--batch')
    
    if cfg.get('map_columns', False):
        args.append('--map-columns')
    
    return args


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    
    # If no CLI args provided, try loading from config file
    if args is None and len(sys.argv) == 1:
        cfg = load_config()
        if cfg is not None:
            config_args = config_to_args(cfg)
            if config_args:
                print(f"📄 Loading settings from {DEFAULT_CONFIG_PATH.name}")
                args = config_args
    
    opts = parser.parse_args(args)
    
    # Configure output
    verbose = opts.verbose and not opts.quiet
    quiet = opts.quiet
    
    def log(msg: str) -> None:
        if not quiet:
            print(msg)
    
    def vlog(msg: str) -> None:
        if verbose:
            print(msg)
    
    try:
        # Initialize ingestor
        ingestor = FileIngestor(verbose=verbose)
        
        # === Handle --map-columns mode ===
        if opts.map_columns:
            log("Analyzing column mappings...")
            mapper = ColumnMapper(verbose=True)
            
            # Show portfolio columns
            log(f"\nPortfolio file: {opts.portfolio}")
            port_df = pd.read_csv(opts.portfolio) if opts.portfolio.endswith('.csv') else pd.read_excel(opts.portfolio)
            detected = mapper.detect_columns(port_df)
            unmapped = mapper.get_unmapped_columns(port_df)
            
            log("\nDetected mappings:")
            for orig, std in detected.items():
                log(f"  {orig} → {std}")
            
            if unmapped:
                log("\nUnmapped columns (will be ignored):")
                for col in unmapped:
                    log(f"  {col}")
            
            return 0
        
        # === Load Portfolio ===
        log("=" * 60)
        log("MARGINAL PML PRICING TOOL")
        log("=" * 60)
        log(f"\n📁 Loading portfolio: {opts.portfolio}")
        
        t0 = time.perf_counter()
        portfolio_df = ingestor.ingest(
            opts.portfolio,
            sheet_name=opts.sheet,
            is_portfolio=True
        )
        load_time = time.perf_counter() - t0
        vlog(f"   Loaded in {load_time:.2f}s")
        
        # Parse return periods
        return_periods = tuple(int(rp.strip()) for rp in opts.return_periods.split(','))
        
        # === Initialize Engine ===
        mode_desc = "RI Pricing (subtract)" if opts.mode == 'subtract' else "New Business (add)"
        log(f"\n🔧 Initializing pricing engine ({mode_desc})...")
        t0 = time.perf_counter()
        engine = create_pricing_engine(
            portfolio_df,
            return_periods=return_periods,
            use_poisson=True,
            mode=opts.mode
        )
        init_time = time.perf_counter() - t0
        
        log(f"   Mode: {engine.mode}")
        log(f"   Portfolio events: {engine.n_portfolio_events:,}")
        log(f"   Valid events: {engine._baseline_calc.n_events:,}")
        log(f"   Initialization: {init_time:.2f}s")
        log(f"\n📊 Baseline PMLs:")
        for rp, pml in engine.baseline_pmls.items():
            log(f"   {rp:>4}yr: ${pml:>15,.0f}")
        
        # === Process Quotes ===
        results = []
        account_files = []
        
        if opts.quote:
            # Single quote mode
            account_files = [(Path(opts.quote).stem, opts.quote)]
        else:
            # Batch mode
            quote_folder = Path(opts.quote_folder)
            if not quote_folder.is_dir():
                raise ValueError(f"Quote folder not found: {opts.quote_folder}")
            
            for ext in ['.csv', '.xlsx', '.xls']:
                for f in quote_folder.glob(f'*{ext}'):
                    account_files.append((f.stem, str(f)))
        
        n_accounts = len(account_files)
        log(f"\n⚡ Processing {n_accounts} account(s)...")
        
        t0 = time.perf_counter()
        for idx, (account_id, account_path) in enumerate(account_files, 1):
            if verbose:
                log(f"   [{idx}/{n_accounts}] {account_id}...")
            
            try:
                account_df = ingestor.ingest(
                    account_path,
                    sheet_name=opts.sheet,
                    is_portfolio=False
                )
                
                pricing_result = engine.price_account(account_df, include_combined_pml=opts.show_combined_pml)
                
                result = {
                    'AccountID': account_id,
                    'SourceFile': Path(account_path).name,
                }
                
                if opts.show_combined_pml:
                    # Add baseline, combined, and marginal columns
                    for rp in return_periods:
                        result[f'Baseline_{rp}'] = pricing_result['baseline'][rp]
                        result[f'Combined_{rp}'] = pricing_result['combined'][rp]
                        result[f'RI_{rp}'] = pricing_result['marginal'][rp]
                else:
                    # Only marginal columns
                    for rp, val in pricing_result.items():
                        result[f'RI_{rp}'] = val
                
                results.append(result)
                
            except Exception as e:
                warnings.warn(f"Error processing {account_id}: {e}")
                result = {
                    'AccountID': account_id,
                    'SourceFile': Path(account_path).name,
                    'Error': str(e)
                }
                for rp in return_periods:
                    result[f'RI_{rp}'] = np.nan
                    if opts.show_combined_pml:
                        result[f'Baseline_{rp}'] = np.nan
                        result[f'Combined_{rp}'] = np.nan
                results.append(result)
        
        pricing_time = time.perf_counter() - t0
        per_account_ms = (pricing_time / n_accounts) * 1000 if n_accounts > 0 else 0
        
        log(f"   Completed in {pricing_time:.2f}s ({per_account_ms:.1f}ms/account)")
        
        # === Generate Report ===
        results_df = pd.DataFrame(results)
        
        # Determine output path
        if opts.output:
            output_path = opts.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'results_{timestamp}.xlsx'
        
        log(f"\n📝 Generating report: {output_path}")
        
        report_gen = SummaryReportGenerator(
            roc=opts.roc,
            capital_rates=SummaryReportGenerator.DEFAULT_CAPITAL_RATES
        )
        
        portfolio_name = Path(opts.portfolio).stem
        report_path = report_gen.generate_report(
            results_df,
            portfolio_name=portfolio_name,
            baseline_pmls=engine.baseline_pmls,
            output_path=output_path,
            pricing_rp=opts.pricing_rp
        )
        
        # === Summary ===
        log(f"\n✅ Report generated: {report_path}")
        
        # Show quick summary
        if not quiet:
            log(f"\n📈 Quick Summary:")
            for rp in return_periods:
                col = f'RI_{rp}'
                if col in results_df.columns:
                    valid = results_df[col].dropna()
                    if len(valid) > 0:
                        log(f"   {rp}yr Marginal PML: "
                            f"min=${valid.min():,.0f}, "
                            f"mean=${valid.mean():,.0f}, "
                            f"max=${valid.max():,.0f}")
        
        log("\n" + "=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
