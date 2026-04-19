#!/usr/bin/env python
"""
Underwriter Workbench - Visual Marginal PML Pricing Interface

A professional UI for underwriters to analyze marginal PML impact
without touching the command line.

Features:
    1. Drag-and-Drop File Ingestion with column mapping preview
    2. Visual EP Curve comparison (Before vs After)
    3. Traffic Light System for risk appetite thresholds

Run with: streamlit run workbench.py
"""

# --- Relaunch guard: if run with 'python workbench.py', relaunch via streamlit ---
import sys as _sys

def _check_streamlit():
    try:
        from streamlit import runtime
        if runtime.exists():
            return
    except (ImportError, AttributeError):
        pass
    import subprocess
    print("Launching Streamlit server...")
    subprocess.run([_sys.executable, "-m", "streamlit", "run", __file__])
    _sys.exit(0)

_check_streamlit()
# --- End relaunch guard ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Import the core engine
from marginal_pml_kernel import (
    create_pricing_engine,
    MarginalPMLEngine,
    PMLCalculator,
    _portfolio_to_events_df,
    PRICING_MODE_SUBTRACT,
    PRICING_MODE_ADD
)
from pml_tool import ColumnMapper, FileIngestor


# =============================================================================
# CONFIG FILE SUPPORT
# =============================================================================

CONFIG_PATH = Path(__file__).parent.parent / 'pml_config.yaml'


@st.cache_data
def load_yaml_config() -> Dict:
    """Load pml_config.yaml once (cached)."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_return_periods_from_config(cfg: Dict) -> list:
    """Parse return_periods from config (handles '50, 100, 250' or [50,100,250])."""
    rps = cfg.get('return_periods', '50,100,250')
    if isinstance(rps, list):
        return [int(r) for r in rps]
    return [int(r.strip()) for r in str(rps).split(',')]


def load_file_from_path(file_path: str) -> Optional[pd.DataFrame]:
    """Load a CSV/Excel file from a filesystem path."""
    p = Path(file_path)
    if not p.exists():
        return None
    ext = p.suffix.lower()
    if ext == '.csv':
        try:
            return pd.read_csv(p)
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding='latin-1')
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(p)
    return None

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Marginal PML Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .traffic-light-green {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .traffic-light-yellow {
        background-color: #F59E0B;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .traffic-light-red {
        background-color: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'portfolio_df': None,
        'portfolio_mapped': None,
        'account_df': None,
        'account_mapped': None,
        'engine': None,
        'pricing_results': None,
        'ep_curve_data': None,
        'config_loaded': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# Load config once
cfg = load_yaml_config()
cfg_return_periods = parse_return_periods_from_config(cfg)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_file(uploaded_file) -> pd.DataFrame:
    """Load uploaded file into DataFrame."""
    if uploaded_file is None:
        return None
    
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    try:
        if file_ext == '.csv':
            # Try different encodings
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def get_column_mappings(df: pd.DataFrame) -> Dict[str, str]:
    """Get column mappings using the ColumnMapper."""
    mapper = ColumnMapper()
    return mapper.detect_columns(df)


def apply_column_mappings(df: pd.DataFrame, mappings: Dict[str, str]) -> pd.DataFrame:
    """Apply column mappings to DataFrame."""
    if df is None:
        return None
    
    result = df.copy()
    rename_map = {orig: std for orig, std in mappings.items() if orig in result.columns}
    result = result.rename(columns=rename_map)
    
    return result


def compute_ep_curve(calc: PMLCalculator, pml_range: np.ndarray) -> np.ndarray:
    """Compute exceedance probabilities for a range of PML values."""
    rps = []
    for pml in pml_range:
        rp = calc.implied_return_period(pml)
        rps.append(min(rp, 10000))  # Cap at 10,000 years for display
    return np.array(rps)


def get_traffic_light(pct_change: float, yellow_threshold: float, red_threshold: float) -> Tuple[str, str]:
    """Determine traffic light status based on percentage change."""
    if pct_change < 0:
        return "green", "🟢 DIVERSIFYING"
    elif pct_change < yellow_threshold:
        return "green", "🟢 ACCEPTABLE"
    elif pct_change < red_threshold:
        return "yellow", "🟡 REVIEW"
    else:
        return "red", "🔴 EXCEEDS APPETITE"


# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    if cfg:
        st.caption(f"Defaults loaded from {CONFIG_PATH.name}")
    
    # Pricing Mode
    cfg_mode = cfg.get('mode', 'subtract')
    pricing_mode = st.selectbox(
        "Pricing Mode",
        options=['subtract', 'add'],
        index=0 if cfg_mode == 'subtract' else 1,
        format_func=lambda x: "Subtract (RI Pricing)" if x == 'subtract' else "Add (New Business)",
        help="Subtract: PML(Portfolio) - PML(Portfolio-Account)\nAdd: PML(Portfolio+Account) - PML(Portfolio)"
    )
    
    # Return Periods
    st.markdown("### 📅 Return Periods")
    rp_50 = st.checkbox("50-year", value=50 in cfg_return_periods)
    rp_100 = st.checkbox("100-year", value=100 in cfg_return_periods)
    rp_250 = st.checkbox("250-year", value=250 in cfg_return_periods)
    rp_500 = st.checkbox("500-year", value=500 in cfg_return_periods)
    
    return_periods = tuple([rp for rp, enabled in [(50, rp_50), (100, rp_100), (250, rp_250), (500, rp_500)] if enabled])
    if not return_periods:
        return_periods = (100,)  # Default
    
    st.markdown("---")
    
    # Traffic Light Thresholds
    st.markdown("### 🚦 Risk Appetite Thresholds")
    st.caption("Set thresholds for % increase in portfolio PML")
    
    yellow_threshold = st.slider(
        "Yellow Threshold (%)",
        min_value=0.5,
        max_value=10.0,
        value=float(cfg.get('yellow_threshold', 2.0)),
        step=0.5,
        help="Flag as Yellow if PML increase exceeds this %"
    )
    
    red_threshold = st.slider(
        "Red Threshold (%)",
        min_value=1.0,
        max_value=20.0,
        value=float(cfg.get('red_threshold', 5.0)),
        step=0.5,
        help="Flag as Red if PML increase exceeds this %"
    )
    
    st.markdown("---")
    
    # Pricing Parameters
    st.markdown("### 💰 Pricing Parameters")
    
    cfg_pricing_rp = int(cfg.get('pricing_rp', 100))
    available_rps = [rp for rp in [50, 100, 250, 500] if rp in return_periods]
    pricing_rp_idx = available_rps.index(cfg_pricing_rp) if cfg_pricing_rp in available_rps else min(1, len(available_rps)-1)
    
    pricing_rp = st.selectbox(
        "Pricing RP",
        options=available_rps,
        index=pricing_rp_idx,
        help="Return period to use for capital calculation"
    )
    
    cfg_roc = float(cfg.get('roc', 0.15)) * 100
    roc_target = st.slider(
        "ROC Target (%)",
        min_value=5.0,
        max_value=30.0,
        value=cfg_roc,
        step=1.0
    ) / 100
    
    cfg_cap_rate = float(cfg.get('capital_rate', cfg.get(f'capital_rate_{cfg_pricing_rp}', 0.25))) * 100
    capital_rate = st.slider(
        "Capital Rate (%)",
        min_value=10.0,
        max_value=50.0,
        value=cfg_cap_rate,
        step=5.0
    ) / 100


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<p class="main-header">📊 Marginal PML Workbench</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze marginal portfolio impact with visual insights</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📁 Data Ingestion", "📈 Impact Analysis", "📋 Pricing Summary"])


# =============================================================================
# TAB 1: DATA INGESTION
# =============================================================================

with tab1:
    # --- Auto-load from config on first run ---
    if not st.session_state.config_loaded and cfg:
        cfg_portfolio = cfg.get('portfolio', '')
        cfg_quote = cfg.get('quote', '')
        
        if cfg_portfolio and st.session_state.portfolio_df is None:
            df = load_file_from_path(cfg_portfolio)
            if df is not None:
                st.session_state.portfolio_df = df
                mappings = get_column_mappings(df)
                if mappings:
                    required = {'EVENTID', 'RATE', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI'}
                    if required <= set(mappings.values()):
                        st.session_state.portfolio_mapped = apply_column_mappings(df, mappings)
        
        if cfg_quote and st.session_state.account_df is None:
            df = load_file_from_path(cfg_quote)
            if df is not None:
                st.session_state.account_df = df
                mappings = get_column_mappings(df)
                if mappings:
                    required = {'EVENTID', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI'}
                    if required <= set(mappings.values()):
                        st.session_state.account_mapped = apply_column_mappings(df, mappings)
        
        st.session_state.config_loaded = True
    
    st.markdown("### Upload Your Data")
    st.markdown("Drag and drop ELT files below, or edit **pml_config.yaml** to pre-load files automatically.")
    
    col1, col2 = st.columns(2)
    
    # --- Helper to show mapping UI for a loaded DataFrame ---
    def _show_mapping_panel(df, label, required_cols):
        """Display column mapping review and return mapped df or None."""
        st.markdown(f"##### Column Mapping")
        mappings = get_column_mappings(df)
        if not mappings:
            st.error("❌ Could not detect column mappings")
            return None
        
        st.success(f"✅ Found {len(mappings)} mappable columns")
        with st.expander("Review Column Mappings", expanded=True):
            for orig, std in mappings.items():
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.text(f"{orig} → {std}")
                with col_b:
                    if std in required_cols:
                        st.markdown("✓ Required")
        
        mapped_std = set(mappings.values())
        missing = required_cols - mapped_std
        if missing:
            st.warning(f"⚠️ Missing required columns: {', '.join(missing)}")
            return None
        
        mapped_df = apply_column_mappings(df, mappings)
        st.info(f"📊 {label}: {len(df):,} events ready")
        return mapped_df
    
    # Portfolio Upload
    with col1:
        st.markdown("#### 📁 Portfolio (Baseline)")
        
        # Show config-loaded info
        cfg_portfolio_path = cfg.get('portfolio', '')
        if st.session_state.portfolio_df is not None and cfg_portfolio_path:
            st.caption(f"Loaded from config: {cfg_portfolio_path}")
        
        portfolio_file = st.file_uploader(
            "Drop portfolio ELT here (or loaded from config)",
            type=['csv', 'xlsx', 'xls'],
            key='portfolio_upload',
            help="Upload your baseline portfolio Event Loss Table"
        )
        
        # Uploaded file takes priority over config
        if portfolio_file is not None:
            df = load_file(portfolio_file)
            if df is not None:
                st.session_state.portfolio_df = df
                mapped = _show_mapping_panel(df, "Portfolio", {'EVENTID', 'RATE', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI'})
                if mapped is not None:
                    st.session_state.portfolio_mapped = mapped
        elif st.session_state.portfolio_df is not None:
            # Show mapping for config-loaded file
            mapped = _show_mapping_panel(st.session_state.portfolio_df, "Portfolio", {'EVENTID', 'RATE', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI'})
            if mapped is not None:
                st.session_state.portfolio_mapped = mapped
    
    # Account Upload
    with col2:
        st.markdown("#### 📁 Account (Quote)")
        
        cfg_quote_path = cfg.get('quote', '')
        if st.session_state.account_df is not None and cfg_quote_path:
            st.caption(f"Loaded from config: {cfg_quote_path}")
        
        account_file = st.file_uploader(
            "Drop account ELT here (or loaded from config)",
            type=['csv', 'xlsx', 'xls'],
            key='account_upload',
            help="Upload the account you want to price"
        )
        
        if account_file is not None:
            df = load_file(account_file)
            if df is not None:
                st.session_state.account_df = df
                mapped = _show_mapping_panel(df, "Account", {'EVENTID', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI'})
                if mapped is not None:
                    st.session_state.account_mapped = mapped
        elif st.session_state.account_df is not None:
            mapped = _show_mapping_panel(st.session_state.account_df, "Account", {'EVENTID', 'PERSPVALUE', 'EXPVALUE', 'STDDEVC', 'STDDEVI'})
            if mapped is not None:
                st.session_state.account_mapped = mapped
    
    st.markdown("---")
    
    # Initialize Engine Button
    if st.session_state.portfolio_mapped is not None and st.session_state.account_mapped is not None:
        if st.button("🚀 Initialize Engine & Price Account", type="primary", use_container_width=True):
            with st.spinner("Initializing pricing engine..."):
                try:
                    # Create engine
                    engine = create_pricing_engine(
                        st.session_state.portfolio_mapped,
                        return_periods=return_periods,
                        use_poisson=True,
                        mode=pricing_mode
                    )
                    st.session_state.engine = engine
                    
                    # Price account
                    results = engine.price_account(
                        st.session_state.account_mapped,
                        include_combined_pml=True
                    )
                    st.session_state.pricing_results = results
                    
                    # Compute EP curves for visualization
                    # Get baseline and combined calculators
                    portfolio_events = _portfolio_to_events_df(st.session_state.portfolio_mapped)
                    baseline_calc = PMLCalculator(portfolio_events, use_poisson=True)
                    
                    # Get combined calculator
                    if pricing_mode == 'subtract':
                        combined_df = engine._fast_subtract_account(st.session_state.account_mapped)
                    else:
                        combined_df = engine._fast_add_account(st.session_state.account_mapped)
                    combined_calc = PMLCalculator(combined_df, use_poisson=True)
                    
                    # Compute curves
                    max_exp = baseline_calc.max_exposure
                    pml_range = np.linspace(max_exp * 0.01, max_exp * 0.99, 100)
                    
                    baseline_rps = compute_ep_curve(baseline_calc, pml_range)
                    combined_rps = compute_ep_curve(combined_calc, pml_range)
                    
                    st.session_state.ep_curve_data = {
                        'pml_range': pml_range,
                        'baseline_rps': baseline_rps,
                        'combined_rps': combined_rps
                    }
                    
                    st.success("✅ Engine initialized and account priced!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.info("👆 Upload both portfolio and account files to proceed")


# =============================================================================
# TAB 2: IMPACT ANALYSIS
# =============================================================================

with tab2:
    if st.session_state.pricing_results is None:
        st.info("👈 Go to Data Ingestion tab to upload files and initialize the engine")
    else:
        results = st.session_state.pricing_results
        engine = st.session_state.engine
        
        st.markdown("### Visual Impact Analysis")
        
        # Traffic Light Summary
        st.markdown("#### 🚦 Risk Appetite Check")
        
        cols = st.columns(len(return_periods))
        for i, rp in enumerate(return_periods):
            baseline = results['baseline'][rp]
            marginal = results['marginal'][rp]
            pct_change = (marginal / baseline) * 100 if baseline > 0 else 0
            
            light_class, light_text = get_traffic_light(pct_change, yellow_threshold, red_threshold)
            
            with cols[i]:
                st.metric(
                    label=f"{rp}-Year",
                    value=f"${marginal:,.0f}",
                    delta=f"{pct_change:+.2f}%",
                    delta_color="inverse" if marginal > 0 else "normal"
                )
                st.markdown(f"<p style='text-align:center'><span class='traffic-light-{light_class}'>{light_text}</span></p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # EP Curve Visualization
        st.markdown("#### 📈 Exceedance Probability Curves")
        
        if st.session_state.ep_curve_data is not None:
            ep_data = st.session_state.ep_curve_data
            
            # Create the EP curve plot
            fig = go.Figure()
            
            # Baseline curve
            fig.add_trace(go.Scatter(
                x=ep_data['pml_range'] / 1e6,
                y=1 / ep_data['baseline_rps'],
                mode='lines',
                name='Portfolio (Baseline)',
                line=dict(color='#3B82F6', width=3),
                hovertemplate='PML: $%{x:.1f}M<br>AEP: %{y:.4f}<extra></extra>'
            ))
            
            # Combined curve
            combined_label = "Portfolio - Account" if pricing_mode == 'subtract' else "Portfolio + Account"
            fig.add_trace(go.Scatter(
                x=ep_data['pml_range'] / 1e6,
                y=1 / ep_data['combined_rps'],
                mode='lines',
                name=combined_label,
                line=dict(color='#EF4444', width=3),
                hovertemplate='PML: $%{x:.1f}M<br>AEP: %{y:.4f}<extra></extra>'
            ))
            
            # Add vertical lines for key return periods
            for rp in return_periods:
                baseline_pml = results['baseline'][rp]
                combined_pml = results['combined'][rp]
                
                fig.add_vline(
                    x=baseline_pml / 1e6,
                    line_dash="dash",
                    line_color="#3B82F6",
                    opacity=0.5,
                    annotation_text=f"{rp}yr: ${baseline_pml/1e6:.1f}M"
                )
            
            fig.update_layout(
                title=dict(
                    text="Aggregate Exceedance Probability Curve",
                    font=dict(size=20)
                ),
                xaxis_title="PML ($ Millions)",
                yaxis_title="Annual Exceedance Probability",
                yaxis_type="log",
                yaxis=dict(
                    range=[-4, 0],  # 0.0001 to 1
                    tickformat=".4f"
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Marginal Impact Bar Chart
            st.markdown("#### 📊 Marginal Impact by Return Period")
            
            fig2 = go.Figure()
            
            colors = ['#10B981' if results['marginal'][rp] < 0 else '#EF4444' for rp in return_periods]
            
            fig2.add_trace(go.Bar(
                x=[f"{rp}yr" for rp in return_periods],
                y=[results['marginal'][rp] for rp in return_periods],
                marker_color=colors,
                text=[f"${results['marginal'][rp]:,.0f}" for rp in return_periods],
                textposition='outside',
                hovertemplate='%{x}: $%{y:,.0f}<extra></extra>'
            ))
            
            fig2.update_layout(
                title="Marginal PML Impact",
                xaxis_title="Return Period",
                yaxis_title="Marginal PML ($)",
                height=400,
                showlegend=False
            )
            
            # Add zero line
            fig2.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Interpretation
            st.markdown("#### 💡 Interpretation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🟢 Green (Negative) Bars:**
                - Account **diversifies** the portfolio
                - Removing it would **increase** portfolio PML
                - Consider offering competitive pricing
                """)
            
            with col2:
                st.markdown("""
                **🔴 Red (Positive) Bars:**
                - Account **adds** tail risk to portfolio
                - Removing it would **decrease** portfolio PML
                - Price accordingly with capital charge
                """)


# =============================================================================
# TAB 3: PRICING SUMMARY
# =============================================================================

with tab3:
    if st.session_state.pricing_results is None:
        st.info("👈 Go to Data Ingestion tab to upload files and initialize the engine")
    else:
        results = st.session_state.pricing_results
        engine = st.session_state.engine
        
        st.markdown("### 💰 Pricing Recommendation")
        
        # Get account AAL
        account_aal = st.session_state.account_mapped['PERSPVALUE'].sum()
        
        # Calculate pricing
        marginal_pml = results['marginal'][pricing_rp]
        marginal_capital = max(0, marginal_pml * capital_rate)
        capital_charge = marginal_capital * roc_target
        technical_price = account_aal + capital_charge
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Account AAL", f"${account_aal:,.0f}")
        with col2:
            st.metric(f"Marginal PML ({pricing_rp}yr)", f"${marginal_pml:,.0f}")
        with col3:
            st.metric("Marginal Capital", f"${marginal_capital:,.0f}")
        with col4:
            st.metric("Technical Price", f"${technical_price:,.0f}", delta=f"+${capital_charge:,.0f} cap charge")
        
        st.markdown("---")
        
        # Detailed breakdown
        st.markdown("#### 📋 Pricing Breakdown")
        
        breakdown_data = {
            'Component': [
                'Account AAL (Expected Loss)',
                f'Marginal PML ({pricing_rp}yr)',
                f'Capital Rate ({capital_rate:.0%})',
                'Marginal Capital',
                f'ROC Target ({roc_target:.0%})',
                'Capital Charge',
                '**Technical Price**'
            ],
            'Value': [
                f'${account_aal:,.0f}',
                f'${marginal_pml:,.0f}',
                f'{capital_rate:.0%}',
                f'${marginal_capital:,.0f}',
                f'{roc_target:.0%}',
                f'${capital_charge:,.0f}',
                f'**${technical_price:,.0f}**'
            ],
            'Formula': [
                'Σ Mean Loss',
                'PML(Portfolio) - PML(Portfolio±Account)' if pricing_mode == 'subtract' else 'PML(Portfolio±Account) - PML(Portfolio)',
                'Regulatory/Economic capital requirement',
                'Marginal PML × Capital Rate',
                'Target return on capital held',
                'Marginal Capital × ROC',
                '**AAL + Capital Charge**'
            ]
        }
        
        st.table(pd.DataFrame(breakdown_data))
        
        st.markdown("---")
        
        # Full results table
        st.markdown("#### 📊 Full Results by Return Period")
        
        full_results = []
        for rp in return_periods:
            baseline = results['baseline'][rp]
            combined = results['combined'][rp]
            marginal = results['marginal'][rp]
            pct_change = (marginal / baseline) * 100 if baseline > 0 else 0
            
            light_class, light_text = get_traffic_light(pct_change, yellow_threshold, red_threshold)
            
            full_results.append({
                'Return Period': f'{rp}-year',
                'Portfolio PML': f'${baseline:,.0f}',
                'Portfolio±Account PML': f'${combined:,.0f}',
                'Marginal PML': f'${marginal:,.0f}',
                '% Change': f'{pct_change:+.2f}%',
                'Status': light_text
            })
        
        st.dataframe(pd.DataFrame(full_results), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Export button
        st.markdown("#### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create export DataFrame
            export_data = {
                'Metric': [
                    'Account AAL',
                    'Pricing Return Period',
                    'Marginal PML',
                    'Capital Rate',
                    'Marginal Capital',
                    'ROC Target',
                    'Capital Charge',
                    'Technical Price'
                ],
                'Value': [
                    account_aal,
                    pricing_rp,
                    marginal_pml,
                    capital_rate,
                    marginal_capital,
                    roc_target,
                    capital_charge,
                    technical_price
                ]
            }
            
            export_df = pd.DataFrame(export_data)
            
            # Download as CSV
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name="pricing_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download as Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Pricing', index=False)
                pd.DataFrame(full_results).to_excel(writer, sheet_name='Return Periods', index=False)
            
            st.download_button(
                label="📊 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name="pricing_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Marginal PML Workbench v1.0 | "
    "Built with Streamlit | Method of Moments + Poisson AEP</p>",
    unsafe_allow_html=True
)

