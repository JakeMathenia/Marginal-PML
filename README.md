# Coastal Property Catastrophe Loss Model
**Portfolio Project — Jake Mathenia**

An end-to-end catastrophe loss model built on FEMA NFIP claims data, combining XGBoost frequency/severity models with Monte Carlo simulation to estimate portfolio Probable Maximum Loss (PML) at multiple return periods — including **marginal PML** analysis for individual account underwriting decisions.

Deployed as an interactive Streamlit dashboard.

---

## What This Does

Insurance companies and reinsurers manage risk by estimating the maximum loss a portfolio could experience at various return periods (e.g., a 1-in-100-year event). This project replicates that workflow end-to-end:

1. **Frequency Model** — predicts the probability a given property generates a claim in a policy year
2. **Severity Model** — predicts the expected dollar loss if a claim occurs (XGBoost, log-transformed target)
3. **Monte Carlo Simulation** — combines frequency and severity into a full portfolio loss distribution (100,000 iterations)
4. **Marginal PML** — calculates the incremental change in portfolio risk from adding a single new account

---

## Key Outputs

| Metric | Description |
|---|---|
| Mean Annual Loss (MAL) | Expected annual loss across the portfolio |
| 1-in-10 PML | 90th percentile of simulated annual losses |
| 1-in-100 PML | 99th percentile |
| 1-in-250 PML | 99.6th percentile |
| Loss Exceedance Curve | Full distribution of losses — standard reinsurance output |
| Marginal PML | Delta PML at each return period when a new account is added |

---

## Project Structure

```
├── src/
│   ├── marginal_pml_kernel.py   ← Core math engine (Beta dist, Method of Moments, AEP)
│   ├── pml_tool.py              ← CLI wrapper for underwriter-facing workflows
│   └── workbench.py             ← Streamlit dashboard
├── scripts/
│   ├── generate_sample_data.py  ← Generate synthetic ELT portfolio + accounts
│   └── generate_sample_yelt.py  ← Generate correlated YELT simulation data
├── data/
│   ├── portfolio.csv            ← Synthetic base portfolio
│   ├── accounts/                ← Individual account ELTs for batch pricing
│   └── yelt/                    ← Year Event Loss Tables by peril
├── pml_config.yaml              ← Configuration (return periods, column mappings, etc.)
└── PROJECT_SCOPE.md             ← Full technical scope and model architecture
```

---

## Marginal PML Engine

The core calculation in `marginal_pml_kernel.py` uses **Method of Moments** to fit a Beta distribution to each event's loss characteristics, then applies a **Poisson frequency** assumption to compute Annual Exceedance Probability (AEP) curves:

- Means are additive across events
- Correlated SD (`SD_C`) is additive
- Independent SD (`SD_I`) is root-sum-of-squares
- PML reported at configurable return periods (10, 50, 100, 250 year)

```python
from marginal_pml_kernel import calculate_marginal_pml

result = calculate_marginal_pml(
    portfolio_elt=portfolio_df,
    account_elt=new_account_df,
    return_periods=[10, 50, 100, 250]
)
print(result.marginal_pml)
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run src/workbench.py

# CLI batch pricing
python src/pml_tool.py --portfolio data/portfolio.csv \
                       --quote-folder data/accounts/ \
                       --batch
```

---

## Data Sources

| Source | Usage |
|---|---|
| [OpenFEMA NFIP Claims](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) | Historical flood losses by location and property type |
| [OpenFEMA NFIP Policies](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2) | Policy-in-force data for loss ratio calculation |
| [NOAA Storm Events](https://www.ncdc.noaa.gov/stormevents/ftp.jsp) | Hurricane/storm event data joined to claims |
| [FEMA Flood Zone Maps API](https://msc.fema.gov/arcgis/rest/services) | Flood zone classification per property |

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `SciPy` · `NumPy` · `pandas` · `Streamlit` · `Plotly` · `SHAP`

---

## Context

This project mirrors the marginal PML pricing workflows used by E&S carriers and reinsurers to make individual account bind/decline decisions. It produces industry-recognized outputs (loss exceedance curves, return-period PML tables) and demonstrates the full ML stack from data ingestion through model deployment.
