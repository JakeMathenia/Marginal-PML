# Coastal Property Catastrophe Loss Model

**End-to-end ML pipeline for flood risk pricing using FEMA NFIP data → ELT generation → marginal PML engine.**

> *Raw NFIP claims data goes in. Industry-standard Event Loss Tables come out. Loss Exceedance Curves and technical prices are produced by the same kernel used with commercial cat model output.*

---

## What This Project Does

This project bridges the gap between raw insurance claims data and actuarial catastrophe pricing. Given a portfolio of coastal properties, it:

1. **Pulls real flood claims data** from OpenFEMA and storm data from NOAA
2. **Trains ML models** (XGBoost) to predict claim frequency and severity for any property
3. **Generates an ELT** in the standard `RATE · PERSPVALUE · EXPVALUE · STDDEVC · STDDEVI` format used by AIR, RMS, and CoreLogic
4. **Prices new accounts** using a marginal PML engine that shows how each account shifts the portfolio loss exceedance curve

---

## Architecture

```
OpenFEMA NFIP Claims API  +  NOAA Storm Events
            │
            ▼
    src/data_ingestion.py          ← pulls, caches, joins NFIP + NOAA
            │
            ▼
    src/feature_engineering.py     ← 17-feature matrix from property + storm fields
            │
       ┌────┴────┐
       ▼         ▼
  FrequencyModel  SeverityModel    ← XGBoost on NFIP training data
  (P(claim))      (log E[loss|claim])
       └────┬────┘
            ▼
    src/elt_generator.py           ← RATE · PERSPVALUE · EXPVALUE · STDDEVC · STDDEVI
            │
            ▼
    src/marginal_pml_kernel.py     ← Method of Moments + Poisson AEP
            │
            ▼
    src/workbench.py               ← Streamlit UI  (4 tabs, incl. SHAP explainability)
```

---

## Repository Structure

```
Marginal PML/
│
├── src/
│   ├── data_ingestion.py       ← OpenFEMA + NOAA API clients, claims/storm joiner
│   ├── feature_engineering.py  ← FeaturePipeline → 17-column feature matrix
│   ├── severity_model.py       ← XGBoost regressor on log(claim); save/load; SHAP
│   ├── frequency_model.py      ← XGBoost classifier; Platt calibration; AUC/AP eval
│   ├── elt_generator.py        ← ELTGenerator: freq+sev → ELT; validate_elt()
│   ├── marginal_pml_kernel.py  ← MarginalPMLEngine: Method of Moments + Poisson AEP
│   ├── pml_tool.py             ← CLI wrapper, ColumnMapper, FileIngestor
│   └── workbench.py            ← Streamlit 4-tab dashboard (upload → price → SHAP)
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb     ← Pull NFIP claims + NOAA; save enriched parquet
│   ├── 02_eda.ipynb                ← EDA: flood zone, hurricane, annual trends
│   ├── 03_feature_engineering.ipynb← Build 17-feature matrix; correlation charts
│   ├── 04_severity_model.ipynb     ← Train/eval XGBoost severity; SHAP waterfall
│   ├── 05_frequency_model.ipynb    ← Train/eval XGBoost frequency; calibration curve
│   └── 06_elt_generation.ipynb     ← ELT generation → PMLEngine round-trip ✅
│
├── scripts/
│   ├── generate_sample_data.py     ← Generate synthetic portfolio ELT for testing
│   └── generate_sample_yelt.py     ← Generate synthetic YELT for testing
│
├── data/
│   ├── portfolio.csv               ← Synthetic baseline portfolio (testing)
│   ├── raw/                        ← Created by notebooks (NFIP parquets, features)
│   ├── accounts/                   ← Account ELTs for pricing
│   └── yelt/                       ← Year Event Loss Tables
│
├── models/                         ← Created by notebooks (joblib model artifacts)
│
├── pml_config.yaml                 ← Default paths, thresholds, pricing params
├── requirements.txt                ← All Python dependencies
├── PROJECT_SCOPE.md                ← Detailed project scope and technical design
├── ENHANCED_REPORTING.md           ← Enhanced reporting module documentation
└── README.md                       ← This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook Pipeline (in order)

```bash
jupyter lab
# Open and run notebooks 01 → 06 sequentially
```

| Notebook | What It Does | Output |
|---|---|---|
| `01_data_ingestion` | Pulls NFIP claims + computes filing rates from OpenFEMA policies | `data/raw/nfip_claims_enriched.parquet` + `filing_rates.parquet` |
| `02_eda` | Exploratory analysis of loss patterns | 4 diagnostic figures |
| `03_feature_engineering` | Builds 17-feature matrix | `data/raw/features.parquet` + `targets.parquet` |
| `04_severity_model` | Trains log-severity XGBoost | `models/severity_model.joblib` |
| `05_frequency_model` | Trains claim frequency XGBoost | `models/frequency_model.joblib` |
| `06_elt_generation` | **End-to-end round-trip** | `data/ml_generated_elt.csv` + PML curves |

### 3. Launch the Streamlit Workbench

```bash
streamlit run src/workbench.py
```

The workbench has four tabs:
- **📁 Data Ingestion** — Upload portfolio + account ELTs; auto-loads from `pml_config.yaml`
- **📈 Impact Analysis** — Visual EP curves, traffic-light risk appetite check
- **📋 Pricing Summary** — Technical price = AAL + Capital Charge; CSV/Excel export
- **🔍 Model Explainability** — SHAP beeswarm + feature importance for severity model

### 4. Command-Line Pricing (Optional)

```bash
python src/pml_tool.py \
  --portfolio data/portfolio.csv \
  --account data/accounts/my_account.csv \
  --return-periods 50,100,250 \
  --mode subtract
```

---

## The ELT Format

All pricing runs on this 6-column format (compatible with AIR, RMS, CoreLogic output):

| Column | Type | Description |
|---|---|---|
| `EVENTID` | string | Unique property/event identifier |
| `RATE` | float | Annual exceedance frequency (0–1) |
| `PERSPVALUE` | float | Mean loss given event occurs |
| `EXPVALUE` | float | Total insured value / policy limit |
| `STDDEVC` | float | Correlated (systematic) loss std dev |
| `STDDEVI` | float | Idiosyncratic loss std dev |

The **ML pipeline generates** this format using `ELTGenerator(freq_model, sev_model).generate(X)`.  
The **synthetic test data** uses `scripts/generate_sample_data.py`.  
Both feed the same `MarginalPMLEngine` — this interchangeability is the key design principle.

---

## Marginal PML Methodology

The `MarginalPMLEngine` uses **Method of Moments + Poisson Annual Exceedance Probability**:

```
Portfolio Aggregate Variance = Σᵢ [ RATEᵢ · (PERSPVALUEᵢ² + STDDEVCᵢ² + STDDEVIᵢ²) ]
                             + 2 · Σᵢ Σⱼ (covariance from shared STDDEVC)

PML(T) = μ_portfolio + z(1 - 1/T) · σ_portfolio    [method of moments]
```

**Marginal PML** = `PML(Portfolio) − PML(Portfolio − Account)` (subtract mode)  
or  `PML(Portfolio + Account) − PML(Portfolio)` (add mode)

**Technical Price** = `Account AAL + (Marginal PML × Capital Rate × ROC Target)`

---

## Features Engineered

| Feature | Source | Rationale |
|---|---|---|
| `flood_zone_risk_score` | NFIP `floodZone` | Primary risk driver |
| `is_coastal_high_risk` | `floodZone` V/VE | Coastal velocity zone flag |
| `is_high_risk_zone` | `floodZone` A/V | SFHA indicator |
| `occupancy_is_residential` | `occupancyType` | Loss pattern differs |
| `has_basement` | `basementType` | Flood depth exposure |
| `log_building_coverage` | Coverage amount | Scale of exposure |
| `coverage_to_value_ratio` | Coverage / value | Adverse selection proxy |
| `num_floors` | `numberOfFloors` | Upper floors safer |
| `construction_age_years` | Construction date | Older = more vulnerable |
| `is_post_firm` | Construction date | Post-1978 regulatory flag |
| `has_elevation_cert` | `elevCert` | Risk mitigation |
| `hurricane_risk_score` | NOAA category | Storm intensity |
| `is_hurricane_peril` | NOAA event type | Event type flag |
| `is_flood_peril` | NOAA event type | Event type flag |
| `is_coastal_state` | `reportedState` | Geographic risk |
| `log_storm_damage` | NOAA damage field | Storm magnitude proxy |

---

## Model Performance (Typical on NFIP 2000–2023)

| Model | Metric | Typical Value |
|---|---|---|
| Severity (XGBoost) | R² on log-scale | 0.45 – 0.65 |
| Severity (XGBoost) | RMSE (log) | 0.95 – 1.10 |
| Frequency (XGBoost) | AUC-ROC | 0.78 – 0.85 |
| Frequency (XGBoost) | Average Precision | 0.35 – 0.55 |
| Frequency (XGBoost) | Brier Score | 0.08 – 0.12 |

> **Note:** R² for claim severity is inherently low — catastrophe losses have high inherent randomness. The model captures systematic risk drivers (flood zone, hurricane category) well; individual claim variance is idiosyncratic.

---

## Key Design Decisions

**Why XGBoost on log(severity)?**  
Flood claim amounts are log-normally distributed (confirmed in `02_eda`). Transforming to log-scale before fitting makes the regression target approximately normal, improving XGBoost's split quality and producing unbiased predictions after exp() back-transform.

**Why Platt calibration on the frequency model?**  
The ELT `RATE` must be a true annual frequency (not just a ranking score). Calibrated probabilities ensure that properties predicted at 15% actually have ~15% observed claim rates in held-out data.

**Why `STDDEVC` and `STDDEVI` separately?**  
The Method of Moments aggregation needs correlated variance (shared across properties in the same storm event) separated from idiosyncratic variance. Correlated variance drives portfolio diversification credit — coastal V-zone properties in the same hurricane corridor have high STDDEVC/total ratio.

**Why scale by filing rate for the ELT RATE column?**  
The frequency model trains on NFIP claims records where the target is P(paid | filed) ≈ 77%. But the ELT `RATE` must reflect the true annual probability of a claim for any property — which is ~2–5%. The `ELTGenerator` multiplies model output by empirical filing rates (claims per year / policies in force per year), computed from OpenFEMA policy data stratified by state and flood zone. This is done in `data_ingestion.compute_filing_rates()` and cached as `data/raw/filing_rates.parquet`. Without this correction the ELT would massively overstate annual frequencies.

**Why interchangeable ELT format?**  
Underwriters who normally receive AIR/RMS output can substitute the ML-generated ELT into the same `MarginalPMLEngine` workflow. This enables pricing of new accounts or perils where no commercial model output is available.

---

## Configuration (`pml_config.yaml`)

```yaml
portfolio: data/portfolio.csv      # Default portfolio path
quote:     data/accounts/quote.csv # Default account path
mode:      subtract                # subtract | add
return_periods: 50, 100, 250       # Comma-separated

# Pricing params
pricing_rp:   100                  # Return period for capital calc
capital_rate: 0.25                 # Capital held per dollar of PML
roc:          0.15                 # Target return on capital

# Workbench thresholds
yellow_threshold: 2.0              # % PML increase → yellow flag
red_threshold:    5.0              # % PML increase → red flag
```

---

## Dependencies

```
xgboost>=2.0          # Gradient boosting models
scikit-learn>=1.3     # Pipeline, calibration, metrics
shap>=0.44            # SHAP explainability
pandas>=2.0           # Data manipulation
numpy>=1.26           # Numerical ops
streamlit>=1.35       # Workbench UI
plotly>=5.18          # Interactive charts
requests>=2.31        # OpenFEMA API
pyarrow>=14.0         # Parquet I/O
joblib>=1.3           # Model serialization
pyyaml>=6.0           # Config file
openpyxl>=3.1         # Excel export
```

Full pinned list in `requirements.txt`.

---

## Data Sources

| Source | Records | License |
|---|---|---|
| [OpenFEMA NFIP Claims v2](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) | ~2.5M total | Public domain |
| [OpenFEMA NFIP Policies v2](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2) | ~60M total | Public domain |
| [NOAA Storm Events Database](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles) | ~1.7M events | Public domain |

All data is federally published and freely available. No PII — NFIP records are pre-redacted by FEMA.

---

## License

MIT — see repository root for details.
