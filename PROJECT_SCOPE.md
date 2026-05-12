# Coastal Property Catastrophe Loss Model
### Portfolio Project — Jake Mathenia

---

## Problem Statement

Given a portfolio of coastal properties, estimate probable maximum loss (PML) at various return periods (1-in-100, 1-in-250 year events) using historical FEMA flood claim data and property characteristics.

This mirrors real actuarial and underwriting workflows used by E&S carriers and reinsurers to assess aggregate exposure and make capital allocation decisions.

---

## Project Goals

- Build a frequency + severity model to estimate individual property loss potential from FEMA NFIP claims data
- Generate **Event Loss Tables (ELTs)** from ML model predictions — bridging the ML pipeline to industry-standard cat model input formats
- Simulate portfolio-level loss distributions using Monte Carlo methods
- Report industry-standard PML metrics at multiple return periods
- Calculate **marginal PML** — the change in portfolio risk from adding a single new account
- Deploy results as an interactive Streamlit dashboard with SHAP explainability

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   DATA SOURCES                       │
│  OpenFEMA Claims · NOAA Storms · FEMA Flood Zones   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                     │
│  Flood zone · Building type · Coast distance · etc.  │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│ FREQUENCY MODEL  │   │ SEVERITY MODEL   │
│ XGBoost/LogReg   │   │ XGBoost Regressor│
│ P(claim | X)     │   │ E[loss | claim]  │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────────┐
│            ELT GENERATOR                             │
│  Converts freq/sev predictions into ELT format:     │
│  EVENTID · RATE · PERSPVALUE · EXPVALUE · STDDEVC   │
└────────────────────────┬────────────────────────────┘
                         │  ← same format as cat model output (AIR/RMS/CoreLogic)
                         ▼
┌─────────────────────────────────────────────────────┐
│         MARGINAL PML ENGINE (CORE PRODUCT)          │
│  Method of Moments · Beta Distribution · Brent's    │
│  Correlated Simulator · YELT Engine (stochastic)    │
│  CLI Pricing Tool · Streamlit Workbench             │
└─────────────────────────────────────────────────────┘
```

The key design insight: the ELT format is a universal interface. Real cat model outputs (AIR, RMS, CoreLogic) arrive as ELTs, and our ML pipeline generates ELTs from raw property data. **The pricing engine consumes both interchangeably.**

---

## Data Sources

| Source | What to Pull | URL |
|---|---|---|
| OpenFEMA NFIP Claims | Historical flood losses by location, property type, flood zone | https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2 |
| OpenFEMA NFIP Policies | Policy in force data — enables loss ratio calculation | https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2 |
| NOAA Storm Events Database | Hurricane/storm event data to tie losses to events | https://www.ncdc.noaa.gov/stormevents/ftp.jsp |
| FEMA Flood Zone Maps API | Flood zone classification per property | https://msc.fema.gov/arcgis/rest/services |

---

## Model Architecture

### Layer 1 — Severity Model
Predict expected claim dollar amount given that a loss event occurs.

- **Model:** XGBoost Regressor
- **Target:** `amountPaidOnBuildingClaim` (log-transformed)
- **Features:**
  - Flood zone classification (AE, VE, X, etc.)
  - Building occupancy type (residential, commercial)
  - Foundation type (slab, crawlspace, elevated)
  - Total coverage amount
  - Number of floors
  - Year of construction
  - Hurricane category (from NOAA join)
  - Distance to coast
  - Census tract features

### Layer 2 — Frequency Model
Predict the probability a given policy generates a claim in a policy year.

- **Model:** XGBoost Classifier / Logistic Regression
- **Target:** Binary — claim occurred (1) or not (0)
- **Features:** Same as severity model
- **Output:** Annual claim probability per property

### Layer 3 — ELT Generator (ML → Cat Model Bridge)
Convert frequency/severity model predictions into Event Loss Tables.

- **RATE** ← frequency model output (annual claim probability)
- **PERSPVALUE** ← severity model mean prediction (expected loss)
- **EXPVALUE** ← policy limit / total insured value
- **STDDEVC** ← correlated variance component (driven by flood zone / storm event)
- **STDDEVI** ← idiosyncratic variance component (property-specific)

This ELT can be consumed directly by the production Marginal PML Engine — the same way a commercial cat model's output would be.

### Layer 4 — Marginal PML Engine (Production Pricing)
Industry-grade PML calculation using the Method of Moments on Beta distributions with Poisson frequency.

- **Analytical path:** Brent's root-finding on the AEP curve (fast, deterministic)
- **Correlated path:** Gaussian copula + Beta marginals via Monte Carlo
- **YELT path:** Exact empirical PML from full stochastic simulation output

**Reported metrics:**
- Mean Annual Loss (MAL)
- 1-in-50 year PML
- 1-in-100 year PML (99th percentile)
- 1-in-250 year PML (99.6th percentile)
- Loss Exceedance Curve

### Layer 5 — Marginal PML
Show how adding a new account shifts the aggregate portfolio PML.

- **Subtract mode:** `Marginal = PML(Portfolio) − PML(Portfolio − Account)` — used for RI pricing
- **Add mode:** `Marginal = PML(Portfolio + Account) − PML(Portfolio)` — used for new business
- **Output:** Delta PML at each return period + traffic light risk appetite signal

---

## Project Structure

```
flood-cat-model/
├── README.md                          ← Project overview, results, and key charts
├── PROJECT_SCOPE.md                   ← This file
├── ENHANCED_REPORTING.md              ← Enhanced PML reporting feature docs
├── requirements.txt                   ← Python dependencies
├── pml_config.yaml                    ← Configuration (return periods, pricing params)
├── .gitignore
│
├── data/
│   ├── portfolio.csv                  ← Synthetic base portfolio ELT
│   ├── accounts/                      ← Individual account ELTs for batch pricing
│   └── yelt/                          ← Year Event Loss Tables by peril
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        ← OpenFEMA API pull + NOAA join
│   ├── 02_eda.ipynb                   ← Exploratory analysis, loss distributions
│   ├── 03_feature_engineering.ipynb   ← Feature construction, encoding
│   ├── 04_severity_model.ipynb        ← XGBoost severity model + SHAP analysis
│   ├── 05_frequency_model.ipynb       ← Frequency classifier + calibration
│   └── 06_elt_generation.ipynb        ← ML predictions → ELT format → pricing engine
│
├── src/
│   ├── data_ingestion.py              ← OpenFEMA API client + NOAA event joiner
│   ├── feature_engineering.py         ← Reusable feature pipeline
│   ├── severity_model.py              ← XGBoost severity model wrapper
│   ├── frequency_model.py             ← XGBoost frequency model wrapper
│   ├── elt_generator.py               ← ML predictions → ELT converter
│   ├── marginal_pml_kernel.py         ← Core math engine (Beta dist, MoM, AEP, YELT)
│   ├── pml_tool.py                    ← CLI wrapper for underwriter-facing workflows
│   └── workbench.py                   ← Streamlit dashboard
│
└── scripts/
    ├── generate_sample_data.py        ← Generate synthetic ELT portfolio + accounts
    └── generate_sample_yelt.py        ← Generate correlated YELT simulation data
```

---

## Streamlit Dashboard Features

1. **Portfolio Upload** — Upload a CSV of properties or use a pre-built synthetic portfolio
2. **Run Simulation** — Trigger pricing engine with configurable return periods
3. **Loss Exceedance Curve** — Visual output showing PML at all return periods
4. **Marginal PML Tool** — Input a new property's characteristics; see its impact on portfolio PML
5. **SHAP Explainability Panel** — Which features drive the largest individual losses (severity model)
6. **Traffic Light System** — Green / Yellow / Red risk appetite signal per account

---

## Key Python Libraries

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
scipy
streamlit
plotly
requests         # OpenFEMA / FEMA API calls
PyYAML
openpyxl
joblib           # model serialization
faker            # synthetic data generation
```

---

## Resume Bullet (Final Framing)

> *"Built an end-to-end catastrophe loss model using FEMA NFIP data — combining XGBoost frequency/severity models with a custom ELT generator to produce industry-standard Event Loss Tables, then consuming those ELTs through a Method of Moments Marginal PML engine to estimate portfolio risk at multiple return periods and deployed as an interactive Streamlit application with SHAP explainability."*

---

## Development Phases

| Phase | Tasks | Status |
|---|---|---|
| **1 — Data** | Pull OpenFEMA data, clean, join NOAA events | 🔲 In Progress |
| **2 — EDA** | Explore loss distributions, feature correlations | 🔲 In Progress |
| **3 — Models** | Build + tune severity and frequency models | 🔲 In Progress |
| **4 — ELT Bridge** | ML predictions → ELT format → pricing engine | 🔲 In Progress |
| **5 — PML Engine** | Method of Moments engine + YELT path + correlated sim | ✅ Complete |
| **6 — Dashboard** | Streamlit workbench + loss exceedance curve + traffic lights | ✅ Complete |
| **7 — CLI Tool** | Batch pricing CLI + Excel reports + YAML config | ✅ Complete |
| **8 — Polish** | SHAP visuals, README, tests, cleanup for GitHub | 🔲 In Progress |

---

## Why This Project Works

- Uses **real, publicly available data** — no proprietary concerns
- Mirrors actuarial workflows common across E&S carriers and reinsurers (PML, marginal PML)
- Produces **industry-recognized outputs** (loss exceedance curve, return period PML, ELT format)
- **ELT format is a universal interface** — bridges the ML world and the cat modeling world
- Demonstrates the full stack: data ingestion → feature engineering → ML modeling → ELT generation → analytical pricing engine → deployment
- Rare in public portfolios — most ML projects don't touch cat modeling
- Speaks directly to hiring managers at reinsurers, E&S carriers, and cat modeling firms
