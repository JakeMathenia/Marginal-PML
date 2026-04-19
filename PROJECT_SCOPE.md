# Coastal Property Catastrophe Loss Model
### Portfolio Project — Jake Mathenia

---

## Problem Statement

Given a portfolio of coastal properties, estimate probable maximum loss (PML) at various return periods (1-in-100, 1-in-250 year events) using historical FEMA flood claim data and property characteristics.

This mirrors real actuarial and underwriting workflows used by E&S carriers and reinsurers to assess aggregate exposure and make capital allocation decisions.

---

## Project Goals

- Build a frequency + severity model to estimate individual property loss potential
- Simulate portfolio-level loss distributions using Monte Carlo methods
- Report industry-standard PML metrics at multiple return periods
- Calculate **marginal PML** — the change in portfolio risk from adding a single new account
- Deploy results as an interactive Streamlit dashboard

---

## Data Sources

| Source | What to Pull | URL |
|---|---|---|
| OpenFEMA NFIP Claims | Historical flood losses by location, property type, flood zone | https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2 |
| OpenFEMA NFIP Policies | Policy in force data — enables loss ratio calculation | https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2 |
| NOAA Storm Events Database | Hurricane/storm event data to tie losses to events | https://www.ncdc.noaa.gov/stormevents/ftp.jsp |
| Census TIGER/Line | Property and geographic features | https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html |
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

### Layer 3 — PML Simulation (Monte Carlo)
Combine frequency and severity into a portfolio-level loss distribution.

- **Method:** Monte Carlo simulation (100,000 iterations)
- **Per iteration:**
  1. For each property, sample claim occurrence from frequency model probability
  2. For properties with a claim, sample loss amount from severity model distribution
  3. Sum losses across the portfolio = one simulated annual loss
- **Output:** Full loss distribution
- **Reported metrics:**
  - Mean Annual Loss (MAL)
  - 1-in-10 year PML (90th percentile)
  - 1-in-100 year PML (99th percentile)
  - 1-in-250 year PML (99.6th percentile)
  - Loss Exceedance Curve (standard reinsurance output)

### Layer 4 — Marginal PML
Show how adding a new account shifts the aggregate portfolio PML.

- **Method:** Run simulation with and without the candidate property
- **Output:** Delta PML at each return period
- **Use case:** Mirrors real underwriting decision — "should we bind this risk given our current book?"

---

## Project Structure

```
flood-cat-model/
├── README.md                        ← Project overview, results, and key charts
├── PROJECT_SCOPE.md                 ← This file
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                         ← Downloaded source files (gitignored)
│   ├── processed/                   ← Cleaned, feature-engineered datasets
│   └── synthetic/                   ← Synthetic portfolio for demo purposes
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb      ← OpenFEMA API pull + NOAA join
│   ├── 02_eda.ipynb                 ← Exploratory analysis, loss distributions
│   ├── 03_feature_engineering.ipynb ← Feature construction, encoding
│   ├── 04_severity_model.ipynb      ← XGBoost severity model + SHAP analysis
│   ├── 05_frequency_model.ipynb     ← Frequency classifier + calibration
│   └── 06_pml_simulation.ipynb      ← Monte Carlo engine + loss exceedance curve
│
├── src/
│   ├── data_ingestion.py            ← OpenFEMA API client
│   ├── feature_engineering.py       ← Reusable feature pipeline
│   ├── severity_model.py            ← Trained model wrapper
│   ├── frequency_model.py           ← Trained model wrapper
│   ├── simulation.py                ← Monte Carlo simulation engine
│   └── pml_calculator.py            ← PML + marginal PML calculations
│
└── app/
    ├── app.py                       ← Streamlit dashboard
    └── assets/                      ← Charts, images for dashboard
```

---

## Streamlit Dashboard Features

1. **Portfolio Upload** — Upload a CSV of properties or use a pre-built synthetic portfolio
2. **Run Simulation** — Trigger Monte Carlo with configurable iterations
3. **Loss Exceedance Curve** — Visual output showing PML at all return periods
4. **Marginal PML Tool** — Input a new property's characteristics; see its impact on portfolio PML
5. **SHAP Explainability Panel** — Which features drive the largest individual losses

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
fastapi          # optional API layer
requests         # OpenFEMA API calls
faker            # synthetic data generation
joblib           # model serialization
```

---

## Resume Bullet (Final Framing)

> *"Built an end-to-end catastrophe loss model using FEMA NFIP data, combining XGBoost frequency/severity models with Monte Carlo simulation to estimate portfolio PML at multiple return periods, including marginal PML analysis — deployed as an interactive Streamlit application."*

---

## Development Phases

| Phase | Tasks | Est. Time |
|---|---|---|
| **1 — Data** | Pull OpenFEMA data, clean, join NOAA events | 1–2 weeks |
| **2 — EDA** | Explore loss distributions, feature correlations | 1 week |
| **3 — Models** | Build + tune severity and frequency models | 2–3 weeks |
| **4 — Simulation** | Monte Carlo engine + PML outputs | 1–2 weeks |
| **5 — Dashboard** | Streamlit app + loss exceedance curve | 1–2 weeks |
| **6 — Polish** | README, SHAP visuals, cleanup for GitHub | 1 week |
| **Total** | | **7–11 weeks** |

---

## Why This Project Works

- Uses **real, publicly available data** — no proprietary concerns
- Mirrors actuarial workflows common across E&S carriers and reinsurers (PML, marginal PML)
- Produces **industry-recognized outputs** (loss exceedance curve, return period PML)
- Demonstrates the full ML stack: data ingestion → feature engineering → modeling → simulation → deployment
- Rare in public portfolios — most ML projects don't touch cat modeling
- Speaks directly to hiring managers at reinsurers, E&S carriers, and cat modeling firms
