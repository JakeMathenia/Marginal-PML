"""
End-to-end validation script for the Coastal Property Catastrophe Loss Model.

Runs each stage of the pipeline on a small synthetic sample to confirm:
  1. All modules import cleanly
  2. Feature pipeline produces expected matrix shape & dtypes
  3. Severity model trains, evaluates, predicts
  4. Frequency model trains, evaluates, calibrates
  5. ELT generator produces valid 6-column ELT
  6. MarginalPMLEngine consumes the ELT and produces PML estimates
  7. Existing synthetic portfolio.csv still works with the engine

Run from project root:
    python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def make_synthetic_claims(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic NFIP-style claims frame that exercises every branch
    of the FeaturePipeline without needing the OpenFEMA API.
    """
    rng = np.random.default_rng(seed)

    zones = rng.choice(
        ["VE", "V", "AE", "A", "X", "B", "C"],
        size=n,
        p=[0.10, 0.05, 0.30, 0.20, 0.20, 0.075, 0.075],
    )
    occupancy = rng.choice([1, 2, 3, 4], size=n, p=[0.7, 0.2, 0.05, 0.05])
    basement = rng.choice([0, 1, 2, 3, 4], size=n)
    floors = rng.choice([1, 2, 3], size=n, p=[0.5, 0.4, 0.1])

    coverage = rng.lognormal(mean=12.5, sigma=0.6, size=n).clip(1_000, 750_000)
    bldg_val = coverage * rng.uniform(0.8, 1.5, size=n)
    construction_year = rng.integers(1950, 2023, size=n)

    states = rng.choice(["FL", "TX", "LA", "NC", "SC", "GA", "NJ", "NY"], size=n)
    years = rng.integers(2000, 2024, size=n)

    # ~12% claim rate; severity log-normal, scaled by zone risk
    zone_mult = pd.Series(zones).map(
        {"VE": 3.0, "V": 2.8, "AE": 1.7, "A": 1.5, "X": 1.0, "B": 0.9, "C": 0.9}
    ).values
    base_p = 0.06 + 0.10 * (zone_mult - 1.0) / 2.0
    base_p = np.clip(base_p, 0.01, 0.5)
    had_claim = rng.binomial(1, base_p).astype(int)

    claim_amt = np.where(
        had_claim == 1,
        np.exp(rng.normal(loc=9.5, scale=1.0, size=n)) * zone_mult,
        0.0,
    )

    # Storm metadata for ~30% of claims to test the storm-join branch
    storm_mask = (rng.random(n) < 0.30) & (had_claim == 1)
    event_type = np.where(
        storm_mask,
        rng.choice(
            ["Hurricane (Typhoon)", "Tropical Storm", "Storm Surge/Tide",
             "Coastal Flood", "Flash Flood"], size=n,
        ),
        None,
    )
    hurricane_cat = np.where(
        storm_mask & np.isin(event_type, ["Hurricane (Typhoon)"]),
        rng.integers(1, 6, size=n),
        np.nan,
    )
    damage_property = np.where(
        storm_mask, rng.lognormal(13, 1.5, size=n), 0.0
    )

    df = pd.DataFrame({
        "floodZone": zones,
        "occupancyType": occupancy,
        "basementEnclosureCrawlspaceType": basement,
        "numberOfFloorsInInsuredBuilding": floors,
        "totalBuildingInsuranceCoverage": coverage,
        "buildingPropertyValue": bldg_val,
        "originalConstructionDate": construction_year.astype(str) + "-01-01",
        "elevationCertificateIndicator": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        "reportedState": states,
        "yearOfLoss": years,
        "amountPaidOnBuildingClaim": claim_amt,
        "had_claim": had_claim,
        # storm join fields
        "matched_storm_id": np.where(storm_mask, np.arange(n), None),
        "EVENT_TYPE": event_type,
        "hurricane_category": hurricane_cat,
        "DAMAGE_PROPERTY_NUM": damage_property,
        "flood_zone_category": pd.Series(zones).map({
            "VE": "coastal_high_risk", "V": "coastal_high_risk",
            "AE": "high_risk", "A": "high_risk",
            "X": "minimal_risk", "B": "moderate_risk", "C": "moderate_risk",
        }).values,
    })
    return df


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def step_1_imports():
    banner("STEP 1 — Import all src modules")
    import data_ingestion        # noqa: F401
    import feature_engineering   # noqa: F401
    import severity_model        # noqa: F401
    import frequency_model       # noqa: F401
    import elt_generator         # noqa: F401
    import marginal_pml_kernel   # noqa: F401
    import pml_tool              # noqa: F401
    print("✅ All modules imported successfully.")


def step_2_features(claims):
    banner("STEP 2 — Feature engineering pipeline")
    from feature_engineering import FeaturePipeline, SEVERITY_FEATURES

    pipe = FeaturePipeline(reference_year=2023)
    X, y_sev, y_freq = pipe.fit_transform(claims, return_targets=True)

    print(f"  Feature matrix shape : {X.shape}")
    print(f"  Feature columns      : {len(X.columns)}")
    print(f"  Severity feature list: {len(SEVERITY_FEATURES)} cols")
    print(f"  y_freq positive rate : {y_freq.mean():.1%}  ({y_freq.sum()} of {len(y_freq)})")
    print(f"  y_sev non-null count : {y_sev.notna().sum()}")
    print(f"  Any NaN in X?        : {bool(X.isna().any().any())}")

    assert len(X) == len(claims), "Feature matrix row count mismatch"
    assert X.isna().sum().sum() == 0, "Features should not contain NaN"
    assert y_freq.dtype.kind in "iuf", "y_freq must be numeric"

    print("✅ Feature pipeline OK")
    return X, y_sev, y_freq


def step_3_severity(X, y_sev):
    banner("STEP 3 — Severity model training & evaluation")
    from severity_model import SeverityModel
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y_sev, test_size=0.25, random_state=0)

    sev = SeverityModel(params={
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "early_stopping_rounds": 20,
    })
    sev.fit(Xtr, ytr, verbose=False)

    metrics = sev.evaluate(Xte, yte)
    print(f"  Test R² (log)      : {metrics['r2']:.4f}")
    print(f"  Test RMSE (log)    : {metrics['rmse_log']:.4f}")
    print(f"  Test MAE  (log)    : {metrics['mae_log']:.4f}")
    print(f"  Train log RMSE (σ) : {sev.train_log_rmse:.4f}")
    print(f"  Test records used  : {metrics['n_test']}")

    assert sev.train_log_rmse > 0
    assert 0 < metrics["rmse_log"] < 10  # sanity bound for log-scale
    print("✅ Severity model OK")
    return sev


def step_4_frequency(X, y_freq):
    banner("STEP 4 — Frequency model training & evaluation")
    from frequency_model import FrequencyModel
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_freq, test_size=0.25, stratify=y_freq, random_state=0
    )

    freq = FrequencyModel(
        params={
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "early_stopping_rounds": 30,
        },
        calibrate=True,
    )
    freq.fit(Xtr, ytr, verbose=False)

    metrics = freq.evaluate(Xte, yte)
    probs = freq.predict_proba(Xte)

    print(f"  Test AUC-ROC      : {metrics['auc']:.4f}")
    print(f"  Test Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"  Brier Score       : {metrics['brier_score']:.4f}")
    print(f"  Log Loss          : {metrics['log_loss']:.4f}")
    print(f"  Baseline rate     : {freq.baseline_rate:.1%}")
    print(f"  Probs range       : [{probs.min():.4f}, {probs.max():.4f}]")

    assert metrics["auc"] > 0.5,            "AUC should beat coin flip"
    assert 0 <= probs.min() and probs.max() <= 1, "probs out of [0,1]"
    print("✅ Frequency model OK")
    return freq


def step_5_elt(X, freq, sev, claims):
    banner("STEP 5 — ELT generation from ML predictions")
    from elt_generator import ELTGenerator, validate_elt

    gen = ELTGenerator(freq_model=freq, sev_model=sev, corr_method="both")
    elt = gen.generate(
        X,
        metadata=claims,
        property_ids=claims.index.astype(str),
        portfolio_name="test_portfolio",
    )

    print(f"  ELT shape           : {elt.shape}")
    print(f"  Columns             : {list(elt.columns)}")
    print(f"  RATE       range    : [{elt['RATE'].min():.5f}, {elt['RATE'].max():.5f}]")
    print(f"  PERSPVALUE range    : [{elt['PERSPVALUE'].min():,.0f}, {elt['PERSPVALUE'].max():,.0f}]")
    print(f"  EXPVALUE   range    : [{elt['EXPVALUE'].min():,.0f}, {elt['EXPVALUE'].max():,.0f}]")
    print(f"  STDDEVC    mean     : {elt['STDDEVC'].mean():,.0f}")
    print(f"  STDDEVI    mean     : {elt['STDDEVI'].mean():,.0f}")

    val = validate_elt(elt)
    print(f"  validate_elt valid? : {val['valid']}")
    print(f"  total MAL           : ${val['total_mal']:,.0f}")
    print(f"  total TIV           : ${val['total_exposure']:,.0f}")
    print(f"  loss ratio          : {val['total_mal'] / val['total_exposure']:.3%}")
    if val["issues"]:
        for issue in val["issues"]:
            print(f"  ⚠️  {issue}")

    # Sanity checks
    expected_cols = {"EVENTID", "RATE", "PERSPVALUE", "EXPVALUE", "STDDEVC", "STDDEVI"}
    assert expected_cols <= set(elt.columns), f"Missing ELT cols: {expected_cols - set(elt.columns)}"
    assert (elt["RATE"] >= 0).all() and (elt["RATE"] <= 1).all()
    assert (elt["PERSPVALUE"] >= 0).all()
    assert (elt["EXPVALUE"] >= 0).all()
    assert (elt["PERSPVALUE"] <= elt["EXPVALUE"] + 1).all(), "Loss must not exceed exposure"

    print("✅ ELT generation OK")
    return elt


def step_6_pml_engine(elt):
    banner("STEP 6 — PML engine on ML-generated ELT")
    from marginal_pml_kernel import MarginalPMLEngine

    rps = (10, 25, 50, 100, 250, 500)

    # --- MarginalPMLEngine baseline EP curve (canonical 6-col ELT in) ---
    engine = MarginalPMLEngine(elt, return_periods=rps)
    baseline = engine.baseline_pmls
    print(f"  MarginalPMLEngine.baseline_pmls (mode={engine.mode}):")
    for rp in rps:
        print(f"    1-in-{rp:>4} yr PML  : ${baseline[rp]:>15,.0f}")
    print(f"  n_portfolio_events: {engine.n_portfolio_events:,}")

    # --- Marginal pricing of a 5-property account against the portfolio ---
    # Slice EVENTIDs that exist in the portfolio so the subtract is meaningful.
    account = elt.tail(5).copy()
    full = engine.price_account(account, include_combined_pml=True)
    print(f"\n  price_account(5-row account):")
    for rp in rps:
        print(f"    1-in-{rp:>4}: baseline=${full['baseline'][rp]:>12,.0f} "
              f"combined=${full['combined'][rp]:>12,.0f} "
              f"marginal=${full['marginal'][rp]:>+12,.0f}")

    # Monotonicity check on the baseline EP curve
    vals = [baseline[rp] for rp in rps]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1] - 1, (
            f"PML not monotonic: rp{rps[i-1]}=${vals[i-1]:,.0f} > rp{rps[i]}=${vals[i]:,.0f}"
        )
    print("\n✅ Baseline EP curve monotonically increases with return period")
    print("✅ MarginalPMLEngine round-trips ML-generated ELT cleanly")
    return baseline


def step_7_synthetic_round_trip():
    banner("STEP 7 — Existing data/portfolio.csv → FileIngestor → MarginalPMLEngine")
    portfolio_path = ROOT / "data" / "portfolio.csv"
    if not portfolio_path.exists():
        print(f"  ⚠️  {portfolio_path} not found — skipping")
        return None

    from pml_tool import FileIngestor
    from marginal_pml_kernel import MarginalPMLEngine

    # FileIngestor handles the column aliasing: EventID→EVENTID, Rate→RATE,
    # ExpectedLoss→PERSPVALUE, MaxExposure→EXPVALUE, StdDevC→STDDEVC, StdDevI→STDDEVI
    ingestor = FileIngestor(verbose=False)
    df = ingestor.ingest(portfolio_path, is_portfolio=True)
    print(f"  Loaded synthetic portfolio: {len(df):,} rows, cols={list(df.columns)}")

    rps = (50, 100, 250)
    engine = MarginalPMLEngine(df, return_periods=rps)
    results = engine.baseline_pmls

    print("  Synthetic ELT baseline PMLs:")
    for rp in rps:
        print(f"    1-in-{rp:>4} yr PML  : ${results[rp]:>15,.0f}")

    print("✅ Synthetic ELT works through FileIngestor + MarginalPMLEngine")
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        step_1_imports()

        claims = make_synthetic_claims(n=2000)
        print(f"\n  Synthetic claims set: {len(claims):,} rows | "
              f"~{claims['had_claim'].mean():.1%} positive class")

        X, y_sev, y_freq = step_2_features(claims)
        sev  = step_3_severity(X, y_sev)
        freq = step_4_frequency(X, y_freq)
        elt  = step_5_elt(X, freq, sev, claims)
        ml_results       = step_6_pml_engine(elt)
        synth_results    = step_7_synthetic_round_trip()

        banner("PIPELINE VALIDATION SUMMARY")
        print(f"  Features built          : {X.shape[1]} columns")
        print(f"  Severity train log RMSE : {sev.train_log_rmse:.4f}")
        print(f"  Frequency baseline rate : {freq.baseline_rate:.1%}")
        print(f"  ELT rows generated      : {len(elt):,}")
        print(f"  ML 100-yr PML           : ${ml_results[100]:,.0f}")
        if synth_results is not None:
            print(f"  Synthetic 100-yr PML    : ${synth_results[100]:,.0f}")
        print("\n  ✅✅✅ ALL PIPELINE STAGES PASSED ✅✅✅\n")
        return 0
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        import traceback; traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
