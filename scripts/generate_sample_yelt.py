"""
Generate realistic sample YELT (Year Event Loss Table) data.

Creates a portfolio YELT and account YELTs with correlated event losses
that mimic output from a catastrophe model (Touchstone / RMS / CoreLogic).

Key realism: The portfolio YELT is the SUM of all underlying account losses
within each (TrialID, EventID).  Account YELTs are subsets of the same
simulation — same event occurrences, same trial IDs — just as a real cat
model would produce.

Events share correlation via a shared "peril group factor" — e.g. all
Florida wind events in the same trial year are driven by a common storm
intensity.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'yelt'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(2026)

# ── Simulation parameters ──────────────────────────────────────────────
N_TRIALS   = 10_000    # simulation years
N_EVENTS   = 200       # distinct catastrophe events
N_ACCOUNTS = 10        # account YELTs to generate
N_BACKGROUND = 15      # extra "background" accounts in portfolio but not exported

# Event catalog
event_ids     = np.arange(1, N_EVENTS + 1)
event_rates   = np.random.uniform(0.002, 0.02, N_EVENTS)
event_mean_lr = np.random.uniform(0.02, 0.25, N_EVENTS)
event_cv      = np.random.uniform(0.3, 1.5, N_EVENTS)

# Assign events to 5 peril-region groups (drives correlation)
n_groups = 5
event_groups = np.random.randint(0, n_groups, N_EVENTS)

# ── Build account structure ────────────────────────────────────────────
account_names = [
    'ACCT_Florida_Wind', 'ACCT_California_EQ', 'ACCT_Gulf_Surge',
    'ACCT_Northeast_Winter', 'ACCT_Midwest_Tornado',
    'ACCT_Texas_Hail', 'ACCT_Pacific_NW_Flood', 'ACCT_Southeast_Convective',
    'ACCT_Mountain_Wildfire', 'ACCT_Atlantic_Hurricane',
]

# Each account (exported + background) has exposure to a random subset of events
all_accounts = []
for a in range(N_ACCOUNTS + N_BACKGROUND):
    n_events_exposed = np.random.randint(20, 80)
    exposed_idx = np.sort(np.random.choice(N_EVENTS, n_events_exposed, replace=False))
    exposure = np.random.uniform(5e5, 10e6, n_events_exposed)
    all_accounts.append((exposed_idx, exposure))

print(f"Generating YELT: {N_TRIALS:,} trials, {N_EVENTS} events, "
      f"{N_ACCOUNTS + N_BACKGROUND} accounts ({N_ACCOUNTS} exported) ...")

# ── Simulate all trials ───────────────────────────────────────────────
# For each trial: decide which events occur, then generate per-account losses
# using correlated severity.

# Pre-allocate storage: list of (TrialID, EventID, acct_index, loss)
all_loss_records = []

for trial in range(1, N_TRIALS + 1):
    # Shared peril-group intensity shocks (correlation source)
    group_shocks = np.random.lognormal(0, 0.5, n_groups)

    # Determine which events occur this trial (single Poisson draw per event)
    event_occurs = np.random.random(N_EVENTS) < event_rates

    for i in np.where(event_occurs)[0]:
        eid = event_ids[i]
        shock = group_shocks[event_groups[i]]

        # Each account exposed to this event gets a correlated loss draw
        for a_idx, (exposed_idx, exposure) in enumerate(all_accounts):
            # Is this account exposed to this event?
            pos = np.searchsorted(exposed_idx, i)
            if pos >= len(exposed_idx) or exposed_idx[pos] != i:
                continue

            mu_lr = event_mean_lr[i]
            cv = event_cv[i]
            sigma_ln = np.sqrt(np.log(1 + cv ** 2))
            mu_ln = np.log(mu_lr) - 0.5 * sigma_ln ** 2

            loss_ratio = np.random.lognormal(mu_ln, sigma_ln) * shock
            loss_ratio = min(loss_ratio, 1.0)
            loss = loss_ratio * exposure[pos]

            all_loss_records.append((trial, int(eid), a_idx, round(loss, 2)))

print(f"  Generated {len(all_loss_records):,} individual loss records")

# Convert to DataFrame
loss_df = pd.DataFrame(all_loss_records,
                        columns=['TrialID', 'EventID', 'AccountIdx', 'Loss'])

# ── Build portfolio YELT (sum across ALL accounts per TrialID/EventID) ──
portfolio_yelt = (
    loss_df.groupby(['TrialID', 'EventID'], sort=False)['Loss']
    .sum()
    .reset_index()
)
portfolio_yelt.to_csv(OUTPUT_DIR / 'portfolio_yelt.csv', index=False)

# Empirical PML sanity check
trial_sums = portfolio_yelt.groupby('TrialID')['Loss'].sum()
full_annual = np.zeros(N_TRIALS)
full_annual[trial_sums.index.values - 1] = trial_sums.values

print(f"\nPortfolio YELT: {len(portfolio_yelt):,} rows")
print(f"  Unique trials with losses: {portfolio_yelt['TrialID'].nunique():,}")
print(f"  AAL: ${np.mean(full_annual):,.0f}")
for rp in [50, 100, 250]:
    pctile = (1 - 1 / rp) * 100
    pml = np.percentile(full_annual, pctile)
    print(f"  {rp}-yr PML: ${pml:,.0f}")

# ── Build and save account YELTs ──────────────────────────────────────
print("\nAccount YELTs:")
for a_idx in range(N_ACCOUNTS):
    acct_losses = loss_df[loss_df['AccountIdx'] == a_idx][['TrialID', 'EventID', 'Loss']].copy()
    fname = f'{account_names[a_idx]}.csv'
    acct_losses.to_csv(OUTPUT_DIR / fname, index=False)

    acct_trial_sums = acct_losses.groupby('TrialID')['Loss'].sum()
    aal = acct_trial_sums.sum() / N_TRIALS
    print(f"  {account_names[a_idx]}: {len(acct_losses):,} rows, AAL=${aal:,.0f}")

print(f"\nDone.  All files in {OUTPUT_DIR}")
