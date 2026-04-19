"""
Generate sample test data for the PML Tool CLI demo.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Generate Portfolio ELT
# =============================================================================
n_events = 2000

portfolio = pd.DataFrame({
    'EventID': range(1, n_events + 1),  # Using common name, not EVENTID
    'Rate': np.random.uniform(0.001, 0.01, n_events),  # Common alias
    'ExpectedLoss': np.random.exponential(1e6, n_events),  # Will map to PERSPVALUE
    'MaxExposure': np.random.exponential(5e6, n_events),  # Will map to EXPVALUE
    'StdDevC': np.random.exponential(2e5, n_events),  # Will map to STDDEVC
    'StdDevI': np.random.exponential(3e5, n_events),  # Will map to STDDEVI
})

# Ensure MaxExposure > ExpectedLoss
portfolio['MaxExposure'] = portfolio[['ExpectedLoss', 'MaxExposure']].max(axis=1) * 1.5

# Save portfolio
portfolio.to_csv(output_dir / "portfolio.csv", index=False)
print(f"Created: {output_dir / 'portfolio.csv'} ({len(portfolio)} events)")

# =============================================================================
# Generate Sample Accounts
# =============================================================================
accounts_dir = output_dir / "accounts"
accounts_dir.mkdir(exist_ok=True)

# Create 10 sample accounts with varying sizes
account_configs = [
    ("ACCT_Commercial_Property", 150, 1e5),
    ("ACCT_Industrial_Complex", 200, 2e5),
    ("ACCT_Retail_Chain", 100, 5e4),
    ("ACCT_Office_Tower", 80, 8e4),
    ("ACCT_Warehouse_District", 120, 6e4),
    ("ACCT_Mixed_Use_Development", 180, 1.2e5),
    ("ACCT_Hotel_Resort", 90, 7e4),
    ("ACCT_Manufacturing_Plant", 110, 9e4),
    ("ACCT_Data_Center", 60, 3e5),
    ("ACCT_Healthcare_Campus", 140, 1.1e5),
]

for acct_name, n_events_acct, scale in account_configs:
    # Select random subset of portfolio events
    selected_events = np.random.choice(n_events, n_events_acct, replace=False) + 1
    
    account = pd.DataFrame({
        'Event_ID': selected_events,  # Different alias to test mapping
        'Mean_Loss': np.random.exponential(scale, n_events_acct),  # Another alias
        'Exposure': np.random.exponential(scale * 5, n_events_acct),  # Another alias
        'SD_C': np.random.exponential(scale * 0.2, n_events_acct),  # Alias
        'SD_I': np.random.exponential(scale * 0.3, n_events_acct),  # Alias
    })
    
    # Ensure Exposure > Mean_Loss
    account['Exposure'] = account[['Mean_Loss', 'Exposure']].max(axis=1) * 1.5
    
    # Save as CSV
    account.to_csv(accounts_dir / f"{acct_name}.csv", index=False)
    print(f"Created: {accounts_dir / acct_name}.csv ({len(account)} events)")

# Also create a single quote file for testing
single_quote = pd.DataFrame({
    'EVENTID': np.random.choice(n_events, 100, replace=False) + 1,
    'PERSPVALUE': np.random.exponential(1e5, 100),
    'EXPVALUE': np.random.exponential(5e5, 100),
    'STDDEVC': np.random.exponential(2e4, 100),
    'STDDEVI': np.random.exponential(3e4, 100),
})
single_quote['EXPVALUE'] = single_quote[['PERSPVALUE', 'EXPVALUE']].max(axis=1) * 1.5

# Save as Excel to test Excel reading
single_quote.to_excel(output_dir / "new_deal.xlsx", index=False)
print(f"Created: {output_dir / 'new_deal.xlsx'} ({len(single_quote)} events)")

print(f"\n✅ Sample data generated in: {output_dir}")
print(f"\nTest commands:")
print(f"  python src/pml_tool.py --portfolio data/portfolio.csv --quote data/new_deal.xlsx")
print(f"  python src/pml_tool.py --portfolio data/portfolio.csv --quote-folder data/accounts/ --batch")
