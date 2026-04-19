# Enhanced PML Reporting Feature

## Overview

The Marginal PML Pricing Engine now supports displaying **baseline portfolio PMLs** and **combined portfolio PMLs** in addition to the marginal impact values.

## What's New

### New Command-Line Flag: `--show-combined-pml`

When you add this flag, the output will include:

1. **Baseline PML**: The original portfolio PML (before any changes)
2. **Combined PML**: The portfolio PML after adding/subtracting the account
3. **Marginal PML**: The difference (what you already had before)

### Example Usage

#### Single Account Pricing

```bash
pml-tool --portfolio portfolio.csv --quote account.csv --show-combined-pml
```

#### Batch Pricing

```bash
pml-tool --portfolio portfolio.csv --quote-folder ./accounts/ --show-combined-pml --batch
```

## Output Format

### Without `--show-combined-pml` (Original Behavior)

Excel columns:
- Account ID
- Marginal PML (50yr)
- Marginal PML (100yr)
- Marginal PML (250yr)
- [pricing columns...]

### With `--show-combined-pml` (Enhanced Reporting)

Excel columns:
- Account ID
- **Portfolio PML (50yr)** ← NEW
- **Portfolio±Account PML (50yr)** ← NEW
- Marginal PML (50yr)
- **Portfolio PML (100yr)** ← NEW
- **Portfolio±Account PML (100yr)** ← NEW
- Marginal PML (100yr)
- **Portfolio PML (250yr)** ← NEW
- **Portfolio±Account PML (250yr)** ← NEW
- Marginal PML (250yr)
- [pricing columns...]

## Example Output

Here's a sample from batch pricing with `--show-combined-pml`:

| Account ID | Portfolio PML (100yr) | Portfolio±Account PML (100yr) | Marginal PML (100yr) |
|------------|----------------------|-------------------------------|---------------------|
| ACCT_Commercial_Property | $7,348,137 | $7,346,773 | $1,364 |
| ACCT_Data_Center | $7,348,137 | $7,348,977 | -$841 |
| ACCT_Healthcare_Campus | $7,348,137 | $7,342,572 | $5,565 |

## Why This Matters

### 1. **Transparency**
You can see the actual portfolio PMls before and after adding/removing the account, not just the difference.

### 2. **Business Decisions**
Sometimes you need to know the absolute PML values for capital planning, not just the marginal impact.

### 3. **Validation**
You can verify the math yourself:
- **Subtract mode**: Marginal = Baseline - Combined
- **Add mode**: Marginal = Combined - Baseline

### 4. **Negative Marginals**
When marginal is negative, you can see that the combined PML is actually higher/lower than baseline, indicating diversification benefit.

## Mathematical Verification

### Subtract Mode (RI Pricing)
```
Formula: Marginal = PML(Portfolio) - PML(Portfolio - Account)

Example:
  Portfolio PML (100yr):        $7,348,137
  Portfolio-Account PML (100yr): $7,346,773
  Marginal Impact:              $1,364

Verification: $7,348,137 - $7,346,773 = $1,364 ✓
```

### Add Mode (New Business)
```
Formula: Marginal = PML(Portfolio + Account) - PML(Portfolio)

Example:
  Portfolio PML (100yr):        $7,348,137
  Portfolio+Account PML (100yr): $7,349,517
  Marginal Impact:              $1,380

Verification: $7,349,517 - $7,348,137 = $1,380 ✓
```

## Implementation Details

### API Changes

The `MarginalPMLEngine.price_account()` method now accepts an optional parameter:

```python
def price_account(
    self,
    account_elt: pd.DataFrame,
    return_dict: bool = True,
    include_combined_pml: bool = False  # ← NEW PARAMETER
) -> Dict[int, float]:
```

**When `include_combined_pml=False` (default):**
```python
{
    50: 2922.0,    # Marginal PML at 50yr
    100: 1364.0,   # Marginal PML at 100yr
    250: 716.0     # Marginal PML at 250yr
}
```

**When `include_combined_pml=True`:**
```python
{
    'baseline': {50: 6333159.0, 100: 7348137.0, 250: 7859724.0},
    'combined': {50: 6330237.0, 100: 7346773.0, 250: 7859008.0},
    'marginal': {50: 2922.0, 100: 1364.0, 250: 716.0}
}
```

### Batch Processing

The `price_account_batch()` method also supports this parameter and will automatically flatten the results into appropriate DataFrame columns.

## Testing

Three test scripts are provided:

1. **test_combined_pml.py**: Basic functionality test
2. **demo_combined_pml.py**: Interactive comparison demo
3. **show_results.py**: Display Excel output in terminal

Run any of these to see the feature in action:

```bash
python test_combined_pml.py
python demo_combined_pml.py
```

## Backward Compatibility

This is a **fully backward-compatible** change:

- Existing code and scripts continue to work unchanged
- The default behavior (`include_combined_pml=False`) matches the original implementation
- The new feature is opt-in via the `--show-combined-pml` flag

## Performance

No performance impact when the flag is not used. When enabled, there is negligible overhead as the combined PML values are already computed internally—we're just exposing them in the output.
