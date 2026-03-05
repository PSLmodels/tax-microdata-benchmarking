# National Reweighting: Targeted Variables Summary

*Generated from `reweight.py` and `soi.csv` analysis.*

## Overview

The national reweighting optimizer targets **550 SOI statistics** (for 2021).
These come from 20 targeted variables divided into two groups.

## AGI-Level Variables (11 variables, 532 targets)

These variables are targeted **by AGI income bin** (19 bins per variable).
Each bin has both an amount target and a count target.

| Variable | Filing statuses | Targets |
|----------|----------------|---------|
| `adjusted_gross_income` | All, Single, MFJ/SS, MFS, HoH | 19 x 5 = 95 |
| `count` | All, Single, MFJ/SS, MFS, HoH | 19 x 5 = 95 |
| `employment_income` | All only | 19 x 2 = 38 |
| `business_net_profits` | All only | 19 x 2 = 38 |
| `capital_gains_gross` | All only | 19 x 2 = 38 |
| `ordinary_dividends` | All only | 19 x 2 = 38 |
| `partnership_and_s_corp_income` | All only | 19 x 2 = 38 |
| `qualified_dividends` | All only | 19 x 2 = 38 |
| `taxable_interest_income` | All only | 19 x 2 = 38 |
| `total_pension_income` | All only | 19 x 2 = 38 |
| `total_social_security` | All only | 19 x 2 = 38 |

**Subtotal: 532 targets**

Notes:
- `adjusted_gross_income` and `count` get all 5 filing statuses (from Table 1.2)
- The other 9 variables get only "All" filing status (from Table 1.4)
- Each variable gets 2 rows per bin: one amount, one count (nonzero returns)
- 19 AGI bins from Table 1.4: (-inf,0), (1,5k), (5k,10k), ..., (10M,inf)

## Aggregate-Level Variables (9 variables, 18 targets)

These variables are targeted at **full-population level only** (no AGI binning).
Each has an amount target and a count target.

| Variable | Targets |
|----------|---------|
| `business_net_losses` | 2 (amount + count) |
| `capital_gains_distributions` | 2 |
| `capital_gains_losses` | 2 |
| `exempt_interest` | 2 |
| `ira_distributions` | 2 |
| `partnership_and_s_corp_losses` | 2 |
| `taxable_pension_income` | 2 |
| `taxable_social_security` | 2 |
| `unemployment_compensation` | 2 |

**Subtotal: 18 targets**

## Not Targeted (in soi.csv but not used by reweight.py)

These variables are present in `soi.csv` but are **commented out** in `reweight.py`
because Tax-Calculator does not model them (all values are zero in `tc_to_soi()`):

- `estate_income` -- not in Tax-Calculator
- `estate_losses` -- not in Tax-Calculator
- `rent_and_royalty_net_income` -- not in Tax-Calculator
- `rent_and_royalty_net_losses` -- not in Tax-Calculator

Additionally, the `_drop_impossible_targets()` function removes any targets where
all data values are zero, providing a safety net even if these were uncommented.

## Target Count by Year

| Year | soi.csv rows | Targeted by optimizer | Notes |
|------|-------------|----------------------|-------|
| 2015 | 1,864 | 550* | Same variable structure |
| 2021 | 1,986 | 550 | Current default year |
| 2022 | 1,830 | ~542** | Excludes rentroyalty/estateincome |

\* Approximate; 2015 has some different AGI bin breakdowns.
\** 2022 has fewer soi.csv rows because rentroyalty/estateincome excluded
(PUF variables don't align with IRS definitions due to passive activity
limitation differences). The optimizer will target the same 9 aggregate
variables that have data.

## AGI Income Bins (19 bins)

From `INCOME_RANGES` in `reweight.py`:

| Bin | Lower bound | Upper bound |
|-----|------------|-------------|
| 1 | -inf | $1 |
| 2 | $1 | $5,000 |
| 3 | $5,000 | $10,000 |
| 4 | $10,000 | $15,000 |
| 5 | $15,000 | $20,000 |
| 6 | $20,000 | $25,000 |
| 7 | $25,000 | $30,000 |
| 8 | $30,000 | $40,000 |
| 9 | $40,000 | $50,000 |
| 10 | $50,000 | $75,000 |
| 11 | $75,000 | $100,000 |
| 12 | $100,000 | $200,000 |
| 13 | $200,000 | $500,000 |
| 14 | $500,000 | $1,000,000 |
| 15 | $1,000,000 | $1,500,000 |
| 16 | $1,500,000 | $2,000,000 |
| 17 | $2,000,000 | $5,000,000 |
| 18 | $5,000,000 | $10,000,000 |
| 19 | $10,000,000 | inf |

## Key Source Files

- Target definitions: `tmd/utils/reweight.py` (lines 73-100)
- Target data: `tmd/storage/input/soi.csv`
- Converter: `tmd/national_targets/potential_targets_to_soi.py`
- IRS-to-PUF-to-TMD mapping: `tmd/national_targets/data/irs_to_puf_map.json`
