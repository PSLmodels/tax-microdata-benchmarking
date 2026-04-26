# Area weight files

Output directory for the area-weighting pipeline.  Each successful
solver run for an area writes a weight file and a solver log here.
Multi-area quality reports add an all-areas summary CSV.

This directory holds **outputs only** — to (re)build them, see
[../README.md](../README.md).

## Layout

```
weights/
├── states/                 51 states + DC
│   ├── ca_tmd_weights.csv.gz
│   ├── ca.log
│   ├── ...
│   └── quality_report_per_area.csv     (after a multi-state quality run)
├── cds_118/                436 CDs on 118th Congress boundaries
│   ├── al01_tmd_weights.csv.gz
│   ├── al01.log
│   └── ...
└── cds_119/                436 CDs on 119th Congress boundaries
    └── ...
```

State files use the lowercase two-letter postal code (e.g., `ca`,
`ny`, `dc`).  CD files use the lowercase state code plus a
two-digit district number (e.g., `mn01`, `ny12`).

## Weight file: `<area>_tmd_weights.csv.gz`

A gzipped CSV with **one row per TMD record** (one filing unit), in
the same row order as the national TMD file
(`tmd/storage/output/tmd.csv.gz`).  This row alignment is what lets
you swap an area weight file into Tax-Calculator in place of the
national weights file.

The columns are weight values, one per calendar year:

```
WT2022, WT2023, WT2024, WT2025, WT2026, WT2027, WT2028,
WT2029, WT2030, WT2031, WT2032, WT2033, WT2034
```

Each value is the area-specific weight for that record in that
year, on the same scale as the national `s006` weight (i.e.,
already divided — not in hundredths).  Weighted sums of any TMD
variable using these weights estimate the corresponding total for
that area.

Records that the optimizer effectively drops from the area appear
with weight 0 in every year.  This is normal; a typical state has
6–10% zero weights.

## Solver log: `<area>.log`

A short text file that records what the solver did for that area.
Useful when you want to confirm a run was clean without opening the
full quality report.  Key sections:

- **Population share** used to seed the optimizer.
- **Pre-solve feasibility** check (LP).
- **Solver status, iterations, solve time** (Clarabel output).
- **Target accuracy** — number of targets, how many hit the
  tolerance band, mean and max relative error.
- **Multiplier distribution** — histogram of the per-record weight
  multiplier (`area_weight / population-proportional weight`),
  including share at zero.
- **Slack** — for any target the solver could not hit exactly,
  how far off it landed and which constraint it was.

## All-areas summary: `quality_report_per_area.csv`

Written by `tmd.areas.quality_report` whenever it runs over more
than one area (e.g., `--scope states` or `--scope cds`).  One row
per area, with these columns:

| Column | Meaning |
|---|---|
| `area` | Area code (e.g., `ca`, `mn01`) |
| `n_records` | Records with non-zero weight |
| `pct_zero` | Share of records the solver dropped to zero |
| `w_median`, `w_p95`, `w_max` | Multiplier distribution (vs. population-proportional baseline of 1.0) |
| `returns` | Weighted return count |
| `agi` | Weighted AGI |
| `avg_agi` | Mean AGI per return |
| `unusualness` | Summary measure of how far this area's record mix is from the national mix |
| `ess` | Kish effective sample size (smaller = more weight concentrated on few records) |

The text quality report truncates per-area tables for CDs (top 20
most-distorted areas) to stay readable; this CSV does not truncate
and is the right input for cross-area analysis, sorting, or
plotting.

## Using a weight file

A weight file is just a CSV with one weight per record per year.
**Computing weighted area totals never requires Tax-Calculator.**
Tax-Calculator computes per-record tax variables (taxes owed,
credits, and so on) without using weights at all — weights enter
only when you aggregate across records.  Once you have a record-
level file that contains the variables you care about, area
totals are just `(values * area_weights).sum()`.

### Variables already on the TMD microdata file

For variables that live directly on `tmd.csv.gz` — input variables
like `e00200` (wages) and `e00300` (taxable interest), the
demographic variables, and so on — you only need pandas.  The
values stored on `tmd.csv.gz` are in TMD-base-year (2022) dollars,
so a weighted sum using the matching `WT2022` column gives the
2022 area total.  For example, weighted total wages in California
in 2022:

```python
import pandas as pd
from tmd.storage import STORAGE_FOLDER

tmd = pd.read_csv(STORAGE_FOLDER / "output" / "tmd.csv.gz")
wts = pd.read_csv(
    STORAGE_FOLDER.parent / "areas" / "weights" / "states"
    / "ca_tmd_weights.csv.gz"
)

ca_wages_2022 = (tmd["e00200"] * wts["WT2022"]).sum()
```

For a later year you also need to age the underlying values to that
year using the TMD growfactors (or use the cached Tax-Calculator
outputs described in the next section, which already contain
year-aged values).

### Tax-Calculator output variables (income tax, payroll tax, etc.)

Tax-Calculator output variables (such as `iitax`, `payrolltax`,
`c00100`, `c18300`) are not on `tmd.csv.gz`.  To compute area
weighted totals of these variables you need a record-level file
that contains them.  Two common ways to get one:

- Use the unweighted cached outputs that the repo builds during
  `make data` (`tmd/storage/output/cached_allvars.csv` covers many
  Tax-Calculator outputs at current law for the TMD tax year).
- For a different policy or year, run Tax-Calculator once over the
  national TMD microdata to produce per-record outputs and save
  them to a CSV.  The choice of national weights does not affect
  the per-record values.

With such a file in hand, area weighted totals are again a one-line
pandas expression — no Tax-Calculator call needed at aggregation
time.

### Computing inside Tax-Calculator with area weights

If you would rather chain calculation and aggregation inside
Tax-Calculator (for example, to apply a policy reform and read off
area totals from its tabulators), the area weight file is a
drop-in replacement for the national `tmd_weights.csv.gz` when
constructing a `Records` object.  Sketched in Python:

```python
import taxcalc
from tmd.imputation_assumptions import TAXYEAR
from tmd.storage import STORAGE_FOLDER

# National TMD inputs (microdata + growfactors)
tmd_csv          = STORAGE_FOLDER / "output" / "tmd.csv.gz"
tmd_growfactors  = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"

# Area weight file (here: California)
area_weights = (
    STORAGE_FOLDER.parent / "areas" / "weights" / "states"
    / "ca_tmd_weights.csv.gz"
)

records = taxcalc.Records(
    data=str(tmd_csv),
    weights=str(area_weights),
    gfactors=taxcalc.GrowFactors(growfactors_filename=str(tmd_growfactors)),
    start_year=TAXYEAR,
    adjust_ratios=None,
    exact_calculations=True,
    weights_scale=1.0,
)

# The same Tax-Calculator policy + calculator setup you use for the
# national file now produces California-level estimates.
```

Two things to keep in mind:

- **Year alignment.**  The weight columns cover `WT2022` through
  `WT2034`.  Tax-Calculator picks the column whose year matches the
  current calculation year.  Calculating outside that range will
  fail or use the boundary weights, depending on Tax-Calculator's
  growth-rule settings.
- **Sum-of-areas vs national.**  For each year, the sum of all area
  weights for a record is likely to be close to that record's
  national `s006`, but for some records the sum may be substantially
  different.  The `quality_report` cross-area aggregation check
  reports how this affects national totals built up from the area
  weights; for policy-relevant variables the discrepancy is small.

For the broader Tax-Calculator workflow with the national TMD
inputs, see the upstream documentation:
[taxcalc.pslmodels.org/usage/data.html](https://taxcalc.pslmodels.org/usage/data.html#irs-public-use-data-tmd-csv).
