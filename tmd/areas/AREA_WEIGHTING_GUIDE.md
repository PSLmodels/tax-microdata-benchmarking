# Area Weighting Guide

How area-specific weights are constructed for TMD, and how to update
them for new data.

## Overview

TMD is a national microdata file — every record represents tax filers
across the entire US.  Area weighting creates per-area weight vectors
that make the same records represent a specific state, congressional
district, or county.

### The optimization problem

For each area we solve:

    minimize    sum( (x[i] - 1)^2 )
    subject to  target[j] * (1 - tol) <= sum(B[j,i] * x[i]) <= target[j] * (1 + tol)
                x_min <= x[i] <= x_max

where `x[i]` is a weight multiplier for record `i`, and the area
weight for that record is:

    area_weight[i] = x[i] * pop_share * national_weight[i]

The term `pop_share * national_weight[i]` is the weight a record
would get if the area looked exactly like the nation, just with fewer
people.  The optimizer adjusts `x[i]` away from 1.0 only as much as
needed to match area-specific targets within tolerance.

### Targets and constraints

Each **target** is a weighted sum we want the area to match — for
example, "total wages in the $50K-$75K AGI bin for Alabama CD-1
should be $2.01 billion."  The constraint matrix `B` encodes which
records contribute to each target and by how much.

Targets come from IRS Statistics of Income (SOI) data, which
publishes geographic breakdowns of income, deductions, and credits.
We scale SOI geographic proportions by TMD national totals to get
area-level targets that sum exactly to national TMD values.

### Why this formulation works

We chose quadratic programming because:

1. **Sensible default:** `x[i] = 1` means "this record gets its
   population-proportional share."  The optimizer only departs from
   this when the data requires it.
2. **Robust:** The quadratic objective + linear constraints always
   has a unique solution (given feasibility), making results
   reproducible across machines.
3. **Fast:** The Clarabel QP solver handles 215,000 records × 92
   constraints in ~40 seconds per area.
4. **Elastic slack:** If a target can't be hit exactly, slack
   variables absorb the gap rather than making the problem infeasible.

### Concrete example: Alabama CD-1

AL-01 has about 315,000 tax returns (0.17% of US returns) and
718,000 people (0.21% of US population).  Its pop_share is 0.00215.

Before optimization, every record gets `x[i] = 1`, meaning the
area looks exactly like 0.215% of the nation.  But AL-01 has
relatively more retirees and fewer high-wage earners than the US
average.

The optimizer adjusts: records with pension income get `x` slightly
above 1.0 (more weight), while records with very high wages get `x`
below 1.0 (less weight).  For AL-01, the median multiplier is 0.93
and the RMSE from 1.0 is 0.48 — most records stay close to their
starting weights.

The 92 targets constrain things like "AL-01 wages in the $100K-$200K
bin should be $4.35B" and "AL-01 total single returns should be
150,125."  The optimizer finds the smallest x-adjustments that hit
all 92 targets within ±0.5%.

## Shares: separating geography from levels

Before diving into the pipeline architecture, it helps to understand
why we decompose targets into **shares** and **national sums**.

A target like "wages in the $50K-$75K AGI bin for AL01" depends on
two things:
1. What fraction of national wages in that bin belong to AL01?
   (geographic distribution — from SOI)
2. What are national wages in that bin?
   (national level — from TMD)

Item 1 comes from IRS SOI data and changes only with a new SOI
vintage (~annually).  Item 2 comes from TMD and changes every time
imputations are updated.

    share  = area_SOI_value / national_SOI_value
    target = TMD_national_sum × share

By pre-computing and saving shares, we avoid re-ingesting SOI data
on every TMD rebuild.  The shares file is a stable artifact; only the
TMD national sums need recomputing when the microdata changes.

## Architecture

```
SOI data + crosswalks → shares (stable)        ← rarely changes
                              ↓
TMD data (cached_allvars) → national sums       ← changes with TMD rebuilds
                              ↓
          shares × national sums = potential targets
                              ↓
          target spec → select from potential    ← changes during recipe tuning
                              ↓
          per-area _targets.csv files
                              ↓
          QP solver (Clarabel)
                              ↓
          per-area _tmd_weights.csv.gz files
```

Three artifacts, three change frequencies:

| Artifact | Example | Changes when |
|----------|---------|-------------|
| Shares file | `cds_shares.csv` | New SOI vintage (~annually) |
| Target spec | `cd_target_spec.csv` | Recipe tuning |
| Target files | `al01_targets.csv` | TMD rebuild or recipe change |

## Variable Name Mapping

The pipeline bridges three naming systems:

| SOI raw | SOI base name | TMD/Tax-Calculator | Description |
|---------|---------------|-------------------|-------------|
| A00100 | 00100 | c00100 | AGI (computed) |
| A00200 | 00200 | e00200 | Wages (input) |
| A01700 | 01700 | e01500 *and* e01700 | Pensions (total vs taxable) |
| A02500 | 02500 | e02400 *and* c02500 | Social Security (total vs taxable) |
| N1 | n1 | c00100 (count=1) | Number of returns |
| MARS1 | mars1 | c00100 (count=1, fstatus=1) | Single returns |

Key subtlety: multiple TMD variables can share the same SOI geographic
distribution.  For example, both `e01500` (total pensions) and `e01700`
(taxable pensions) use SOI `A01700` for their geographic shares because
SOI only publishes the taxable component.  The shares are the same —
only the TMD national sum differs.

The mapping is defined in `ALL_SHARING_MAPPINGS` in `constants.py`.
Extended targets add more mappings in `EXTENDED_SHARING_MAPPINGS`
in `prepare_shares.py`.

## The Target Spec

The spec is a flat CSV where each row is one target — what you see is
what gets solved.

```csv
varname,count,scope,fstatus,agilo,agihi,description
XTOT,0,0,0,-9e+99,9e+99,Population amount all bins
c00100,0,1,0,-9e+99,1.0,AGI amount <$0K
c00100,0,1,0,1.0,10000.0,AGI amount $0K-$10K
...
eitc,0,1,0,-9e+99,9e+99,EITC amount all bins
```

Column meanings:
- **varname**: TMD variable name
- **count**: 0 = dollar amount, 1 = all returns, 2 = nonzero count
- **scope**: 0 = all records (XTOT only), 1 = PUF records
- **fstatus**: 0 = all, 1 = single, 2 = MFJ, 4 = HoH
- **agilo/agihi**: AGI bin boundaries (-9e99/9e99 = all bins)
- **description**: human-readable label (ignored by pipeline)

To add a target: add a row.  To remove one: delete the row.
No crossing, no exclude lists, no indirection.

## Searching for Proxies

Not every TMD variable has a direct SOI counterpart.  When the
variable you want to target doesn't have SOI data at the right
geographic level, you need a proxy.

**Strategy for finding proxies:**

1. **Direct match:** TMD `e00200` (wages) → SOI `A00200` (wages).
   Best case.

2. **Related variable:** TMD `e01500` (total pensions) → SOI `A01700`
   (taxable pensions).  The taxable component has a similar geographic
   distribution to total pensions.

3. **Census data:** SALT deductions → Census state/local finance data
   provides property tax and sales tax collections by state.  Better
   geographic distribution than SOI for capped deductions. Only
   available at the state level, not CD or county.

4. **State-average approximation:** When proxy data exists at a coarser
   level (e.g., state but not CD), use the coarse share and distribute
   within the state by SOI proportions.  Example: CD SALT targets use
   SOI CD columns (a18425, a18500) as a proxy for Census state data.

5. **Aggregate (no AGI breakdown):** For variables where per-bin
   geographic variation is unreliable, use a single all-bins target.
   Examples: EITC, CTC.

## Establishing Base Targets and Expanding Incrementally

Start conservative, expand as feasibility allows.

### Base targets (high confidence)
- Income amounts by AGI bin: AGI, wages, interest, pensions, SS, SALT, partnership
- Return counts in upper-income bins ($25K+)
- Filing-status totals (single, MFJ, HoH — one all-bins target each)
- Population (XTOT)

### First extension: total-only (one target per variable, no bins)
- Additional income types: dividends, business income, capital gains
- Deductions: mortgage interest, charitable
- SALT components: income/sales, real estate
- Credits: EITC, CTC (amount + nonzero count)

Total-only targets are almost risk-free — they add one constraint
each and the solver has full freedom to distribute across bins.

### Second extension: per-bin for selected variables and stubs
- Capital gains in upper stubs ($100K+) — rich people have them
- Use developer mode difficulty table to assess feasibility

### What NOT to target
- Variables with very thin cells (few records in a bin)
- Variables where the SOI and TMD definitions diverge significantly
- Negative-AGI bins for variables concentrated among retirees (e02400)
- Per-bin credit targets (EITC, CTC) — see targeting rules below

## Targeting Rules of Thumb

Lessons learned from CD pipeline development (436 areas).

### 1. Target difficulty = gap from proportionate share

Use `python -m tmd.areas.developer_mode --difficulty AL01` to see how
far each target is from what the area would get under population-
proportionate allocation.  This is the single most useful diagnostic.

- **Easy (<5% gap):** Solver barely moves weights.  Free to add.
- **Moderate (5–20%):** Some weight distortion.  Generally fine.
- **Hard (20–50%):** Significant weight movement.  Worth targeting
  if the variable is policy-relevant, but watch for interactions.
- **Very hard (>50%):** Extreme weight distortion.  May destabilize
  other targets.  Consider total-only instead of per-bin.

Example: Alabama CD-1 mean |gap| = 23%, manageable.  Manhattan
(NY-12) mean |gap| = 340% — an extreme outlier requiring many
dropped targets or raised tolerances.

### 2. Solve time scales super-linearly with target count

Clarabel QP solver time scales worse than O(n²) in the number of
targets.  Benchmarks on a single CD (AL-01):

| Targets | All-bin rows | Solve time | Notes |
|---------|-------------|-----------|-------|
| 78 | 5 | 7s | Base recipe |
| 92 | 19 | 12s | +14 total-only extended |
| 95 | 19 | 14s | +3 capgains upper bins |
| 107 | 19 | 92s | +12 credit per-bin (problematic) |

The cost of additional targets depends on what they are, not just
how many.

### 3. Dense constraint rows are expensive

An "all-bin" target (agilo=-inf, agihi=+inf) touches every PUF
record in the B matrix (~97% of records).  A per-bin target touches
only records in that AGI bin (~5–30%).  Dense rows make the solver's
matrix factorization harder.

However, total-only targets are still worthwhile — they add modest
cost (12s→14s for 14 total-only targets) and constrain aggregate
quantities that would otherwise drift.

### 4. Per-bin credit targets are extremely difficult

EITC and CTC have sharp eligibility cliffs.  Per-bin targets require
the solver to match both income distribution AND credit distribution
within each bin simultaneously.  For AL-01:
- EITC $10K–$25K: +86% gap (needs heavy upweighting)
- CTC $10K–$25K: -89% gap (needs heavy downweighting)
- These pull weights in opposite directions → solver struggles

**Recommendation:** Target credits as all-bin totals only.  The
totals constrain aggregate credit amounts without forcing per-bin
precision that the microdata can't deliver.

### 5. Capital gains per-bin targets are feasible (for upper stubs)

Capital gains are concentrated in high-income bins where there are
plenty of records.  Adding 3 per-bin targets ($100K+) costs only
1.5s and all hit.  The gap is moderate (~50% for $500K+).

### 6. Conflicting targets cause solver explosions

When two targets require opposite weight adjustments for overlapping
record sets, the solver thrashes.  Signs of conflict:
- Solve time jumps disproportionately (12s → 92s for 15 targets)
- Many violated targets in the solution
- High RMSE (weight multipliers far from 1.0)

Use the difficulty table to spot targets with large gaps in opposite
directions.  If variable A needs +60% and variable B needs -60% in
the same AGI bin, one should be dropped or made total-only.

### 7. Start total-only, then add bins selectively

The incremental approach that works:
1. Total-only targets for all variables (low risk, fast)
2. Per-bin targets for variables where geography matters most and
   the difficulty table shows moderate gaps
3. Use developer mode to test each addition on a few representative
   areas before committing to a full batch run

### 8. Area-specific overrides are normal

Not every area can hit every target.  The override YAML file records
per-area adjustments.  For 436 CDs with 95 targets:
- ~80% solve with default params
- ~17% need 1–8 targets dropped
- ~3% need raised tolerance or multiplier cap

This is acceptable.  The alternative — a recipe so conservative that
every area solves — would sacrifice accuracy for the 80% of areas
that can handle more targets.

## Developer Workflow: Expanding a Recipe

Step-by-step process for adding new targets to a recipe.

### Step 1: Identify high-value targets

Decide which variables matter for your use case.  Rank by policy
importance.  Income variables (wages, AGI, capital gains) are
usually more important than deduction details.  Credits matter
for distributional analysis.

### Step 2: Run difficulty tables on representative areas

Pick 3–4 areas spanning the difficulty spectrum:
- A typical/easy area (e.g., AL-01, MN-03)
- A hard area (e.g., NY-12, TX-20)
- An area similar to your analysis focus

```bash
python -m tmd.areas.developer_mode --difficulty AL01
python -m tmd.areas.developer_mode --difficulty NY12
```

For each proposed target, check the gap%.  If most areas show
<30% gap, the target is likely feasible.  If the hard areas
show >100% gap, consider total-only instead of per-bin.

### Step 3: Test on a single easy area

Add the new targets to the spec and solve one easy area:

```bash
python -m tmd.areas.prepare_targets --scope AL01
python -m tmd.areas.solve_weights --scope AL01
```

Check solve time, violations, and RMSE.  If solve time jumps
disproportionately (e.g., 12s → 90s for 15 new targets),
the new targets have constraint interactions.  Try adding
them one at a time to find the culprit.

### Step 4: Test on a hard area

Repeat on NY-12 or another difficult CD.  If it fails, the
auto-relaxation cascade can find which targets to drop:

```bash
python -m tmd.areas.developer_mode --scope NY12 --verbose
```

### Step 5: Run dual analysis on problem areas

If a target causes unexpected solver difficulty despite a moderate
gap%, check the shadow prices:

```bash
python -m tmd.areas.developer_mode --dual AL01
```

High duals identify constraints that conflict with each other.
A target may look easy (10% gap) but have a massive dual because
it pulls against another target in the same record set.

### Step 6: Full batch run + quality report

Once satisfied with representative areas, run the full batch:

```bash
python -m tmd.areas.developer_mode --scope cds --workers 16
python -m tmd.areas.solve_weights --scope cds --workers 16
python -m tmd.areas.quality_report --scope cds --output
```

Compare the quality report against the previous version.
Check bystander distortion for the newly targeted variables.

### Step 7: Iterate

If too many areas need overrides, the recipe is too aggressive.
If bystanders show high distortion, more targeting is needed.
The goal is a recipe where ~80% of areas solve cleanly and the
remainder need minor per-area adjustments.

## Developer Mode

A toolkit of diagnostics and automated relaxation for area weights.

### When to run
- After changing the target spec (adding/removing targets)
- After a new SOI data vintage (shares changed)
- After significant TMD data changes

### How it works

For each area, developer mode tries a relaxation cascade:

1. **Level 0:** Solve with default parameters
2. **Level 1:** Drop unreachable targets (automatic)
3. **Level 2:** Reduce slack penalties on problematic constraints
4. **Level 3:** Drop specific targets identified by LP feasibility
5. **Level 4:** Raise multiplier cap (50x → 100x)
6. **Level 5:** Raise constraint tolerance (0.5% → 1.0%)

Most areas solve at level 0.  A handful of extreme areas (e.g., NY-12
/ Manhattan with its extreme high-income profile) need level 3.

### Usage

```bash
# LP feasibility check only (fast diagnostic):
python -m tmd.areas.developer_mode --scope cds --lp-only --workers 16

# Full relaxation cascade:
python -m tmd.areas.developer_mode --scope cds --workers 16

# Debug a single area:
python -m tmd.areas.developer_mode --scope NY12 --verbose
```

### Output
- **Override YAML:** `prepare/recipes/cd_solver_overrides.yaml` —
  committed to repo, read by production solver
- **Developer report:** `weights/cds/developer_report.txt` —
  per-area relaxation details

### Override file format

```yaml
_defaults:
  multiplier_max: 50
  constraint_tol: 0.005

ny12:
  drop_targets:
    - "c00100/cnt=1/scope=1/agi=[500000.0,9e+99)/fs=0"
    - "e26270/cnt=0/scope=1/agi=[100000.0,200000.0)/fs=0"
```

The production solver reads this file and applies per-area
customizations automatically.  No manual tuning needed.

## Updating for a New Year

### New SOI data vintage

1. **Get the data:** Download SOI CSV files for the new year.
   Place in `prepare/data/soi_states/` or `prepare/data/soi_cds/`.

2. **Update constants:** Add the new year's CSV filename to
   `SOI_STATE_CSV_PATTERNS` or `SOI_CD_CSV_PATTERNS` in `constants.py`.
   Check if AGI stubs changed (rare but possible).

3. **Recompute shares:**
   ```bash
   python -m tmd.areas.prepare_shares --scope states --year 2023
   python -m tmd.areas.prepare_shares --scope cds --year 2023
   ```

4. **Regenerate targets:**
   ```bash
   python -m tmd.areas.prepare_targets --scope cds
   ```

5. **Run developer mode:**
   ```bash
   python -m tmd.areas.developer_mode --scope cds --workers 16
   ```

6. **Solve weights:**
   ```bash
   python -m tmd.areas.solve_weights --scope cds --workers 16
   ```

7. **Quality check:**
   ```bash
   python -m tmd.areas.quality_report --scope cds --output
   ```

### New TMD rebuild (same SOI year)

Only steps 4-7 needed — shares don't change.

### Adding a new target variable

1. Check if SOI has the variable (or a proxy) at the right
   geographic level.

2. Add the mapping to `EXTENDED_SHARING_MAPPINGS` in `prepare_shares.py`.

3. Recompute shares: `python -m tmd.areas.prepare_shares --scope cds`

4. Add a row to the target spec CSV.

5. Regenerate targets and run developer mode to check feasibility.

### Adding a new area type (e.g., counties)

1. Write SOI data ingestion module (like `soi_cd_data.py`).

2. Define AGI cuts and area type in `constants.py`.

3. Create a target spec and shares file.

4. Add scope handling to `solve_weights.py` and `quality_report.py`.

5. Start with a very conservative recipe — counties as small as 40
   returns will need far fewer targets than CDs.  Use tiered recipes
   by county size.

## File Locations

```
tmd/areas/
├── prepare/
│   ├── constants.py          # AGI cuts, ALL_SHARING_MAPPINGS, AreaType
│   ├── recipes/
│   │   ├── cd_target_spec.csv      # CD recipe (92 targets)
│   │   ├── state_target_spec.csv   # State recipe (179 targets)
│   │   ├── cd_solver_overrides.yaml # Per-area solver params
│   │   ├── cds.json                # [legacy] JSON recipe
│   │   └── states.json             # [legacy] JSON recipe
│   ├── data/
│   │   ├── soi_states/             # Raw SOI state CSVs
│   │   ├── soi_cds/                # Raw SOI CD CSVs
│   │   ├── cds_shares.csv          # Pre-computed CD shares
│   │   └── states_shares.csv       # Pre-computed state shares
│   ├── target_sharing.py     # Share computation, TMD national sums
│   ├── target_file_writer.py # [legacy] Recipe-based target writing
│   ├── soi_state_data.py     # State SOI data ingestion
│   ├── soi_cd_data.py        # CD SOI data + crosswalk
│   └── extended_targets.py   # State extended targets (Census, credits)
├── prepare_targets.py        # Target file generation (spec-based)
├── prepare_shares.py         # Share pre-computation
├── developer_mode.py         # Auto-relaxation cascade
├── solve_weights.py          # QP batch solver
├── create_area_weights.py    # QP solver core (Clarabel)
├── quality_report.py         # Cross-area quality diagnostics
├── solver_overrides.py       # Per-area override management
├── targets/
│   ├── states/               # Per-state target CSVs
│   └── cds/                  # Per-CD target CSVs
└── weights/
    ├── states/               # Per-state weight files + logs
    └── cds/                  # Per-CD weight files + logs
```

## Quality Report

The quality report (`python -m tmd.areas.quality_report --scope cds`)
provides:

- **Target accuracy:** Per-area hit rates, violation details
- **Weight distortion:** Multiplier distribution (how far weights moved from population-proportional)
- **Weight distribution by AGI stub:** National vs sum-of-areas returns and AGI per bin
- **Weight exhaustion:** Whether records are over/under-used across areas
- **Cross-area aggregation:** Sum-of-areas vs national for key variables
- **Bystander analysis:** Distortion of untargeted variables (both aggregate and per-bin)

Use `--output` to auto-save to file.  For CDs/counties, only the top
20 most-distorted areas are shown in the per-area table.
