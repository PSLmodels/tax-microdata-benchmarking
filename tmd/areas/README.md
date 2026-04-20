# Area Weighting

Generates sub-national area weight files from national PUF-based
microdata. Area weights let Tax-Calculator produce state-level
estimates using national input data and growfactors.

## Preparing State Targets

```bash
# All 51 states (50 states + DC), ~4 seconds:
python -m tmd.areas.prepare_targets --scope states

# Specific states:
python -m tmd.areas.prepare_targets --scope MN,CA,TX

# Use 2021 SOI shares:
python -m tmd.areas.prepare_targets --scope states --year 2021
```

**Prerequisite**: TMD national data must exist (`make tmd_files`).

Output: one CSV per state in `tmd/areas/targets/states/`.

## How Targets Work

Each target constrains a weighted sum to match a state-level value.
Targets combine two data sources:

- **TMD national totals** provide the level (weighted sums from the
  national PUF microdata).
- **IRS SOI state data** provides the geographic distribution (each
  state's share of the US total, by AGI bin and filing status).

Formula: `state_target = TMD_national × (state_SOI / US_SOI)`

Extended targets also use **Census state/local finance data** for
SALT distribution and **SOI credit data** for EITC/CTC.

## Target Composition (~178 per state)

**Base targets** (recipe-driven, all 10 AGI bins):
AGI amounts, total return counts, return counts by filing status,
wages (amount + nonzero count), taxable interest, pensions, Social
Security, SALT deduction, partnership/S-corp income.

**Extended targets** (SOI/Census-shared, high-income bins only):
Taxable pensions, taxable Social Security, IRA distributions, net
capital gains, dividends, business income, mortgage interest,
charitable contributions, SALT by source (Census), EITC, CTC.

Filing-status count targets are excluded from the $1M+ AGI bin to
avoid excessive weight distortion on small cells.

## Solving for State Weights

```bash
# All 51 states, 8 parallel workers:
python -m tmd.areas.solve_weights --scope states --workers 8

# Specific states:
python -m tmd.areas.solve_weights --scope MN,CA,TX --workers 4
```

Uses the Clarabel constrained QP solver to find per-record weight
multipliers that hit each state's targets within 0.5% tolerance.

Output: weight files in `tmd/areas/weights/states/` and solver
logs alongside them.

## Quality Report

```bash
python -m tmd.areas.quality_report
python -m tmd.areas.quality_report --scope CA,NY
```

Cross-state summary: solve status, target accuracy, weight
distortion, weight exhaustion, and national aggregation checks.

## Pipeline Modules

**Target preparation** (PR 1):

| Module | Purpose |
|--------|---------|
| `prepare_targets.py` | CLI entry point |
| `prepare/soi_state_data.py` | SOI state CSV ingestion |
| `prepare/target_sharing.py` | TMD x SOI share computation |
| `prepare/target_file_writer.py` | Recipe expansion, CSV output |
| `prepare/extended_targets.py` | Census/SOI extended targets |
| `prepare/constants.py` | AGI bins, mappings, metadata |
| `prepare/census_population.py` | State population data |

**Weight solving** (PR 2):

| Module | Purpose |
|--------|---------|
| `solve_weights.py` | CLI entry point |
| `create_area_weights_clarabel.py` | Clarabel QP solver |
| `batch_weights.py` | Parallel batch runner |
| `quality_report.py` | Cross-state diagnostics |
| `sweep_params.py` | Parameter grid search utility |

## Congressional Districts (118th or 119th Congress)

The SOI CD micro-data are published on **117th Congress** boundaries
for both tax years 2021 and 2022.  The CD pipeline crosswalks those
data to either 118th or 119th Congress boundaries using a Geocorr
2022 population-weighted crosswalk.  **Both Congressional sessions
produce 436 CDs** (435 voting + DC); they differ only in the
district geography for AL, GA, LA, NY, and NC.

All CD CLI tools require an explicit `--congress 118` or
`--congress 119` flag — there is no default.

```bash
# Build shares, targets, and weights for the 118th Congress:
python -m tmd.areas.prepare_shares  --scope cds --congress 118
python -m tmd.areas.prepare_targets --scope cds --congress 118
python -m tmd.areas.solve_weights   --scope cds --congress 118 --workers 16

# Same for the 119th Congress (new AL/GA/LA/NY/NC maps):
python -m tmd.areas.prepare_shares  --scope cds --congress 119
python -m tmd.areas.prepare_targets --scope cds --congress 119
python -m tmd.areas.solve_weights   --scope cds --congress 119 --workers 16

# Or via Makefile convenience targets:
make -C tmd/areas cds-118 WORKERS=16
make -C tmd/areas cds-119 WORKERS=16
```

Outputs are written to per-Congress subdirectories:

- `tmd/areas/targets/cds_118/` and `tmd/areas/targets/cds_119/`
- `tmd/areas/weights/cds_118/` and `tmd/areas/weights/cds_119/`
- `tmd/areas/prepare/data/cds_118_shares.csv` and
  `tmd/areas/prepare/data/cds_119_shares.csv`

The targeting recipe is identical between the two Congressional
sessions — only the geographic allocation differs.

See [tmd/areas/prepare/data/README.md](prepare/data/README.md) for
details on the crosswalk files, their Geocorr settings, and
validation checks.

## Lessons Learned

See [AREA_WEIGHTING_LESSONS.md](AREA_WEIGHTING_LESSONS.md) for
practical guidance on parameter tuning, weight exhaustion, SALT
targeting, dual variable analysis, and recommendations for
extending to Congressional districts.
