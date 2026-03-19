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

## Pipeline Modules

| Module | Purpose |
|--------|---------|
| `prepare_targets.py` | CLI entry point |
| `prepare/soi_state_data.py` | SOI state CSV ingestion |
| `prepare/target_sharing.py` | TMD × SOI share computation |
| `prepare/target_file_writer.py` | Recipe expansion, CSV output |
| `prepare/extended_targets.py` | Census/SOI extended targets |
| `prepare/constants.py` | AGI bins, mappings, metadata |
| `prepare/census_population.py` | State population data |
