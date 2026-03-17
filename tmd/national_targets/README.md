# National Targets Pipeline

Produces `soi.csv` — the available national reweighting targets — from IRS SOI
Excel files. `reweight.py` reads `soi.csv` and selects which targets to use.

## For most users

You don't need to run this pipeline. The repo ships a vetted `soi.csv` and
`make data` uses it directly.

## For maintainers

The pipeline is how `soi.csv` gets created and updated. Run it when adding
a new tax year or correcting an extraction error:

```bash
python -m tmd.national_targets.extract_irs_to_csv --overwrite
python -m tmd.national_targets.build_targets
python -m tmd.national_targets.potential_targets_to_soi

# Pipeline-specific tests (not part of make data)
python -m pytest tests/national_targets_pipeline -v

# Verify the full build still passes
make clean && make data
```

See [docs/adding_a_new_year.md](docs/adding_a_new_year.md) for step-by-step
instructions on adding a new tax year.

## Pipeline stages

```
IRS Excel files (.xls)
    ↓  extract_irs_to_csv.py        [one-time per year]
data/extracted/{year}/{table}.csv
    ↓  build_targets.py
data/irs_aggregate_values.csv       [all years, all tables, not deduplicated]
    ↓  potential_targets_to_soi.py
tmd/storage/input/soi.csv           [deduplicated, ready for reweight.py]
```

## Key files

| File | Purpose |
|------|---------|
| `extract_irs_to_csv.py` | Stage 1: IRS Excel → per-table CSVs |
| `build_targets.py` | Stage 2: per-table CSVs → `irs_aggregate_values.csv` |
| `potential_targets_to_soi.py` | Stage 3: → `soi.csv` |
| `config/table_layouts.py` | Column definitions for all IRS tables and years |
| `data/irs_aggregate_values.csv` | Complete audit trail (not deduplicated) |
| `data/irs_to_puf_map.json` | IRS variable → PUF variable mapping |
| `docs/adding_a_new_year.md` | Maintainer guide for new tax years |
