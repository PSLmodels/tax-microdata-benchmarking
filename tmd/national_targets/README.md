# National Targets Pipeline

Produces `soi.csv` — the available national reweighting targets — from IRS SOI
Excel files. `reweight.py` reads `soi.csv` and selects which targets to use.

## For most users

You don't need to run this pipeline. The repo ships a vetted `soi.csv` and
`make data` uses it directly.

## For maintainers

The pipeline is how `soi.csv` gets created and updated. Run it when adding
a new tax year or correcting an extraction error. The stages are wired
together in `tmd/national_targets/Makefile`:

```bash
# Run the full pipeline (all three stages) and the pipeline tests:
make -C tmd/national_targets all

# Individual stages (each depends on the prior stage):
make -C tmd/national_targets extract   # stage 1: IRS Excel -> per-table CSVs
make -C tmd/national_targets build     # stages 1 + 2: -> irs_aggregate_values.csv
make -C tmd/national_targets soi       # stages 1 + 2 + 3: -> soi.csv
make -C tmd/national_targets test      # pipeline tests only

# Stage 1 skips per-table CSVs that already exist; force re-extraction with:
make -C tmd/national_targets extract OVERWRITE=1

# Verify the full build still passes:
make clean && make data
```

The manual command sequence below is equivalent to `make -C tmd/national_targets all`
and is retained for reference:

```bash
python -m tmd.national_targets.extract_irs_to_csv --overwrite
python -m tmd.national_targets.build_targets
python -m tmd.national_targets.potential_targets_to_soi
python -m pytest tests/national_targets_pipeline -v
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
