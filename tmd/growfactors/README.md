# Growth Factors Pipeline

Produces `tmd/storage/input/taxdata<VINTAGE>_growfactors.csv` — the raw
Tax-Calculator-style growth factors file — from Census population, CBO
baseline, IRS return projection, SOI estimate, and benefit program source
data.

This logic was migrated from the TaxData repository (`puf_stage1/`) so
that TMD can generate its own growth factors rather than vendoring a file
generated elsewhere.

## For most users

You don't need to run this pipeline. The repo ships a vetted growth
factors file and `make data` uses it directly, via
`tmd/create_raw_growth_factors.py`.

## For maintainers

Run the pipeline when refreshing the source data for a new vintage. The
stages are wired together in `tmd/growfactors/Makefile`:

```bash
# Run both stages and the pipeline tests:
make -C tmd/growfactors all

# Individual stages (stage 2 depends on stage 1):
make -C tmd/growfactors stage1    # source data -> data/Stage_I_factors.csv
make -C tmd/growfactors factors   # -> tmd/storage/input/taxdata26_growfactors.csv
make -C tmd/growfactors test      # pipeline tests only

# Produce a new vintage rather than rebuilding the current one:
make -C tmd/growfactors factors VINTAGE=27

# Verify the full build still passes:
make clean && make data
```

See [docs/annual_update.md](docs/annual_update.md) for the annual
source-data refresh procedure.

## Pipeline stages

```
data/NP2014_D1.csv, NC-EST2014-AGESEX-RES.csv, US-EST00INT-ALLDATA.csv
data/CBO_baseline.csv, IRS_return_projection.csv, SOI_estimates.csv
data/benefitprograms.csv
    |  build_stage1_factors.py
data/Stage_I_factors.csv               [level indexes, normalized to SYR=2011]
    |  build_growfactors.py  (+ data/benefit_growth_rates.csv)
tmd/storage/input/taxdata26_growfactors.csv
    |  tmd/create_raw_growth_factors.py                    [`make data`]
tmd/storage/output/growfactors.csv
    |  tmd/create_taxcalc_growth_factors.py                [`make data`]
tmd/storage/output/tmd_growfactors.csv
```

**Stage 1** (`build_stage1_factors.py`) extrapolates SOI aggregates
forward from the last available SOI year using growth rates from Census
population, the CBO baseline, and the IRS return projection, then
normalizes everything to a level index with `SYR` (2011) equal to one.

**Stage 2** (`build_growfactors.py`) converts the aggregate indexes to a
per-capita basis, converts the level indexes into "one plus annual
proportional change" growth factors, appends the benefit growth factors
derived from `data/benefit_growth_rates.csv`, and drops the variables
Tax-Calculator does not use.

## Key files

| Path | Role |
| --- | --- |
| `data/CBO_baseline.csv` | CBO economic projections; refreshed by `update/update_cbo.py` |
| `data/SOI_estimates.csv` | SOI aggregates through `SOI_YR`; refreshed by `update/update_soi.py` |
| `data/IRS_return_projection.csv` | IRS Pub 6187 return counts; updated by hand |
| `data/benefit_growth_rates.csv` | average benefit amounts by program and year |
| `data/benefitprograms.csv` | benefit program costs; vendored from TaxData `taxdata/cps/` |
| `data/NP2014_D1.csv` | Census 2014 national population projection |
| `data/NC-EST2014-AGESEX-RES.csv` | Census postcensal estimates, 2010-2014 |
| `data/US-EST00INT-ALLDATA.csv` | Census intercensal estimates, 2000-2010 |
| `data/Stage_I_factors.csv` | committed intermediate written by stage 1 |

## Source data refresh scripts

The scripts in `update/` fetch from live CBO, BLS, SSA, and IRS websites.
They are **never** run by `make data`, by `make -C tmd/growfactors all`,
or by any test. They need extra packages:

```bash
pip install -e ".[growfactors-update]"
```

See [docs/annual_update.md](docs/annual_update.md).

## Differences from the TaxData original

- The code that wrote `Stage_II_targets.csv` has been removed. It served
  only the TaxData stage2 weighting logic; TMD reweights against
  `tmd/storage/input/soi.csv` instead. The `aggregates` DataFrame in
  `build_stage1_factors.py` was named `Stage_II_targets` in TaxData and
  is still required, because the stage 1 factors are computed from it.
- Benefit growth factors are now padded with ones for years beyond the
  last year in `benefit_growth_rates.csv`. TaxData's current code leaves
  those years undefined, which silently writes empty values for 2032
  through 2036.
- Two chained assignments that are no-ops under pandas copy-on-write
  have been rewritten (see the NOTE comments in the two build scripts).
- `update/update_soi.py` no longer updates the CPS copy of
  `SOI_estimates.csv`, which has no TMD counterpart.
