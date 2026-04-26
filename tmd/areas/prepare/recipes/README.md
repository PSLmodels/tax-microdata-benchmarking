# Target recipes

A target recipe lists the constraints the area-weight optimizer
must satisfy.  The production pipeline reads the recipe as a flat
CSV (one row per target) — what you see in the file is what gets
solved.

## Files in this directory

| File | Status | Read by |
|---|---|---|
| `state_target_spec.csv` | **current** | `tmd.areas.prepare_targets` (state pipeline) |
| `cd_target_spec.csv` | **current** | `tmd.areas.prepare_targets` (CD pipeline, both 118 and 119) |
| `state_variable_mapping.csv` | supports legacy JSON path (see below) | `tests/test_prepare_targets.py` |
| `cd_solver_overrides.yaml` | **current** | `tmd.areas.solve_weights` and `tmd.areas.developer_tools` |
| `states.json` | **legacy** — retained for test coverage of the JSON path | `tests/test_prepare_targets.py` |

The CD recipe is shared across the 118th and 119th Congresses.
Only the area codes and the underlying SOI shares differ between
the two; the targeting recipe itself is identical.

## CSV target spec format

```csv
varname,count,scope,fstatus,agilo,agihi,description
XTOT,0,0,0,-9e+99,9e+99,Population amount all bins
c00100,0,1,0,-9e+99,1.0,AGI amount <$0K
c00100,0,1,0,1.0,10000.0,AGI amount $0K-$10K
...
eitc,0,1,0,-9e+99,9e+99,EITC amount all bins
```

| Column | Meaning |
|---|---|
| `varname` | TMD variable name (Tax-Calculator input or output column) |
| `count` | 0 = dollar amount, 1 = all returns, 2 = nonzero count, 3 = positive count, 4 = negative count |
| `scope` | 0 = all records (XTOT only), 1 = PUF-derived records, 2 = CPS-derived records |
| `fstatus` | Filing status: 0 = all, 1 = single, 2 = MFJ, 4 = HoH |
| `agilo`, `agihi` | AGI bin: lower bound (inclusive) and upper bound (exclusive). `-9e+99` / `9e+99` mean "no bound" |
| `description` | Human-readable label; ignored by the pipeline |

To add a target, add a row.  To remove one, delete the row.  No
exclude lists, no indirection.  See [tmd/areas/AREA_WEIGHTING_GUIDE.md](../../AREA_WEIGHTING_GUIDE.md)
for guidance on which targets are safe to add and which carry
solver risk.

For variables that need a SOI proxy (because no direct SOI
counterpart exists at the right geographic level), the proxy is
defined in `ALL_SHARING_MAPPINGS` in
[`../constants.py`](../constants.py) or in `EXTENDED_SHARING_MAPPINGS`
in [`../../prepare_shares.py`](../../prepare_shares.py).

## Solver overrides (CDs only)

`cd_solver_overrides.yaml` records per-CD solver customizations
needed for the small fraction of districts (~3% at last count) that
cannot be solved with the default parameters.  The schema:

```yaml
_defaults:
  multiplier_max: 50
  constraint_tol: 0.005

ny12:
  drop_targets:
    - "c00100/cnt=1/scope=1/agi=[500000.0,9e+99)/fs=0"
    - "e26270/cnt=0/scope=1/agi=[100000.0,200000.0)/fs=0"
```

`tmd.areas.solve_weights` reads this file automatically.  Per-area
adjustments are produced by `tmd.areas.developer_tools` (the
relaxation cascade) and committed to this file.

State runs do not currently use a solver-overrides file.

## Legacy JSON recipe

`states.json` is the older recipe format that paired a JSON file
with `state_variable_mapping.csv`.  It is no longer used by the
CLI (`tmd.areas.prepare_targets` always uses the CSV spec above)
but is kept on disk so that `tests/test_prepare_targets.py` can
exercise the JSON code path in
[`../target_file_writer.py`](../target_file_writer.py).  New work
should use the CSV spec.
