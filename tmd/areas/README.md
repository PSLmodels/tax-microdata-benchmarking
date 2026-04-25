# Sub-national area weights

This folder contains everything needed to produce **per-area weight
files** that adapt the national TMD microdata to a specific state or
Congressional district.

The national TMD file has one record per tax-filing unit and one
weight per record (`s006`).  Those weights make weighted sums match
national totals.  An area weight file replaces `s006` with a set of
**area-specific** weights, so weighted sums of the same records match
state-level (or CD-level) totals from IRS Statistics of Income (SOI).
The records do not change; only the weights do.

This README is the user-facing entry point.  For deeper background
on how the targets are constructed and how the optimizer works, see:

- [AREA_WEIGHTING_GUIDE.md](AREA_WEIGHTING_GUIDE.md) — concepts,
  target architecture, recipe contents, file locations.
- [AREA_WEIGHTING_LESSONS.md](AREA_WEIGHTING_LESSONS.md) — practical
  lessons from production runs (parameter choices, weight
  exhaustion, bystander effects, SALT targeting).
- [weights/README.md](weights/README.md) — weight-file format and
  how to use the files with Tax-Calculator.
- [targets/README.md](targets/README.md) — targets-file format.

## What you can do here

| Task | Section |
|---|---|
| Build all 51 state weight files | [Quickstart](#quickstart) |
| Build all 436 CD weight files (118th or 119th Congress) | [Quickstart](#quickstart) |
| Re-solve weights for one area or a few | [Single area](#building-weights-for-a-single-area) |
| Decide which stage to rerun after an input change | [When to rerun which stage](#when-to-rerun-which-stage) |
| Check overall quality across many areas | [Multi-area quality report](#multi-area-quality-report) |
| Inspect a single area in detail | [Individual-area detail report](#individual-area-detail-report) |
| Diagnose why a specific area is hard to fit | [Single-area diagnostics](#single-area-diagnostics) |

## Prerequisites

The national TMD files must already exist in `tmd/storage/output/`.
If they do not, run `make data` from the repository root first.

## Quickstart

The full pipeline for each scope (shares → targets → weights → tests
and quality report) is wired into `tmd/areas/Makefile`.  From the
repository root:

```bash
# All 51 states (50 states + DC), 8 parallel workers:
make -C tmd/areas states WORKERS=8

# All 436 Congressional districts on 118th Congress boundaries:
make -C tmd/areas cds-118 WORKERS=16

# All 436 CDs on 119th Congress boundaries (different district maps
# in AL, GA, LA, NY, NC; identical elsewhere):
make -C tmd/areas cds-119 WORKERS=16
```

State runs take a few minutes; CD runs take longer because there are
more areas.  These Make targets always rebuild every stage (shares,
targets, weights, tests, quality report) — once you know what
changed upstream, the single-stage commands in
[When to rerun which stage](#when-to-rerun-which-stage) are
faster.

For Congressional districts the choice of Congressional session is required —
there is no default.  118th and 119th both have 436 districts (435
voting + DC); only the geographies differ.

## Outputs

After a successful run you get three kinds of files:

- **Targets** — one CSV per area in `tmd/areas/targets/states/` or
  `tmd/areas/targets/cds_118/` (or `cds_119/`).  Each row is one
  weighted sum the area is asked to match.  See
  [targets/README.md](targets/README.md) for the file format.
- **Weights** — one gzipped CSV per area in `tmd/areas/weights/...`,
  named `<area>_tmd_weights.csv.gz`.  One row per TMD record, with
  thirteen weight columns (`WT2022` through `WT2034`, one per year).
  See [weights/README.md](weights/README.md) for how to use these
  with Tax-Calculator.
- **Solver log** — `<area>.log` next to each weight file, with the
  population share used, target accuracy statistics, the weight
  multiplier distribution, and per-target slack.

## Building weights for a single area

You can re-solve any single state or single CD by passing its area
code to `--scope`.  This is much faster than running the full batch
and is the right approach when you are tuning solver parameters,
testing a recipe change, or investigating one area.

State codes are two-letter postal codes (lowercase or uppercase).
CD codes are state code + two-digit district, e.g., `MN01`, `NY12`.

```bash
# Re-solve one state, using existing target and shares files:
python -m tmd.areas.solve_weights --scope CA

# A few states at once:
python -m tmd.areas.solve_weights --scope CA,NY,TX --workers 4

# A single Congressional district (Congress is required):
python -m tmd.areas.solve_weights --scope MN01 --congress 118
```

These commands assume the area's targets file already exists.  If
it does not, or if any of the inputs that feed into targets has
changed, see the next section.

## When to rerun which stage

A target is constructed as

    target = TMD national sum × geographic share

so a target file goes stale only when one of those inputs changes.
This table shows what to rerun when something upstream changes:

| What changed | Rerun |
|---|---|
| Solver parameter only (e.g., `multiplier_max`) | `solve_weights` |
| Recipe / target spec | `prepare_targets`, then `solve_weights` |
| TMD national file (`make data` was rerun) | `prepare_targets`, then `solve_weights` |
| New SOI vintage, or Census data update | `prepare_shares`, then `prepare_targets`, then `solve_weights` |
| Adding a previously-unsolved area | `solve_weights --scope <area>` |

`make -C tmd/areas states` (and `cds-118` / `cds-119`) always reruns
every stage; that is the simplest and safest entry point if you are
unsure what is stale.  The single-stage commands are the right tool
once you know which input changed.

## Quality reports

Two reports help you judge whether the area weights are good:

- The **multi-area quality report** is a roll-up across many areas
  (all states, all CDs, or a list).  Use it to see overall hit
  rates, weight distortion, and bystander effects.
- The **individual-area detail report** is the same report restricted
  to a single area or a small list, with full per-target detail.
  Use it to dig into one place that looked unusual in the roll-up.

Both modes are produced by the same command,
`tmd.areas.quality_report`.  The only difference is what you pass
to `--scope`.

### Multi-area quality report

```bash
# All 51 states:
python -m tmd.areas.quality_report --scope states

# All CDs on 118th Congress boundaries (--congress is required):
python -m tmd.areas.quality_report --scope cds --congress 118

# Save to a file alongside the weights:
python -m tmd.areas.quality_report --scope states --output
python -m tmd.areas.quality_report --scope cds --congress 119 --output
```

The roll-up covers:

- **Solve status** — how many areas solved cleanly, how many used
  per-area overrides, how many failed.
- **Target accuracy** — per-area hit rates and the largest
  violations.
- **Weight distortion** — distribution of weight multipliers
  (how far weights moved from population-proportional).
- **Weight exhaustion** — which records get oversubscribed across
  areas.
- **Cross-area aggregation** — sum-of-areas vs national for key
  variables, so you can confirm the area weights are consistent
  with the national totals.
- **Bystander analysis** — variables that are not targeted but are
  still pushed around by the optimizer; >2% distortion is flagged.

A multi-area run also writes one machine-readable CSV alongside the
weight files:

- **`quality_report_per_area.csv`** — one row per area, with columns
  for record counts, weighted returns, weighted AGI, average AGI,
  the multiplier distribution (median, p95, max, share zero), and
  two summary diagnostics (`unusualness` and effective sample size
  `ess`).  This file covers **every** area in the run, so it is the
  right input when you want to sort, filter, or plot results across
  the full set.

For CDs, the per-area table inside the text report shows only the
20 most-distorted areas to keep the report a manageable length; the
CSV does not truncate.

### Individual-area detail report

To zoom in on one area, pass its code (or a short list) to
`--scope`.  The report layout is the same; the per-area sections
just have one entry.

```bash
# A single state:
python -m tmd.areas.quality_report --scope CA

# A short list of states:
python -m tmd.areas.quality_report --scope CA,NY,TX

# A single CD or a list of CDs (Congress is required):
python -m tmd.areas.quality_report --scope MN01 --congress 118
python -m tmd.areas.quality_report --scope MN01,MN02 --congress 118

# Send the report to a named file:
python -m tmd.areas.quality_report --scope CA -o ca_report.txt
```

### Single-area diagnostics

For a deeper look at a single area, the developer-tools module
provides two purpose-built diagnostics that are also useful to
analysts:

```bash
# Why is this area hard to fit?  Per-target gap from a
# population-proportionate share, sorted from largest to smallest:
python -m tmd.areas.developer_tools --difficulty AL01 --congress 118

# Which constraints are most expensive?  Shadow prices from the
# solved optimization, identifying targets that pull weights hard:
python -m tmd.areas.developer_tools --dual NY12 --congress 118
```

The difficulty table is the single most useful diagnostic for
understanding why an area gets large weight movement.  A target
with a 5% gap from the proportional share is essentially free; a
target with a 100% gap will move many weights.

## Map of files in this folder

```
tmd/areas/
├── README.md                       (this file)
├── AREA_WEIGHTING_GUIDE.md         concepts and target architecture
├── AREA_WEIGHTING_LESSONS.md       lessons from production runs
├── Makefile                        states / cds-118 / cds-119 pipelines
├── prepare_shares.py               compute SOI geographic shares
├── prepare_targets.py              produce per-area target CSVs
├── solve_weights.py                solve weights (parallel batch)
├── create_area_weights.py          single-area QP solver core
├── quality_report.py               multi- and single-area reports
├── developer_tools.py              difficulty, dual, relaxation cascade
├── batch_weights.py                parallel-runner support
├── solver_overrides.py             read per-area override YAML
├── sweep_params.py                 parameter grid-search utility
├── make_all.py                     end-to-end driver used in CI
├── prepare/                        shares + targets building blocks
│   ├── data/                       SOI inputs and CD crosswalks
│   ├── recipes/                    target specs
│   └── (modules)
├── targets/                        per-area target CSVs (output)
└── weights/                        per-area weight files (output)
```

For the sub-folder details see the respective READMEs (linked at the
top of this file).
