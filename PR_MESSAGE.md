# PR: Improve area weight solver: robustness, memory, and testing

> **This is PR 1 of 4** adding congressional district (CD) weighting to TMD. The
> PRs are stacked and should be reviewed/merged in order. None of the PRs changes
> national code or results. There are changes to Makefile to exclude certain new
> tests that relate to areas and may be best run on an as-needed basis.
>
> 1. **Solver robustness** (this PR) — This improves the area weight solution
>    infrastructure so that it is more flexible and efficient. This provides
>    modest benefits for state target creation and weight solution, and will
>    provide substantial benefits when we prepare targets and solve weights for
>    436 Congressional Districts (including DC). The changes include memory
>    reduction, feasibility checks, per-constraint penalties, and new tests.
> 2. **Spec-based target pipeline** — CSV-driven target specification, SOI CD
>    data ingestion, geographic shares. Both states and CDs route through the
>    new spec pipeline. State targets are regenerated via the new pipeline
>    (verified identical results via fingerprint test).
> 3. **Quality report enhancements** — CD-aware reporting, violation summaries,
>    human-readable labels.
> 4. **Congressional district pipeline** — CD solver integration, developer
>    mode, override YAML, 436-CD batch solving.

## Summary of this PR

Improve the area weight QP solver with robustness enhancements, major memory
reductions, and new test coverage. These changes benefit states and, in the
future, congressional districts.

**Solver robustness (3 improvements):**

- **Range-based target filtering:** `_drop_impossible_targets()` now checks
  whether each target is achievable within multiplier bounds, not just whether
  the constraint matrix row is all zeros. Catches geometrically unreachable
  targets that the old check missed.
- **LP feasibility pre-check:** New `_check_feasibility()` runs a fast linear
  program (scipy HiGHS) before the QP to identify which constraints will need
  slack. Runs on every area solve (not just development). Diagnostic only —
  logs which constraints are tight but does not change solutions.
- **Per-constraint slack penalties:** New `_assign_slack_penalties()` gives
  reduced penalty (1e3 vs 1e6) to inherently noisy targets: e02400/e00300/e26270
  amounts in low-AGI bins, and filing-status counts in the lowest bins. The
  solver relaxes these targets in preference to distorting weights globally to
  meet targets.

**Memory reductions (3 changes, net -36% vs master):**

The PR reduces memory usage, especially per-worker usage, to make it practical
to use more workers on multi-processor systems:

- Build constraint matrix B directly in sparse COO format, eliminating two dense
  intermediates (~620 MB saved per worker).
- Use sparse matrices in LP feasibility check (~1.2 GB saved per worker).
- Trim unused TMD DataFrame columns (109 to ~30) and preload TMD in the parent
  process before forking workers (shared via copy-on-write).

Peak memory per worker: **1,244 MB (master) reduced to 798 MB (this PR)**. With 16
workers: ~20 GB reduced to ~13 GB.

**New infrastructure:**

- `solver_overrides.py`: YAML-based per-area solver parameter management.
  Provides infrastructure for customizing solver settings (tolerance, multiplier
  bounds, etc.) per area. No override files are included in this PR; actual
  per-area overrides will be generated and committed in a later PR when the
  congressional district pipeline is added.

**New tests:**

- `test_state_weight_results.py`: post-solve validation of state weight files
  (existence, nonnegativity, no NaN, correct columns, target accuracy within
  tolerance). Run as part of the test suite if weight files exist.
- `test_fingerprint.py`: on-demand reproducibility test. Rounds weights to
  integers, sums per area, and hashes. Detects any change in results across runs
  or machines. Not part of `make test`; run manually with `pytest
  tests/test_fingerprint.py -v`.

## Impact on state weights

State weight results will change numerically due to per-constraint slack
penalties. This is expected and is an improvement — noisy low-AGI targets that
previously forced weight distortion across all records are now relaxed
preferentially. The constraint tolerance (0.5%) and multiplier bounds (0-25x)
are unchanged.

## Files changed (9 files, +998 / -61)

| File                                 | Change                                                                                         |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `tmd/areas/create_area_weights.py`   | Sparse matrix construction, enhanced feasibility check, LP pre-check, per-constraint penalties |
| `tmd/areas/batch_weights.py`         | Parent-process TMD preloading, override support, per-constraint penalties wiring               |
| `tmd/areas/solver_overrides.py`      | NEW — YAML-based per-area override management                                                  |
| `tmd/areas/solve_weights.py`         | Minor CLI scope fix                                                                            |
| `tmd/areas/quality_report.py`        | Fix --scope states CLI parsing                                                                 |
| `tests/test_state_weight_results.py` | NEW — post-solve state weight validation                                                       |
| `tests/test_fingerprint.py`          | NEW — on-demand reproducibility test                                                           |
| `tests/conftest.py`                  | Add --update-fingerprint CLI option                                                            |
| `Makefile`                           | Exclude fingerprint test from `make test`                                                      |

## Test plan

```bash
make format                                                    # no changes
make lint                                                      # passes clean
make clean && make data                                        # build TMD + run all tests
python -m tmd.areas.prepare_targets --scope states             # generate state target files
python -m pytest tests/test_prepare_targets.py -v              # verify targets
python -m tmd.areas.solve_weights --scope states --workers 16  # solve state weights
python -m pytest tests/test_state_weight_results.py -v         # verify weights
python -m tmd.areas.quality_report --scope states              # quality report
pytest tests/test_fingerprint.py -v --update-fingerprint       # save fingerprint
```

## Reproducibility

Verified: 8-worker and 16-worker solves produce identical fingerprints (hash
`8b36ae1c2ee0c384`, integer weight sums match per area exactly).

Prepared by @donboyd5 and [Claude Code](https://claude.com/claude-code)
