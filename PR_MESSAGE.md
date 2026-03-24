# PR: Add spec-based target pipeline with SOI CD data ingestion

> **This is PR 2 of 4** adding congressional district (CD) weighting to TMD.
> Based on PR 1 (solver robustness). See PR 1 for the full roadmap.

## Summary

Replace the hardcoded target generation logic with a clean three-artifact
architecture that separates concerns by change frequency:

1. **Shares** (stable) — SOI geographic distribution, pre-computed from SOI
   data. Changes only with a new SOI vintage (~annually).
2. **Spec** (recipe) — flat CSV, one row per target. WYSIWYG. Changes during
   recipe tuning.
3. **Targets** (volatile) — `target = TMD_national_sum * share`. Recomputed
   whenever TMD data or the spec changes.

This PR also adds SOI congressional district data ingestion with the
117th-to-118th Congress crosswalk, and a workaround for a confirmed SOI data
bug (column A59664 in dollars instead of $1,000s).

## Architecture

```
SOI data + crosswalks -> shares (stable)        <- rarely changes
                              |
TMD data (cached_allvars) -> national sums       <- changes with TMD rebuilds
                              |
          shares * national sums = potential targets
                              |
          target spec -> select from potential    <- changes during recipe tuning
                              |
          per-area _targets.csv files
```

## Key design decisions

- **Shares use CD file's own totals as denominators** — internally consistent,
  sums to 1.0 across all CDs for each variable/bin.
- **XTOT uses Census 2020 population** from the geocorr crosswalk, matching the
  state pipeline approach.
- **117th-to-118th Congress crosswalk** properly handles MT (1 to 2 districts),
  at-large states (AK, DC, DE, ND, SD, VT, WY), and split districts.
- **SOI A59664 bug workaround** — CD file column A59664 (EITC, 3+ children) is
  in dollars instead of $1,000s. Divided by 1000 on ingestion. State file is not
  affected. Reported to SOI. @donboyd5 will report the bug to the IRS SOI unit.
- **Variable name mapping** — SOI raw names (A00100) map to TMD names (c00100)
  via `ALL_SHARING_MAPPINGS` in constants.py. Multiple TMD variables can share
  one SOI proxy (e.g., e01500 and e01700 both use SOI 01700).

## New files

| File                                              | Purpose                                                 |
| ------------------------------------------------- | ------------------------------------------------------- |
| `tmd/areas/prepare_shares.py`                     | Pre-compute SOI geographic shares for states and CDs    |
| `tmd/areas/prepare/soi_cd_data.py`                | CD SOI data reader, crosswalk, base target construction |
| `tmd/areas/prepare/recipes/cd_target_spec.csv`    | CD recipe — 107 targets per CD                          |
| `tmd/areas/prepare/recipes/state_target_spec.csv` | State recipe — 169 targets per state                    |
| `tmd/areas/prepare/data/soi_cds/22incd.csv`       | Raw 2022 SOI CD data                                    |

## Modified files

| File                                  | Change                                                            |
| ------------------------------------- | ----------------------------------------------------------------- |
| `tmd/areas/prepare/constants.py`      | Add AreaType.CD, CD_AGI_CUTS, helper functions, extended mappings |
| `tmd/areas/prepare/target_sharing.py` | Add capgains_net synthetic variable, CD share functions           |
| `tmd/areas/prepare_targets.py`        | Add `prepare_targets_from_spec()`, unified CLI routing            |
| `.gitignore`                          | Whitelist SOI CD CSV and recipe CSV files                         |

## CLI routing

Both states and CDs route through `prepare_targets_from_spec()`. The old
`prepare_state_targets()` and `prepare_cd_targets()` remain in the code but
are no longer called by the CLI.

A prerequisite step — `prepare_shares` — must be run before `prepare_targets`
to generate shares files. Shares only need regenerating when SOI data changes.

## SALT targeting approach

Available SALT (e18400 income/sales, e18500 real estate) uses **Census
state/local finance data** for geographic distribution, targeted as **all-bins
totals only**. This is a change from the old pipeline which targeted per-bin
SALT using Census totals distributed by SOI bin proportions. The new approach
is more defensible:

- SOI SALT is capped at $10K (TCJA), so per-bin SOI shares systematically
  understate available SALT for high-income filers in high-tax states
- Census measures actual tax collections — the right source for uncapped SALT
- Per-bin decomposition would require combining Census totals with
  cap-distorted SOI bin proportions — an approximation we can avoid

Deductible SALT (c18300) continues to use SOI shares per-bin — correct because
SOI directly measures what was actually deducted (after the cap).

## Impact on state weights

State targets change from 179 (PR 1, old pipeline) to 169 (this PR, new
spec pipeline). The 10-target reduction comes from replacing 12 per-bin
e18400/e18500 rows with 2 total-only rows. State weights will differ from
PR 1 accordingly. The fingerprint is updated in this PR.

## Test plan

```bash
make format                                                    # no changes
make lint                                                      # passes clean
make clean && make data                                        # build TMD + run all tests

# State pipeline (new spec-based)
python -m tmd.areas.prepare_shares --scope states              # generate states_shares.csv
python -m tmd.areas.prepare_targets --scope states             # generate 169 state targets
python -m pytest tests/test_prepare_targets.py -v              # verify targets
python -m tmd.areas.solve_weights --scope states --workers 16  # solve state weights
python -m pytest tests/test_state_weight_results.py -v         # verify weights
python -m tmd.areas.quality_report --scope states              # quality report
pytest tests/test_fingerprint.py -v --update-fingerprint       # save updated fingerprint

# CD target preparation (new spec pipeline, no weight solving in this PR)
python -m tmd.areas.prepare_shares --scope cds                 # generate cds_shares.csv
python -m tmd.areas.prepare_targets --scope cds                # generate 436 CD target files
python -m pytest tests/test_prepare_targets.py -v -k CD        # verify CD shares and targets
```

Prepared by @donboyd5 and [Claude Code](https://claude.com/claude-code)
