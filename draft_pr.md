# PR: Replace PyTorch L-BFGS reweighting with Clarabel QP solver

Addresses #NNN (replace with issue number after creating the issue).

## Summary

- Add Clarabel constrained QP solver as default reweighting method, replacing the PyTorch L-BFGS penalty-based approach
- All ~800 SOI targets now satisfied within +/-0.5% (previously, some targets like unemployment compensation were ~8% off)
- Cross-machine deterministic: identical weight fingerprints verified on two different machines
- Solve time ~17 seconds (vs ~7 min scipy L-BFGS-B, ~3 min PyTorch L-BFGS)

## What changed

| File | Change |
|------|--------|
| `tmd/utils/reweight_clarabel.py` | New: Clarabel QP solver with elastic slack, constraint scaling, verbose diagnostics |
| `tmd/datasets/tmd.py` | Default solver selection: Clarabel (with `PYTORCH_REWEIGHT=1` and `SCIPY_REWEIGHT=1` overrides) |
| `tmd/imputation_assumptions.py` | New parameters: `CLARABEL_CONSTRAINT_TOL`, `CLARABEL_SLACK_PENALTY`, `CLARABEL_MAX_ITER` |
| `tmd/utils/reweight.py` | Added scipy L-BFGS-B solver (`reweight_lbfgsb`), kept as alternative; added `build_loss_matrix` and `_drop_impossible_targets` shared utilities |
| `setup.py` | Added `clarabel` dependency |
| `Makefile` | Added `clarabel` to pip install |
| `tests/test_reweight_clarabel.py` | New: unit tests for constraint bounds and solver selection |
| `tests/test_weights.py` | Updated to 10-stat fingerprint matching Clarabel output |
| `tests/expected_tax_expenditures` | Updated expected values for Clarabel weights |
| `tests/test_tax_expenditures.py` | Tightened tolerance: atol 0.1 to 0.05, rtol to default |
| `tests/test_imputed_variables.py` | Updated OBBBA expected values for Clarabel weights |

## Formulation

The previous approach combined target accuracy and weight smoothness in a single penalty objective. The optimizer minimized the sum, allowing individual targets to drift if that reduced total loss.

The new approach separates concerns:
- **Objective**: minimize weight distortion `sum((x_i - 1)^2)`
- **Constraints**: every SOI target within +/-0.5% of its value (hard constraints, not penalties)
- **Elastic slacks**: if a constraint is geometrically impossible, the solver reports it rather than failing

With ~225K weight multipliers and ~800 constraints, the feasible region is large and the solver finds a solution satisfying all targets.

## Test plan

- [x] `make data` completes successfully
- [x] All 51 tests pass (`make test`)
- [x] `make format` and `make lint` clean
- [x] Cross-machine reproducibility: identical results on two different machines
- [ ] Reviewer runs `make clean && make data` and confirms all tests pass

Generated with [Claude Code](https://claude.com/claude-code)
