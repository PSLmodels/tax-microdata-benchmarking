# Replace PyTorch L-BFGS reweighting with Clarabel QP solver

## Problem

The current reweighting uses a penalty-based objective (PyTorch L-BFGS) that combines target accuracy and weight smoothness into a single sum. This has two structural problems:

1. **No per-target accuracy guarantee.** The optimizer minimizes the total penalty, so hard-to-satisfy targets can drift far from their SOI values if that reduces total loss. Unemployment compensation targets, for example, are ~8% off in the current weights.

2. **Cross-machine non-reproducibility.** Issue #400 and prior work (PRs #407, #411) improved this substantially, but PyTorch L-BFGS with `torch.clamp()` bounds still has inherent non-determinism across different hardware (flat gradient regions from clamping, BLAS differences, float32 reduction order).

Issue #413 proposed switching to scipy L-BFGS-B, which solved the reproducibility problem by using projected-gradient bounds and analytical gradients. However, it kept the same penalty-based objective, so the per-target accuracy issue remained.

## Proposal

Replace the penalty-based formulation with a **constrained quadratic program** solved by the [Clarabel](https://oxfordcontrol.github.io/ClarabelDocs/) interior-point solver:

```
minimize    sum((x_i - 1)^2)                              [weight distortion]
subject to  target_j * (1 - eps) <= achieved_j <= target_j * (1 + eps)   [+/-0.5%]
            x_min <= x_i <= x_max                          [multiplier bounds]
```

Key differences from the current approach:

- **Targets are hard constraints, not penalties.** Every SOI target must be satisfied within +/-0.5% (configurable). No target can be sacrificed to improve others.
- **Deterministic across machines.** Clarabel is a Rust-based interior-point solver operating in float64. Interior-point methods converge to a unique optimum determined by the KKT conditions, not by reduction order.
- **Elastic slack variables** let the solver report which constraints (if any) cannot be satisfied, rather than failing entirely.
- **Faster.** ~17 seconds vs ~7 minutes for scipy L-BFGS-B, ~3 minutes for PyTorch L-BFGS.
- **No GPU or PyTorch dependency** for the reweighting step.

## Results

All ~800 SOI targets are satisfied within the 0.5% tolerance band. Previously problematic targets (unemployment compensation at ~8% error) are now within 0.5%. Cross-machine reproducibility has been confirmed: identical weight fingerprints on two different machines (different CPUs, different OS configurations).

## Backward compatibility

The PyTorch and scipy solvers are preserved and selectable via environment variables (`PYTORCH_REWEIGHT=1`, `SCIPY_REWEIGHT=1`). Clarabel becomes the default.
