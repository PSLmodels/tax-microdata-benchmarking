"""
This module provides utilities for reweighting a flat file
to match AGI targets.
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
from tmd.storage import STORAGE_FOLDER
from tmd.utils.soi_replication import tc_to_soi
from tmd.imputation_assumptions import (
    TAXYEAR,
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    REWEIGHT_DEVIATION_PENALTY,
)

INCOME_RANGES = [
    -np.inf,
    1,
    5e3,
    1e4,
    1.5e4,
    2e4,
    2.5e4,
    3e4,
    4e4,
    5e4,
    7.5e4,
    1e5,
    2e5,
    5e5,
    1e6,
    1.5e6,
    2e6,
    5e6,
    1e7,
    np.inf,
]


def fmt(x):
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    if x < 1e3:
        return f"{x:.0f}"
    if x < 1e6:
        return f"{x / 1e3:.0f}k"
    if x < 1e9:
        return f"{x / 1e6:.1f}m"
    return f"{x / 1e9:.1f}bn"


def build_loss_matrix(df, targets, time_period):
    """Build loss matrix and target array for reweighting.

    Returns (loss_matrix, targets_array) where loss_matrix is a
    DataFrame with one column per target and targets_array is the
    corresponding SOI target values.
    """
    columns = {}
    df = tc_to_soi(df, time_period)
    agi = df["adjusted_gross_income"].values
    filer = df["is_tax_filer"].values
    targets_array = []
    soi_subset = targets
    soi_subset = soi_subset[soi_subset.Year == time_period]
    agi_level_targeted_variables = [
        "adjusted_gross_income",
        "count",
        "employment_income",
        "business_net_profits",
        "capital_gains_gross",
        "ordinary_dividends",
        "partnership_and_s_corp_income",
        "qualified_dividends",
        "taxable_interest_income",
        "total_pension_income",
        "total_social_security",
    ]
    aggregate_level_targeted_variables = [
        "business_net_losses",
        "capital_gains_distributions",
        "capital_gains_losses",
        # "estate_income",  # all zeros in tc_to_soi (not in Tax-Calculator)
        # "estate_losses",  # all zeros in tc_to_soi (not in Tax-Calculator)
        "exempt_interest",
        "ira_distributions",
        "partnership_and_s_corp_losses",
        # "rent_and_royalty_net_income",  # all zeros in tc_to_soi (not in TC)
        # "rent_and_royalty_net_losses",  # all zeros in tc_to_soi (not in TC)
        "taxable_pension_income",
        "taxable_social_security",
        "unemployment_compensation",
    ]
    aggregate_level_targeted_variables = [
        variable
        for variable in aggregate_level_targeted_variables
        if variable in df.columns
    ]
    soi_subset = soi_subset[
        soi_subset.Variable.isin(agi_level_targeted_variables)
        & (
            (soi_subset["AGI lower bound"] != -np.inf)
            | (soi_subset["AGI upper bound"] != np.inf)
        )
        | (
            soi_subset.Variable.isin(aggregate_level_targeted_variables)
            & (soi_subset["AGI lower bound"] == -np.inf)
            & (soi_subset["AGI upper bound"] == np.inf)
        )
    ]
    for _, row in soi_subset.iterrows():
        if row["Taxable only"]:
            continue  # exclude "taxable returns" statistics

        mask = (
            (agi >= row["AGI lower bound"])
            * (agi < row["AGI upper bound"])
            * filer
        ) > 0

        if row["Filing status"] == "Single":
            mask *= df["filing_status"].values == "SINGLE"
        elif row["Filing status"] == "Married Filing Jointly/Surviving Spouse":
            mask *= df["filing_status"].values == "JOINT"
        elif row["Filing status"] == "Head of Household":
            mask *= df["filing_status"].values == "HEAD_OF_HOUSEHOLD"
        elif row["Filing status"] == "Married Filing Separately":
            mask *= df["filing_status"].values == "SEPARATE"

        values = df[row["Variable"]].values

        if row["Count"]:
            values = (values > 0).astype(float)

        lob = row["AGI lower bound"]
        hib = row["AGI upper bound"]
        left = "(" if lob == -np.inf else "["
        agi_range_label = f"{left}{fmt(lob)}, {fmt(hib)})"
        taxable_label = (
            "taxable" if row["Taxable only"] else "all" + " returns"
        )
        filing_status_label = row["Filing status"]

        variable_label = row["Variable"].replace("_", " ")

        if row["Count"] and not row["Variable"] == "count":
            label = (
                f"{variable_label}/count/AGI in "
                f"{agi_range_label}/{taxable_label}/"
                f"{filing_status_label}"
            )
        elif row["Variable"] == "count":
            label = (
                f"{variable_label}/count/AGI in "
                f"{agi_range_label}/{taxable_label}/"
                f"{filing_status_label}"
            )
        else:
            label = (
                f"{variable_label}/total/AGI in "
                f"{agi_range_label}/{taxable_label}/"
                f"{filing_status_label}"
            )

        if label not in columns:
            columns[label] = mask * values
            targets_array.append(row["Value"])

    loss_matrix = pd.DataFrame(columns)
    targets_arr = np.array(targets_array)
    return _drop_impossible_targets(loss_matrix, targets_arr)


def _drop_impossible_targets(loss_matrix, targets_arr):
    """Drop targets where all data values are zero.

    No reweighting can produce nonzero estimates from all-zero data,
    so these targets must be excluded before optimization.

    Returns (filtered_loss_matrix, filtered_targets_arr).
    """
    all_zero_mask = (loss_matrix.values == 0).all(axis=0)
    if all_zero_mask.any():
        impossible_labels = loss_matrix.columns[all_zero_mask].tolist()
        label_list = "\n  - ".join(impossible_labels)
        warnings.warn(
            f"Dropping {len(impossible_labels)} impossible targets "
            f"(all data values are zero):\n  - {label_list}",
            UserWarning,
            stacklevel=2,
        )
        loss_matrix = loss_matrix.loc[:, ~all_zero_mask]
        targets_arr = targets_arr[~all_zero_mask]
    return loss_matrix.copy(), targets_arr


def reweight(
    flat_file: pd.DataFrame,
    time_period: int = TAXYEAR,
    weight_multiplier_min: float = REWEIGHT_MULTIPLIER_MIN,
    weight_multiplier_max: float = REWEIGHT_MULTIPLIER_MAX,
    weight_deviation_penalty: float = REWEIGHT_DEVIATION_PENALTY,
    use_gpu: bool = True,
):
    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")

    if time_period not in targets.Year.unique():
        raise ValueError(f"Year {time_period} not in targets.")
    print(f"...reweighting for year {time_period}")
    print(f"...weight deviation penalty: {weight_deviation_penalty}")
    print(
        f"...weight multiplier bounds: "
        f"[{weight_multiplier_min}, {weight_multiplier_max}]"
    )

    # Save original unscaled weights for final comparison
    original_unscaled_weights = flat_file.s006.values.copy()

    # GPU Detection and Device Selection
    gpu_available = torch.cuda.is_available()
    use_gpu_actual = use_gpu and gpu_available

    device = torch.device("cuda" if use_gpu_actual else "cpu")

    if use_gpu_actual:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"...GPU acceleration enabled: {gpu_name} ({gpu_mem:.1f} GB)")
    elif use_gpu and not gpu_available:
        print("...GPU requested but not available, using CPU")
    elif not use_gpu and gpu_available:
        print("...GPU available but disabled by user, using CPU")
    else:
        print("...GPU not available, using CPU")

    # Reset GPU state for reproducibility: prior PyTorch operations
    # (e.g., PolicyEngine Microsimulation) can leave the GPU in a
    # state that causes non-deterministic results.
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    rng_seed = 65748392
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)

    # Pre-multiply weights so the filer total matches the SOI
    # target. This gives the optimizer a better starting point and
    # ensures the weight deviation penalty only penalizes
    # redistributive changes, not the overall level shift.
    soi_filer_total_row = targets[
        (targets.Year == time_period)
        & (targets.Variable == "count")
        & (targets["Filing status"] == "All")
        & (targets["AGI lower bound"] == -np.inf)
        & (targets["AGI upper bound"] == np.inf)
        & (~targets["Taxable only"])
    ]
    if len(soi_filer_total_row) == 1:
        target_filer_total = soi_filer_total_row["Value"].values[0]
        soi_df = tc_to_soi(flat_file.copy(), time_period)
        filer_mask = soi_df["is_tax_filer"].values.astype(bool)
        current_filer_total = (flat_file.s006.values * filer_mask).sum()
        prescale = target_filer_total / current_filer_total
        flat_file["s006"] *= prescale
        print(
            f"...pre-scaled weights: "
            f"target filers={target_filer_total:,.0f}, "
            f"current filers={current_filer_total:,.0f}, "
            f"scale={prescale:.6f}"
        )
    else:
        print(
            "WARNING: Could not find unique SOI filer total, "
            "skipping weight pre-scaling"
        )

    # Create tensors directly on the selected device
    # to avoid non-leaf tensor issues
    weights = torch.tensor(
        flat_file.s006.values, dtype=torch.float64, device=device
    )
    weight_multiplier = torch.tensor(
        np.ones_like(flat_file.s006.values),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    original_weights = weights.clone()
    output_matrix, target_array = build_loss_matrix(
        flat_file, targets, time_period
    )

    print(f"Targeting {len(target_array)} SOI statistics")

    # Diagnostic: input data summary
    input_weights = flat_file.s006.values
    print(
        f"...input records: {len(flat_file)}, "
        f"columns: {len(flat_file.columns)}"
    )
    print(
        f"...input weights: total={input_weights.sum():.2f}, "
        f"mean={input_weights.mean():.6f}, "
        f"sdev={input_weights.std():.6f}"
    )

    # print out non-numeric columns
    for col in output_matrix.columns:
        try:
            torch.tensor(output_matrix[col].values, dtype=torch.float64)
        except ValueError:
            print(f"Column {col} is not numeric")
    output_matrix_tensor = torch.tensor(
        output_matrix.values, dtype=torch.float64, device=device
    )
    target_array = torch.tensor(
        target_array, dtype=torch.float64, device=device
    )

    outputs = (weights * output_matrix_tensor.T).sum(axis=1)
    original_loss_value = (((outputs + 1) / (target_array + 1) - 1) ** 2).sum()
    print(f"...initial loss: {original_loss_value.item():.10f}")

    # L-BFGS optimizer: quasi-Newton method with line search.
    # Converges much faster than Adam for this smooth problem.
    # The closure function is called multiple times per step
    # (line search).
    max_lbfgs_iter = 800
    optimizer = torch.optim.LBFGS(
        [weight_multiplier],
        max_iter=20,  # max line-search iterations per step
        history_size=10,  # past gradients for Hessian approx
        line_search_fn="strong_wolfe",
    )

    step_count = 0
    loss_value = None

    def closure():
        nonlocal loss_value
        optimizer.zero_grad()
        new_weights = weights * torch.clamp(
            weight_multiplier,
            min=weight_multiplier_min,
            max=weight_multiplier_max,
        )
        outputs = (new_weights * output_matrix_tensor.T).sum(axis=1)
        weight_deviation = (
            ((new_weights - original_weights) ** 2).sum()
            / (original_weights**2).sum()
            * weight_deviation_penalty
            * original_loss_value
        )
        loss_value = (
            ((outputs + 1) / (target_array + 1) - 1) ** 2
        ).sum() + weight_deviation
        loss_value.backward()
        return loss_value

    print(f"...starting L-BFGS optimization (up to {max_lbfgs_iter} steps)")
    optimization_start_time = time.time()

    for step_count in range(1, max_lbfgs_iter + 1):
        optimizer.step(closure)
        current_loss = loss_value.item()
        grad_norm = (
            weight_multiplier.grad.norm().item()
            if weight_multiplier.grad is not None
            else float("inf")
        )
        if step_count % 10 == 0 or step_count <= 5:
            print(
                f"    step {step_count:>4d}: loss={current_loss:.10f}, "
                f"grad={grad_norm:.2e}"
            )
        # Convergence check: gradient norm is the proper first-order
        # optimality condition (vs loss-change which can falsely trigger
        # when the Hessian approximation is poor and steps are tiny)
        if grad_norm < 1e-5:
            print(
                f"    converged at step {step_count} "
                f"(grad norm {grad_norm:.2e} < 1e-5)"
            )
            break

    # Recompute final weights and outputs after optimization
    new_weights = weights * torch.clamp(
        weight_multiplier,
        min=weight_multiplier_min,
        max=weight_multiplier_max,
    )
    outputs = (new_weights * output_matrix_tensor.T).sum(axis=1)

    optimization_end_time = time.time()
    optimization_duration = optimization_end_time - optimization_start_time

    final_loss = loss_value.item()
    print(
        f"...optimization completed in "
        f"{optimization_duration:.1f} seconds "
        f"({step_count} steps)"
    )
    print(f"...final loss: {final_loss:.10f}")

    # Move final weights back to CPU for numpy conversion
    final_weights = new_weights.detach().cpu().numpy()
    print(
        f"...final weights: total={final_weights.sum():.2f}, "
        f"mean={final_weights.mean():.6f}, "
        f"sdev={final_weights.std():.6f}"
    )

    # Target hit statistics
    rel_errors = (
        ((outputs + 1) / (target_array + 1) - 1).detach().cpu().numpy()
    )
    abs_rel_errors = np.abs(rel_errors)
    print(f"...target accuracy ({len(target_array)} targets):")
    print(f"    mean |relative error|: " f"{abs_rel_errors.mean():.6f}")
    print(f"    max  |relative error|: " f"{abs_rel_errors.max():.6f}")
    pct_bins = [0.001, 0.01, 0.05, 0.10]
    for threshold in pct_bins:
        n_within = (abs_rel_errors <= threshold).sum()
        print(
            f"    within {threshold * 100:5.1f}%: "
            f"{n_within:>4d}/{len(target_array)} "
            f"({n_within / len(target_array) * 100:.1f}%)"
        )
    # Show worst 10 targets
    worst_idx = np.argsort(abs_rel_errors)[::-1][:10]
    print("    worst targets:")
    for idx in worst_idx:
        label = output_matrix.columns[idx]
        print(f"      {abs_rel_errors[idx] * 100:7.3f}% | {label}")

    # Weight change distribution (vs original unscaled weights)
    ratio = final_weights / np.where(
        original_unscaled_weights == 0,
        1e-10,
        original_unscaled_weights,
    )
    abs_pct = np.abs(ratio - 1) * 100
    print("...weight changes (vs pre-optimization weights):")
    print("    weight ratio (new/original):")
    print(
        f"      min={ratio.min():.6f}, "
        f"p5={np.percentile(ratio, 5):.6f}, "
        f"median={np.median(ratio):.6f}, "
        f"p95={np.percentile(ratio, 95):.6f}, "
        f"max={ratio.max():.6f}"
    )
    bins = [0, 0.01, 0.1, 1, 5, 10, 100, float("inf")]
    labels = [
        "<0.01%",
        "0.01-0.1%",
        "0.1-1%",
        "1-5%",
        "5-10%",
        "10-100%",
        ">100%",
    ]
    print("    distribution of |% change|:")
    for i in range(len(bins) - 1):
        count = ((abs_pct >= bins[i]) & (abs_pct < bins[i + 1])).sum()
        print(
            f"      {labels[i]:>10s}: "
            f"{count:>7,} "
            f"({count / len(abs_pct) * 100:.1f}%)"
        )

    # Reproducibility fingerprint: compare these values across
    # machines to verify near-identical results (agreement to
    # ~4-6 significant figures = good)
    print("...REPRODUCIBILITY FINGERPRINT:")
    print(
        f"    weights: n={len(final_weights)}, "
        f"total={final_weights.sum():.6f}, "
        f"mean={final_weights.mean():.6f}, "
        f"sdev={final_weights.std():.6f}"
    )
    print(
        f"    weights: min={final_weights.min():.6f}, "
        f"p25={np.percentile(final_weights, 25):.6f}, "
        f"p50={np.median(final_weights):.6f}, "
        f"p75={np.percentile(final_weights, 75):.6f}, "
        f"max={final_weights.max():.6f}"
    )
    print(f"    sum(weights^2)=" f"{np.sum(final_weights**2):.6f}")
    print(f"    final loss: {final_loss:.10f}")

    print("...reweighting finished")
    flat_file["s006"] = final_weights
    return flat_file


def reweight_lbfgsb(
    flat_file,
    time_period=TAXYEAR,
    weight_multiplier_min=REWEIGHT_MULTIPLIER_MIN,
    weight_multiplier_max=REWEIGHT_MULTIPLIER_MAX,
    weight_deviation_penalty=REWEIGHT_DEVIATION_PENALTY,
    verbose=None,
):
    """Reweight using scipy L-BFGS-B (Fortran, projected-gradient bounds).

    Uses the same penalty-based objective as the PyTorch L-BFGS solver
    but with scipy's L-BFGS-B which has proper projected-gradient
    bounds (no clamping plateau problem), analytical gradient, and
    float64 precision throughout.

    Parameters
    ----------
    flat_file : pd.DataFrame
        Input data with s006 weights column.
    time_period : int
        Tax year for targets.
    weight_multiplier_min, weight_multiplier_max : float
        Bounds on weight multipliers.
    weight_deviation_penalty : float
        Penalty for weight deviation from original.
    verbose : bool or None
        Print full diagnostics.  None = check VERBOSE_REWEIGHT env var.

    Returns
    -------
    pd.DataFrame
        flat_file with updated s006 weights.
    """
    # pylint: disable=import-outside-toplevel
    import os
    from scipy.optimize import minimize as scipy_minimize

    # pylint: enable=import-outside-toplevel

    if verbose is None:
        verbose = os.environ.get("VERBOSE_REWEIGHT", "").lower() in (
            "1",
            "true",
            "yes",
        )

    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")
    if time_period not in targets.Year.unique():
        raise ValueError(f"Year {time_period} not in targets.")

    print(f"...scipy L-BFGS-B reweighting for year {time_period}")
    print(f"...weight deviation penalty: {weight_deviation_penalty}")
    print(
        f"...weight multiplier bounds: "
        f"[{weight_multiplier_min}, {weight_multiplier_max}]"
    )

    original_unscaled_weights = flat_file.s006.values.copy()

    # Prescaling (same as reweight())
    soi_filer_total_row = targets[
        (targets.Year == time_period)
        & (targets.Variable == "count")
        & (targets["Filing status"] == "All")
        & (targets["AGI lower bound"] == -np.inf)
        & (targets["AGI upper bound"] == np.inf)
        & (~targets["Taxable only"])
    ]
    if len(soi_filer_total_row) == 1:
        target_filer_total = soi_filer_total_row["Value"].values[0]
        soi_df = tc_to_soi(flat_file.copy(), time_period)
        filer_mask = soi_df["is_tax_filer"].values.astype(bool)
        current_filer_total = (flat_file.s006.values * filer_mask).sum()
        prescale = target_filer_total / current_filer_total
        flat_file["s006"] *= prescale
        print(
            f"...pre-scaled weights: "
            f"target filers={target_filer_total:,.0f}, "
            f"current filers={current_filer_total:,.0f}, "
            f"scale={prescale:.6f}"
        )
    else:
        print(
            "WARNING: Could not find unique SOI filer total, "
            "skipping weight pre-scaling"
        )

    output_matrix, target_array = build_loss_matrix(
        flat_file, targets, time_period
    )
    print(f"Targeting {len(target_array)} SOI statistics")

    w0_np = flat_file.s006.values.astype(np.float64)
    A_np = output_matrix.values.astype(np.float64)
    t_np = target_array.astype(np.float64)

    # Build scaled matrix: B[i,j] = w0[i] * A[i,j] / (t[j] + 1)
    scale = 1.0 / (t_np + 1.0)
    B = w0_np[:, None] * A_np * scale[None, :]
    c = t_np * scale

    # Initial loss for penalty scaling
    residual0 = B.T @ np.ones(len(w0_np)) - c
    L0 = np.dot(residual0, residual0)
    print(f"...initial loss: {L0:.10f}")

    # Penalty coefficient per multiplier
    w0_sq_sum = np.sum(w0_np**2)
    lam = weight_deviation_penalty * L0 * (w0_np**2) / w0_sq_sum

    n_calls = [0]
    iter_state = {"last_loss": None, "last_grad": None}

    def objective_and_grad(m):
        n_calls[0] += 1
        residual = B.T @ m - c
        target_loss = np.dot(residual, residual)
        dev = m - 1.0
        penalty_loss = np.dot(lam, dev**2)
        loss = target_loss + penalty_loss
        grad = 2.0 * (B @ residual) + 2.0 * lam * dev
        iter_state["last_loss"] = loss
        iter_state["last_grad"] = np.max(np.abs(grad))
        return loss, grad

    def scipy_callback(_xk):
        nit = n_calls[0]
        loss = iter_state["last_loss"]
        gnorm = iter_state["last_grad"]
        if verbose and (nit <= 5 or nit % 50 == 0):
            print(
                f"    iter {nit:>5d}: "
                f"loss={loss:.10f}, "
                f"grad={gnorm:.2e}",
                flush=True,
            )

    m0 = np.ones(len(w0_np), dtype=np.float64)
    bounds = [(weight_multiplier_min, weight_multiplier_max)] * len(m0)

    print("...starting scipy L-BFGS-B optimization")
    t_start = time.time()

    result = scipy_minimize(
        objective_and_grad,
        m0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=scipy_callback,
        options={
            "maxiter": 4000,
            "ftol": 0,
            "gtol": 1e-5,
            "maxcor": 10,
            "disp": False,
        },
    )

    elapsed = time.time() - t_start

    m_final = result.x
    final_weights = w0_np * np.clip(
        m_final, weight_multiplier_min, weight_multiplier_max
    )

    print(
        f"...optimization completed in {elapsed:.1f} seconds "
        f"({result.nit} iterations, "
        f"{n_calls[0]} function evals)"
    )
    print(
        f"...result: success={result.success}, " f"message='{result.message}'"
    )
    print(f"...final loss: {result.fun:.10f}")
    print(
        f"...final weights: total={final_weights.sum():.2f}, "
        f"mean={final_weights.mean():.6f}, "
        f"sdev={final_weights.std():.6f}"
    )

    # Target accuracy
    outputs_np = final_weights @ A_np
    rel_errors = np.abs((outputs_np + 1) / (t_np + 1) - 1)
    print(f"...target accuracy ({len(t_np)} targets):")
    print(f"    mean |relative error|: {rel_errors.mean():.6f}")
    print(f"    max  |relative error|: {rel_errors.max():.6f}")

    # Worst 10 targets
    worst_idx = np.argsort(rel_errors)[::-1][:10]
    print("    worst targets:")
    target_labels = list(output_matrix.columns)
    for idx in worst_idx:
        print(
            f"      {rel_errors[idx] * 100:7.3f}% " f"| {target_labels[idx]}"
        )

    if verbose:
        # Reproducibility fingerprint
        print("...REPRODUCIBILITY FINGERPRINT:")
        print(
            f"    weights: n={len(final_weights)}, "
            f"total={final_weights.sum():.6f}, "
            f"mean={final_weights.mean():.6f}, "
            f"sdev={final_weights.std():.6f}"
        )
        print(
            f"    weights: min={final_weights.min():.6f}, "
            f"p25={np.percentile(final_weights, 25):.6f}, "
            f"p50={np.median(final_weights):.6f}, "
            f"p75={np.percentile(final_weights, 75):.6f}, "
            f"max={final_weights.max():.6f}"
        )
        print(f"    sum(weights^2)=" f"{np.sum(final_weights**2):.6f}")

        # Weight change distribution
        ratio = final_weights / np.where(
            original_unscaled_weights == 0,
            1e-10,
            original_unscaled_weights,
        )
        print("...weight changes (vs pre-optimization weights):")
        print(
            f"    ratio: min={ratio.min():.6f}, "
            f"median={np.median(ratio):.6f}, "
            f"max={ratio.max():.6f}"
        )

    print("...scipy L-BFGS-B reweighting finished")
    flat_file["s006"] = final_weights
    return flat_file
