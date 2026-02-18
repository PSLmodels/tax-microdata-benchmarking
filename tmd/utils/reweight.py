"""
This module provides utilities for reweighting a flat file
to match AGI targets.
"""

import time
import numpy as np
import pandas as pd
import torch
from tmd.storage import STORAGE_FOLDER
from tmd.utils.soi_replication import tc_to_soi
from tmd.imputation_assumptions import (
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    REWEIGHT_DEVIATION_PENALTY,
)

TAX_YEAR = 2021  # set equal to TAXYEAR in create_taxcalc_input_variables.py

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
        return f"{x / 1e6:.0f}m"
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
        "estate_income",
        "estate_losses",
        "exempt_interest",
        "ira_distributions",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_income",
        "rent_and_royalty_net_losses",
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
        agi_range_label = f"{fmt(lob)}-{fmt(hib)}"
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

    # Drop impossible targets: columns where all data values are
    # zero (no reweighting can produce nonzero estimates from
    # all-zero data)
    loss_matrix_arr = loss_matrix.values
    targets_arr = np.array(targets_array)
    all_zero_mask = (loss_matrix_arr == 0).all(axis=0)
    if all_zero_mask.any():
        impossible_labels = [
            loss_matrix.columns[i]
            for i in range(len(all_zero_mask))
            if all_zero_mask[i]
        ]
        print(
            f"WARNING: Dropping {len(impossible_labels)} impossible "
            f"targets (all-zero data columns):"
        )
        for label in impossible_labels:
            print(f"  - {label}")
        keep_mask = ~all_zero_mask
        loss_matrix = loss_matrix.loc[:, keep_mask]
        targets_arr = targets_arr[keep_mask]

    return loss_matrix.copy(), targets_arr


def reweight(
    flat_file: pd.DataFrame,
    time_period: int = TAX_YEAR,
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
        print(
            f"...GPU acceleration enabled: " f"{gpu_name} ({gpu_mem:.1f} GB)"
        )
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

    # Check for NaN columns
    for i in range(len(target_array)):
        if torch.isnan(outputs[i]).any():
            print(f"Column {output_matrix.columns[i]} has NaN values")

    # L-BFGS optimizer: quasi-Newton method with line search.
    # Converges much faster than Adam for this smooth problem.
    # The closure function is called multiple times per step
    # (line search).
    max_lbfgs_iter = 200
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

    print(
        f"...starting L-BFGS optimization " f"(up to {max_lbfgs_iter} steps)"
    )
    optimization_start_time = time.time()

    prev_loss = float("inf")
    for step_count in range(1, max_lbfgs_iter + 1):
        optimizer.step(closure)
        current_loss = loss_value.item()
        if step_count % 10 == 0 or step_count <= 5:
            print(f"    step {step_count:>4d}: " f"loss={current_loss:.10f}")
        # Convergence check
        if abs(prev_loss - current_loss) < 1e-12:
            print(
                f"    converged at step {step_count} " f"(loss change < 1e-12)"
            )
            break
        prev_loss = current_loss

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
