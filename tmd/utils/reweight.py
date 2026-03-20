"""
Constrained quadratic programming (QP) reweighting using Clarabel solver.

Finds weight multipliers (x) that keep weights as close to their
original values as possible while requiring that weighted totals match
SOI targets within a specified tolerance (default +/-0.5%).

Formulation (Quadratic Program):
    minimize    sum((x_i - 1)^2)         [minimal weight distortion]
    subject to  t_j*(1-eps) <= (Bx)_j <= t_j*(1+eps)  [target bounds]
                x_min <= x_i <= x_max    [multiplier bounds]

With elastic slack variables for graceful infeasibility handling:
    minimize    sum((x_i - 1)^2) + M * sum(s^2)
    subject to  lb_j <= (Bx)_j + s_lo_j - s_hi_j <= ub_j
                s >= 0

The slack variables let the solver find a solution even when some
targets cannot be exactly satisfied.  A large penalty M (default 1e6)
means the solver tries very hard to satisfy all constraints, but if
a constraint is geometrically impossible it will report which ones
needed slack rather than failing entirely.

Constraint scaling: each constraint row is divided by |target_j| so
the right-hand side values are near 1.0 instead of ~1e8.  This does
not change the feasible region but roughly halves the iteration count
by improving numerical conditioning.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import (
    csc_matrix,
    diags as spdiags,
    eye as speye,
    hstack,
    vstack,
)
import clarabel
from tmd.storage import STORAGE_FOLDER
from tmd.utils.soi_replication import taxcalc_to_soi
from tmd.imputation_assumptions import (
    TAXYEAR,
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    CLARABEL_CONSTRAINT_TOL,
    CLARABEL_SLACK_PENALTY,
    CLARABEL_MAX_ITER,
)

_ABS_TOL_FLOOR = 0.0


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
    """
    Build loss matrix and target array for reweighting.

    Returns (loss_matrix, targets_array) where loss_matrix is a
    DataFrame with one column per target and targets_array is the
    corresponding SOI target values.
    """
    columns = {}
    df = taxcalc_to_soi(df, time_period)
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
        # "estate_income",  # all zeros in Tax-Calculator
        # "estate_losses",  # all zeros in Tax-Calculator
        "exempt_interest",
        "ira_distributions",
        "partnership_and_s_corp_losses",
        # "rent_and_royalty_net_income",  # all zeros in Tax-Calculator
        # "rent_and_royalty_net_losses",  # all zeros in Tax-Calculator
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
    """
    Drop targets where all data values are zero.

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


def _compute_constraint_bounds(
    targets,
    rel_tol,
    abs_tol_floor,
    target_labels=None,
    tolerance_overrides=None,
    verbose=False,
):
    """Compute per-constraint lower and upper bounds.

    For each target t_j, the bounds are:
        t_j - |t_j| * rel_tol  <=  achieved_j  <=  t_j + |t_j| * rel_tol

    Negative targets are handled correctly (tolerance is on |t_j|).

    tolerance_overrides is an optional dict mapping substring patterns
    to relative tolerances.  For example, {"unemployment_compensation": 0.05}
    widens UC targets to +/-5% while everything else uses rel_tol.
    """
    abs_targets = np.abs(targets)

    rel_tols = np.full(len(targets), rel_tol)
    if tolerance_overrides and target_labels is not None:
        for pattern, override_tol in tolerance_overrides.items():
            n_matched = 0
            for j, label in enumerate(target_labels):
                if pattern in label:
                    rel_tols[j] = override_tol
                    n_matched += 1
            if n_matched > 0 and verbose:
                print(
                    f"...tolerance override: '{pattern}' -> "
                    f"+-{override_tol * 100:.1f}% "
                    f"({n_matched} targets)"
                )

    tol_band = np.maximum(abs_targets * rel_tols, abs_tol_floor)
    cl = targets - tol_band
    cu = targets + tol_band
    return cl, cu


def _print_diagnostics(
    x_opt,
    s_lo,
    s_hi,
    B_csc,
    targets,
    cl,
    cu,
    prescaled_weights,
    original_weights,
    target_labels,
    info,
    elapsed,
    verbose,
):
    """
    Print solver results.
    Always prints a compact production summary;
    prints full diagnostics when verbose is True.
    """
    n = len(x_opt)
    m = len(targets)
    final_weights = prescaled_weights * x_opt

    status_msg = info.get("status_msg", b"unknown")
    if isinstance(status_msg, bytes):
        status_msg = status_msg.decode()

    n_iter = info.get("clarabel_iter", "?")
    solve_time = info.get("clarabel_solve_time", elapsed)
    print(f"...solve time: {solve_time:.1f}s ({n_iter} iterations)")
    print(f"...solver status: {status_msg}")

    # target accuracy (always printed)
    achieved = np.asarray(B_csc.T @ x_opt).ravel()
    abs_errors = np.abs(achieved - targets)
    rel_errors = abs_errors / np.maximum(np.abs(targets), 1.0)

    print(f"...target accuracy ({m} targets):")
    print(f"    mean |relative error|: {rel_errors.mean():.6f}")
    print(f"    max  |relative error|: {rel_errors.max():.6f}")
    n_violated = int((rel_errors > CLARABEL_CONSTRAINT_TOL + 1e-9).sum())
    if n_violated > 0:
        print(f"    VIOLATED: {n_violated}/{m} targets")

    # worst 10 targets (always printed)
    worst_idx = np.argsort(rel_errors)[::-1][:10]
    print("    worst targets:")
    for idx in worst_idx:
        label = target_labels[idx]
        print(
            f"      {rel_errors[idx] * 100:7.3f}% "
            f"| target={targets[idx]:15.0f} "
            f"| achieved={achieved[idx]:15.0f} "
            f"| {label}"
        )

    # final weight stats (always printed)
    print(
        f"...final weights: total={final_weights.sum():.2f}, "
        f"mean={final_weights.mean():.6f}, "
        f"sdev={final_weights.std():.6f}"
    )

    if not verbose:
        return

    # === Verbose-only output below ===

    # reproducibility fingerprint
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
    print(f"    sum(weights^2)={np.sum(final_weights ** 2):.6f}")
    print(
        f"    objective (weight deviation): "
        f"{np.sum((x_opt - 1.0) ** 2):.10f}"
    )

    # accuracy bins
    pct_bins = [0.001, 0.005, 0.01, 0.05, 0.10]
    for threshold in pct_bins:
        n_within = int((rel_errors <= threshold + 1e-9).sum())
        print(
            f"    within {threshold * 100:5.1f}%: "
            f"{n_within:>4d}/{m} "
            f"({n_within / m * 100:.1f}%)"
        )

    # constraint status
    at_lower = np.abs(achieved - cl) < 1e-4 * (np.abs(cl) + 1.0)
    at_upper = np.abs(achieved - cu) < 1e-4 * (np.abs(cu) + 1.0)
    binding = at_lower | at_upper
    violated_lo = achieved < cl - 1e-6 * (np.abs(cl) + 1.0)
    violated_hi = achieved > cu + 1e-6 * (np.abs(cu) + 1.0)
    violated = violated_lo | violated_hi
    n_binding = int(binding.sum())
    n_violated_v = int(violated.sum())
    n_interior = m - n_binding - n_violated_v
    print(f"...constraint status ({m} constraints):")
    print(f"    interior (slack): {n_interior}")
    print(f"    binding (at boundary): {n_binding}")
    print(f"    violated: {n_violated_v}")

    # elastic slack report
    total_slack = s_lo + s_hi
    n_active_slack = int(np.sum(total_slack > 1e-6))
    if n_active_slack > 0:
        print(
            f"...elastic slack active on " f"{n_active_slack}/{m} constraints:"
        )
        slack_idx = np.where(total_slack > 1e-6)[0]
        slack_order = slack_idx[np.argsort(total_slack[slack_idx])[::-1]]
        for idx in slack_order[:20]:
            label = target_labels[idx]
            pct_err = rel_errors[idx] * 100
            print(
                f"      slack={total_slack[idx]:12.2f} "
                f"| err={pct_err:7.3f}% "
                f"| target={targets[idx]:15.0f} "
                f"| achieved={achieved[idx]:15.0f} "
                f"| {label}"
            )
        if len(slack_order) > 20:
            print(f"      ... and {len(slack_order) - 20} more")
    else:
        print("...all constraints satisfied without slack")

    # dual-based constraint cost analysis
    duals = info.get("dual_constraints")
    n_tc = info.get("n_target_constraints")
    if duals is not None and n_tc is not None and n_tc > 0:
        dual_upper = duals[:n_tc]
        dual_lower = duals[n_tc : 2 * n_tc]
        dual_per_target = np.maximum(np.abs(dual_upper), np.abs(dual_lower))
        cost_per_pp = dual_per_target * np.maximum(np.abs(targets), 1.0) * 0.01
        ranked = np.argsort(cost_per_pp)[::-1]
        n_show = min(15, len(ranked))
        print(
            f"...constraint cost analysis " f"(top {n_show} most expensive):"
        )
        print(
            "    Marginal cost = approx objective reduction "
            "if tolerance relaxed by 1pp"
        )
        for rank, idx in enumerate(ranked[:n_show]):
            label = target_labels[idx]
            side = (
                "upper"
                if np.abs(dual_upper[idx]) >= np.abs(dual_lower[idx])
                else "lower"
            )
            print(
                f"    {rank + 1:3d}. "
                f"cost/pp={cost_per_pp[idx]:10.4f} "
                f"| dual={dual_per_target[idx]:.6f} "
                f"| side={side:5s} "
                f"| err={rel_errors[idx] * 100:7.3f}% "
                f"| {label}"
            )

    # x0 vs solution improvement
    achieved_x0 = np.asarray(B_csc.T @ np.ones(n)).ravel()
    rel_err_x0 = np.abs(achieved_x0 - targets) / np.maximum(
        np.abs(targets), 1.0
    )
    print("...constraint improvement (x0 vs solution):")
    print(
        f"    x0:  mean|err|={rel_err_x0.mean():.6f}, "
        f"max|err|={rel_err_x0.max():.6f}"
    )
    print(
        f"    sol: mean|err|={rel_errors.mean():.6f}, "
        f"max|err|={rel_errors.max():.6f}"
    )

    # weight change distribution
    ratio = final_weights / np.where(
        original_weights == 0, 1e-10, original_weights
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
        count = int(((abs_pct >= bins[i]) & (abs_pct < bins[i + 1])).sum())
        print(
            f"      {labels[i]:>10s}: "
            f"{count:>7,} "
            f"({count / len(abs_pct) * 100:.1f}%)"
        )

    # multiplier distribution
    print("...multiplier distribution:")
    print(
        f"    min={x_opt.min():.6f}, "
        f"p5={np.percentile(x_opt, 5):.6f}, "
        f"median={np.median(x_opt):.6f}, "
        f"p95={np.percentile(x_opt, 95):.6f}, "
        f"max={x_opt.max():.6f}"
    )


def reweight(
    flat_file,
    time_period=TAXYEAR,
    multiplier_min=REWEIGHT_MULTIPLIER_MIN,
    multiplier_max=REWEIGHT_MULTIPLIER_MAX,
    constraint_tol=CLARABEL_CONSTRAINT_TOL,
    slack_penalty=CLARABEL_SLACK_PENALTY,
    max_iter=CLARABEL_MAX_ITER,
    verbose=None,
    tolerance_overrides=None,
):
    """
    Reweight using Clarabel constrained QP solver.

    Finds weight multipliers that minimize total weight distortion
    while requiring that all SOI targets are met within the specified
    tolerance.  Uses elastic slack variables so the solver always
    returns a solution, reporting which constraints (if any) could
    not be satisfied.

    Parameters
    ----------
    flat_file : pd.DataFrame
        Input data with s006 weights column.
    time_period : int
        TAXYEAR for targets.
    multiplier_min, multiplier_max : float
        Bounds on weight multipliers.
    constraint_tol : float
        Relative tolerance on target constraints (default +/-0.5%).
    slack_penalty : float
        Penalty for constraint violations via elastic slack.
    max_iter : int
        Maximum solver iterations.
    verbose : bool or None
        Print full diagnostics.  None = check VERBOSE_REWEIGHT env var.
    tolerance_overrides : dict, optional
        Per-constraint tolerance overrides.  Maps substring patterns
        to relative tolerances.

    Returns
    -------
    pd.DataFrame
        flat_file with updated s006 weights.
    """
    if verbose is None:
        verbose = os.environ.get("VERBOSE_REWEIGHT", "").lower() in (
            "1",
            "true",
            "yes",
        )

    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")
    if time_period not in targets.Year.unique():
        raise ValueError(f"Year {time_period} not in targets.")

    print(f"...constrained QP reweighting for year {time_period}")
    print(f"...constraint tolerance: +-{constraint_tol * 100:.1f}%")
    print(f"...multiplier bounds: [{multiplier_min}, {multiplier_max}]")

    original_unscaled_weights = flat_file.s006.values.copy()

    # prescaling
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
        soi_df = taxcalc_to_soi(flat_file.copy(), time_period)
        filer_mask = soi_df["is_tax_filer"].values.astype(bool)
        current_filer_total = (flat_file.s006.values * filer_mask).sum()
        prescale = target_filer_total / current_filer_total
        flat_file["s006"] *= prescale
        print(
            f"...prescale factor: {prescale:.6f} "
            f"(target={target_filer_total:,.0f}, "
            f"current={current_filer_total:,.0f})"
        )
    else:
        print(
            "WARNING: Could not find unique SOI filer total, "
            "skipping weight pre-scaling"
        )

    # build constraint matrix
    output_matrix, target_array = build_loss_matrix(
        flat_file, targets, time_period
    )
    target_labels = list(output_matrix.columns)
    n_records = len(flat_file)
    n_targets = len(target_array)

    print(f"...targets: {n_targets}, records: {n_records}")

    w0 = flat_file.s006.values.copy()

    # B[i,j] = w0[i] * A[i,j]
    A_csc = csc_matrix(output_matrix.values)
    del output_matrix
    B_csc = spdiags(w0) @ A_csc
    del A_csc

    nnz = B_csc.nnz
    density = nnz / (n_records * n_targets) * 100
    print(
        f"...constraint matrix: "
        f"{n_records}x{n_targets}, "
        f"nnz={nnz:,}, density={density:.2f}%"
    )

    # constraint bounds
    cl, cu = _compute_constraint_bounds(
        target_array,
        constraint_tol,
        _ABS_TOL_FLOOR,
        target_labels,
        tolerance_overrides,
        verbose,
    )

    # build QP
    m = n_targets
    n_total = n_records + 2 * m  # x + s_lo + s_hi

    # diagonal Hessian: 2 for x, 2*M for slacks
    hess_diag = np.empty(n_total)
    hess_diag[:n_records] = 2.0
    hess_diag[n_records:] = 2.0 * slack_penalty
    P = spdiags(hess_diag, format="csc")

    # linear term: -2 for x (from expanding (x-1)^2), 0 for slacks
    q = np.zeros(n_total)
    q[:n_records] = -2.0

    # extended constraint matrix: [B^T | I_M | -I_M]
    B_T = B_csc.T.tocsc()
    I_M = speye(m, format="csc")
    A_full = hstack([B_T, I_M, -I_M], format="csc")

    # constraint scaling for numerical conditioning
    target_scale = np.maximum(np.abs(target_array), 1.0)
    D_inv = spdiags(1.0 / target_scale)
    A_scaled = (D_inv @ A_full).tocsc()
    cl_scaled = cl / target_scale
    cu_scaled = cu / target_scale

    # variable bounds
    var_lb = np.empty(n_total)
    var_ub = np.empty(n_total)
    var_lb[:n_records] = multiplier_min
    var_ub[:n_records] = multiplier_max
    var_lb[n_records:] = 0.0
    var_ub[n_records:] = 1e20

    # clarabel form: Ax + s = b, s in NonnegativeCone (i.e. Ax <= b)
    I_n = speye(n_total, format="csc")
    A_clar = vstack(
        [
            A_scaled,
            -A_scaled,
            I_n,
            -I_n,
        ],
        format="csc",
    )
    b_clar = np.concatenate([cu_scaled, -cl_scaled, var_ub, -var_lb])

    m_constraints = len(b_clar)
    # pylint: disable=no-member
    cones = [clarabel.NonnegativeConeT(m_constraints)]

    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.max_iter = max_iter
    settings.tol_gap_abs = 1e-7
    settings.tol_gap_rel = 1e-7
    settings.tol_feas = 1e-7

    # solve
    print("...starting solver")
    t_start = time.time()

    solver = clarabel.DefaultSolver(P, q, A_clar, b_clar, cones, settings)
    result = solver.solve()

    elapsed = time.time() - t_start
    status_str = str(result.status)

    # extract duals (convert back from scaled space)
    duals = np.array(result.z)
    duals[:m] /= target_scale
    duals[m : 2 * m] /= target_scale

    info = {
        "status": 0 if "Solved" in status_str else 1,
        "status_msg": status_str.encode(),
        "clarabel_iter": result.iterations,
        "clarabel_solve_time": result.solve_time,
        "dual_constraints": duals,
        "n_target_constraints": m,
    }

    # extract solution
    y_opt = np.array(result.x)
    x_opt = y_opt[:n_records]
    s_lo = y_opt[n_records : n_records + m]
    s_hi = y_opt[n_records + m :]

    x_opt = np.clip(x_opt, multiplier_min, multiplier_max)

    # diagnostics
    _print_diagnostics(
        x_opt,
        s_lo,
        s_hi,
        B_csc,
        target_array,
        cl,
        cu,
        w0,
        original_unscaled_weights,
        target_labels,
        info,
        elapsed,
        verbose,
    )

    # apply new weights to flat_file and return
    flat_file["s006"] = w0 * x_opt
    return flat_file
