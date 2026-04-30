"""
Clarabel-based constrained QP area weight optimization.

Finds weight multipliers (x) for each record such that area-specific
weighted sums match area targets within a tolerance band, while
minimizing deviation from population-proportional weights.

Formulation:
    minimize    sum((x_i - 1)^2)
    subject to  t_j*(1-eps) <= (Bx)_j <= t_j*(1+eps)   [target bounds]
                x_min <= x_i <= x_max                    [multiplier bounds]

where:
    x_i = area_weight_i / (pop_share * national_weight_i)
    pop_share = area_population / national_population
    B[j,i] = (pop_share * national_weight_i) * A[j,i]

    x_i = 1 means record i gets its population-proportional share.
    The optimizer adjusts x_i to hit area-specific targets.

Elastic slack variables handle infeasibility gracefully:
    minimize    sum((x_i - 1)^2) + M * sum(s^2)
    subject to  lb_j <= (Bx)_j + s_lo - s_hi <= ub_j,  s >= 0

Follows the same QP construction as tmd/utils/reweight.py.

This module is a library, not a CLI.  It is called by
``tmd.areas.solve_weights`` (parallel batch solver, the production
entry point) and ``tmd.areas.developer_tools`` (relaxation cascade).
Key public entry points:

    create_area_weights_file(area, ...)   — solve one area, write
                                             ``<area>_tmd_weights.csv.gz``
                                             and ``<area>.log``.
    cd_target_dir(congress)               — directory of CD target
                                             files for the given session.
    cd_weight_dir(congress)               — directory of CD weight
                                             files for the given session.
"""

import sys
import time

import re
from pathlib import Path

import clarabel
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog
from scipy.sparse import (
    coo_matrix,
    diags as spdiags,
    eye as speye,
    hstack,
    vstack,
)

from tmd.areas import AREAS_FOLDER
from tmd.imputation_assumptions import POPULATION_FILE, TAXYEAR
from tmd.storage import STORAGE_FOLDER

FIRST_YEAR = TAXYEAR
LAST_YEAR = 2034
INFILE_PATH = STORAGE_FOLDER / "output" / "tmd.csv.gz"
POPFILE_PATH = STORAGE_FOLDER / "input" / POPULATION_FILE
TAXCALC_AGI_CACHE = STORAGE_FOLDER / "output" / "cached_c00100.npy"
CACHED_ALLVARS_PATH = STORAGE_FOLDER / "output" / "cached_allvars.csv"

# Tax-Calculator output variables to load from cached_allvars for targeting
CACHED_TC_OUTPUTS = [
    "c18300",
    "c04470",
    "c02500",
    "c19200",
    "c19700",
    "eitc",
    "ctc_total",
]

# Default solver parameters
AREA_CONSTRAINT_TOL = 0.005
AREA_SLACK_PENALTY = 1e6
AREA_REDUCED_SLACK_PENALTY = 1e3
AREA_MAX_ITER = 2000
AREA_MULTIPLIER_MIN = 0.0
AREA_MULTIPLIER_MAX = 25.0
CD_MULTIPLIER_MAX = 50.0

# Default target/weight directories
STATE_TARGET_DIR = AREAS_FOLDER / "targets" / "states"
STATE_WEIGHT_DIR = AREAS_FOLDER / "weights" / "states"


def cd_target_dir(congress: int) -> "Path":
    """Return the target-directory path for the given Congress session."""
    if congress not in (118, 119):
        raise ValueError(
            f"Unsupported Congress session: {congress}. Supported: (118, 119)"
        )
    return AREAS_FOLDER / "targets" / f"cds_{congress}"


def cd_weight_dir(congress: int) -> "Path":
    """Return the weight-directory path for the given Congress session."""
    if congress not in (118, 119):
        raise ValueError(
            f"Unsupported Congress session: {congress}. Supported: (118, 119)"
        )
    return AREAS_FOLDER / "weights" / f"cds_{congress}"


# Back-compat aliases (118 only).  New code should call
# ``cd_target_dir(congress)`` / ``cd_weight_dir(congress)`` with an
# explicit Congress session.
CD_TARGET_DIR = cd_target_dir(118)
CD_WEIGHT_DIR = cd_weight_dir(118)


def _load_taxcalc_data():
    """Load TMD data with cached AGI and selected Tax-Calculator outputs.

    After loading, drops columns not used by the solver to reduce
    per-worker memory (~150 MB savings with 109 → ~25 columns).
    """
    vdf = pd.read_csv(INFILE_PATH)
    new_cols = {"c00100": np.load(TAXCALC_AGI_CACHE)}
    if CACHED_ALLVARS_PATH.exists():
        allvars = pd.read_csv(CACHED_ALLVARS_PATH, usecols=CACHED_TC_OUTPUTS)
        for col in CACHED_TC_OUTPUTS:
            if col in allvars.columns:
                new_cols[col] = allvars[col].values
    vdf = pd.concat([vdf, pd.DataFrame(new_cols, index=vdf.index)], axis=1)
    # Synthetic combined variable for net capital gains targeting
    if "p22250" in vdf.columns and "p23250" in vdf.columns:
        capgains_net = vdf["p22250"].values + vdf["p23250"].values
        vdf = pd.concat(
            [
                vdf,
                pd.DataFrame({"capgains_net": capgains_net}, index=vdf.index),
            ],
            axis=1,
        )
    assert np.all(vdf.s006 > 0), "Not all weights are positive"
    # Drop columns not used by the solver to reduce memory.
    # Keep infrastructure columns + any column that could be a target
    # variable (e*, c*, p* prefixes, plus synthetic variables).
    # This adapts automatically as target specs change.
    _INFRA_COLS = {"s006", "MARS", "data_source", "XTOT", "RECID"}
    _SYNTH_COLS = {"capgains_net", "eitc", "ctc_total"}
    keep = set()
    for col in vdf.columns:
        if col in _INFRA_COLS or col in _SYNTH_COLS:
            keep.add(col)
        elif col[:1] in ("e", "c", "p") and col[1:2].isdigit():
            keep.add(col)
    vdf = vdf[sorted(keep)]
    return vdf


def _read_params(area, out, target_dir=None):
    """Read optional area-specific parameters YAML file."""
    if target_dir is None:
        target_dir = STATE_TARGET_DIR
    params = {}
    pfile = f"{area}_params.yaml"
    params_path = target_dir / pfile
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f.read())
        exp_params = [
            "target_ratio_tolerance",
            "dump_all_target_deviations",
            "delta_init_value",
            "delta_max_loops",
            "iprint",
            "constraint_tol",
            "slack_penalty",
            "max_iter",
            "multiplier_min",
            "multiplier_max",
        ]
        act_params = list(params.keys())
        all_ok = len(set(act_params)) == len(act_params)
        for param in act_params:
            if param not in exp_params:
                all_ok = False
                out.write(
                    f"WARNING: {pfile} parameter" f" {param} is not expected\n"
                )
        if not all_ok:
            out.write(f"IGNORING CONTENTS OF {pfile}\n")
            params = {}
        elif params:
            out.write(f"USING CUSTOMIZED PARAMETERS IN {pfile}\n")
    return params


def _build_constraint_matrix(area, vardf, out, target_dir=None):
    """
    Build constraint matrix B and target array from area targets CSV.

    Returns (B_csc, targets, target_labels, pop_share) where:
    - B_csc is a sparse matrix (n_targets x n_records)
    - targets is 1-D array of target values
    - target_labels is a list of descriptive strings
    - pop_share is the area's population share of national population
    """
    if target_dir is None:
        target_dir = STATE_TARGET_DIR
    national_population = (vardf.s006 * vardf.XTOT).sum()
    numobs = len(vardf)
    targets_file = target_dir / f"{area}_targets.csv"
    tdf = pd.read_csv(targets_file, comment="#")

    # Pre-cache column arrays to avoid repeated DataFrame lookups
    c00100_arr = vardf["c00100"].values.astype(float)
    mars_arr = vardf["MARS"].values.astype(float)
    data_source_arr = vardf["data_source"].values
    scope1_mask = (data_source_arr == 1).astype(float)
    scope2_mask = (data_source_arr == 0).astype(float)
    ones_arr = np.ones(numobs, dtype=float)

    unique_vars = tdf.loc[tdf["count"] != 1, "varname"].unique()
    var_cache = {}
    for vname in unique_vars:
        var_cache[vname] = vardf[vname].astype(float).values

    targets_list = []
    labels_list = []
    pop_share = None
    w0 = None

    # Build B matrix directly in sparse COO format, avoiding dense
    # intermediates that would require ~620 MB for 215K × 179.
    row_indices = []
    col_indices = []
    data_values = []

    for row_idx, row in enumerate(tdf.itertuples(index=False)):
        line = f"{area}:target{row_idx + 1}"

        # extract target value
        target_val = row.target
        targets_list.append(target_val)

        # build label
        label = (
            f"{row.varname}"
            f"/cnt={row.count}"
            f"/scope={row.scope}"
            f"/agi=[{row.agilo},{row.agihi})"
            f"/fs={row.fstatus}"
        )
        labels_list.append(label)

        # first row must be XTOT population target
        if row_idx == 0:
            assert row.varname == "XTOT", f"{line}: first target must be XTOT"
            assert row.count == 0 and row.scope == 0
            assert row.agilo < -8e99 and row.agihi > 8e99
            assert row.fstatus == 0
            pop_share = row.target / national_population
            out.write(
                f"pop_share = {row.target:.0f}"
                f" / {national_population:.0f}"
                f" = {pop_share:.6f}\n"
            )
            w0 = pop_share * vardf.s006.values

        # construct variable array from cache
        assert 0 <= row.count <= 4, f"count {row.count} not in [0,4] on {line}"
        if row.count == 0:
            var_array = var_cache[row.varname]
        elif row.count == 1:
            var_array = ones_arr
        elif row.count == 2:
            var_array = (var_cache[row.varname] != 0).astype(float)
        elif row.count == 3:
            var_array = (var_cache[row.varname] > 0).astype(float)
        else:
            var_array = (var_cache[row.varname] < 0).astype(float)

        # construct mask using cached arrays
        assert 0 <= row.scope <= 2, f"scope {row.scope} not in [0,2] on {line}"
        if row.scope == 1:
            mask = scope1_mask.copy()
        elif row.scope == 2:
            mask = scope2_mask.copy()
        else:
            mask = ones_arr.copy()

        mask *= (c00100_arr >= row.agilo) & (c00100_arr < row.agihi)

        assert (
            0 <= row.fstatus <= 5
        ), f"fstatus {row.fstatus} not in [0,5] on {line}"
        if row.fstatus > 0:
            mask *= mars_arr == row.fstatus

        # B[j,i] = w0[i] * mask[i] * var_array[i]
        # Store only nonzero entries (sparse COO format)
        b_row = w0 * mask * var_array
        nz = np.nonzero(b_row)[0]
        if len(nz) > 0:
            row_indices.append(np.full(len(nz), row_idx, dtype=np.int32))
            col_indices.append(nz.astype(np.int32))
            data_values.append(b_row[nz])

    assert pop_share is not None, "XTOT target not found"

    # Assemble sparse B directly from COO — no dense intermediate
    n_targets = len(targets_list)
    B_csc = coo_matrix(
        (
            np.concatenate(data_values),
            (np.concatenate(row_indices), np.concatenate(col_indices)),
        ),
        shape=(n_targets, numobs),
    ).tocsc()

    targets_arr = np.array(targets_list)
    return B_csc, targets_arr, labels_list, pop_share


def _drop_impossible_targets(
    B_csc,
    targets,
    labels,
    out,
    multiplier_min=AREA_MULTIPLIER_MIN,
    multiplier_max=AREA_MULTIPLIER_MAX,
    constraint_tol=AREA_CONSTRAINT_TOL,
):
    """
    Drop targets that cannot be reached within tolerance.

    A target is impossible if:
      - All B matrix values are zero (no records contribute), or
      - The target is outside the achievable range even with all
        multipliers at their extreme bounds.
    """
    m = len(targets)
    drop = np.zeros(m, dtype=bool)

    for j in range(m):
        # Extract one sparse row at a time — avoids 310 MB dense copy
        row = B_csc.getrow(j).toarray().ravel()
        pos = row > 0
        neg = row < 0

        # All zeros
        if not pos.any() and not neg.any():
            out.write(
                f"DROPPING impossible target" f" (all zeros): {labels[j]}\n"
            )
            drop[j] = True
            continue

        # Compute achievable range
        max_val = (
            multiplier_max * row[pos].sum() + multiplier_min * row[neg].sum()
        )
        min_val = (
            multiplier_min * row[pos].sum() + multiplier_max * row[neg].sum()
        )

        tgt = targets[j]
        tol = abs(tgt) * constraint_tol
        if max_val < tgt - tol or min_val > tgt + tol:
            out.write(
                f"DROPPING unreachable target: {labels[j]}"
                f" (target={tgt:.0f},"
                f" achievable=[{min_val:.0f}, {max_val:.0f}])\n"
            )
            drop[j] = True

    n_drop = int(drop.sum())
    if n_drop > 0:
        keep = ~drop
        # Slice sparse matrix directly — no dense conversion
        keep_idx = np.where(keep)[0]
        B_csc = B_csc[keep_idx, :]
        targets = targets[keep]
        labels = [lab for lab, k in zip(labels, keep) if k]
        out.write(
            f"Dropped {n_drop} impossible/unreachable targets,"
            f" {len(targets)} remaining\n"
        )
    return B_csc, targets, labels


def _check_feasibility(
    B_csc,
    targets,
    labels,
    n_records,
    constraint_tol=AREA_CONSTRAINT_TOL,
    multiplier_min=AREA_MULTIPLIER_MIN,
    multiplier_max=AREA_MULTIPLIER_MAX,
    out=None,
):
    """
    Fast LP feasibility check before running the full QP.

    Solves: minimize 0 subject to
        cl <= Bx + s_lo - s_hi <= cu
        multiplier_min <= x <= multiplier_max
        s_lo, s_hi >= 0

    If infeasible, identifies which constraints contribute most
    to the infeasibility by solving a relaxed LP that minimizes
    total slack.

    Returns
    -------
    dict with keys:
        feasible : bool
        slack_needed : np.ndarray or None
            Per-constraint slack needed (0 if feasible).
        worst_labels : list of (label, slack_amount)
            Top constraints requiring the most slack.
    """
    if out is None:
        out = sys.stdout

    m = len(targets)
    abs_targets = np.abs(targets)
    tol_band = abs_targets * constraint_tol
    cl = targets - tol_band
    cu = targets + tol_band

    # LP: minimize sum(s_lo + s_hi)
    # Variables: [x (n_records), s_lo (m), s_hi (m)]
    n_total = n_records + 2 * m

    # Objective: minimize sum of slacks
    c_obj = np.zeros(n_total)
    c_obj[n_records:] = 1.0  # minimize total slack

    # Constraints: cl <= Bx + s_lo - s_hi <= cu
    # Rewrite as:
    #   Bx + s_lo - s_hi >= cl  =>  -Bx - s_lo + s_hi <= -cl
    #   Bx + s_lo - s_hi <= cu
    # Build entirely in sparse format to avoid ~1.2 GB dense alloc
    I_m = speye(m, format="csc")
    # [B | I | -I] x <= cu
    A_upper = hstack([B_csc, I_m, -I_m], format="csc")
    # [-B | -I | I] x <= -cl
    A_lower = hstack([-B_csc, -I_m, I_m], format="csc")
    A_ub = vstack([A_upper, A_lower], format="csc")
    b_ub = np.concatenate([cu, -cl])

    # Bounds
    if isinstance(multiplier_max, np.ndarray):
        x_ub = multiplier_max
    else:
        x_ub = np.full(n_records, multiplier_max)
    bounds = np.column_stack(
        [
            np.concatenate(
                [
                    np.full(n_records, multiplier_min),
                    np.zeros(2 * m),
                ]
            ),
            np.concatenate(
                [
                    x_ub,
                    np.full(2 * m, np.inf),
                ]
            ),
        ]
    )

    result = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 30},
    )

    info = {"feasible": False, "slack_needed": None, "worst_labels": []}

    if result.success:
        s_lo = result.x[n_records : n_records + m]
        s_hi = result.x[n_records + m :]
        total_slack = s_lo + s_hi
        info["feasible"] = total_slack.max() < 1e-6
        info["slack_needed"] = total_slack

        if not info["feasible"]:
            # Report constraints needing the most slack
            worst_idx = np.argsort(total_slack)[::-1]
            worst = []
            for idx in worst_idx[:10]:
                if total_slack[idx] > 1e-6:
                    rel = total_slack[idx] / max(abs(targets[idx]), 1)
                    worst.append((labels[idx], total_slack[idx], rel))
            info["worst_labels"] = worst

            out.write(
                f"PRE-SOLVE FEASIBILITY: INFEASIBLE"
                f" — {len(worst)} constraints need slack\n"
            )
            for lbl, slk, rel in worst[:5]:
                out.write(
                    f"  {lbl}: slack={slk:.0f}" f" ({rel:.1%} of target)\n"
                )
        else:
            out.write("PRE-SOLVE FEASIBILITY: OK\n")
    else:
        out.write(
            f"PRE-SOLVE FEASIBILITY: LP solver failed" f" ({result.message})\n"
        )

    return info


def _assign_slack_penalties(
    labels,
    default_penalty=AREA_SLACK_PENALTY,
    reduced_penalty=AREA_REDUCED_SLACK_PENALTY,
):
    """
    Assign per-constraint slack penalties based on label content.

    Most constraints get the default (high) penalty. Constraints
    involving variables and AGI bins where SOI-to-TMD mapping is
    inherently noisy get a reduced penalty — the solver will still
    try to hit them but will relax them before distorting weights.

    Reduced-penalty targets:
      - e02400, e00300, e26270 amounts in AGI stubs 1-3 (<$25K).
        These income types are concentrated among high-income or
        elderly filers; low-AGI bin targets are noisy and can
        conflict with other constraints in extreme CDs.
      - Filing-status counts (fs=1,2,4) in stubs 1-2 (<$10K).
        Very small cells in the lowest income bins.

    Returns
    -------
    np.ndarray
        Penalty value for each constraint, same length as labels.
    """
    penalties = np.full(len(labels), default_penalty)
    reduced_vars = {"e02400", "e00300", "e26270"}

    for i, label in enumerate(labels):
        parts = label.split("/")
        varname = parts[0]
        attrs = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                attrs[k] = v

        cnt = int(attrs.get("cnt", -1))
        fs = int(attrs.get("fs", 0))
        agi_raw = attrs.get("agi", "")

        # Parse AGI upper bound from "[lo,hi)"
        agihi = 9e99
        m = re.match(r"\[([^,]+),([^)]+)\)", agi_raw)
        if m:
            agihi = float(m.group(2))

        # Reduced penalty for problematic variable-bin combos
        if varname in reduced_vars and cnt == 0 and agihi <= 25000:
            penalties[i] = reduced_penalty
        elif cnt == 1 and fs > 0 and agihi <= 10000:
            penalties[i] = reduced_penalty

    return penalties


def _solve_area_qp(  # pylint: disable=unused-argument
    B_csc,
    targets,
    labels,
    n_records,
    constraint_tol=AREA_CONSTRAINT_TOL,
    slack_penalty=AREA_SLACK_PENALTY,
    slack_penalties=None,
    max_iter=AREA_MAX_ITER,
    multiplier_min=AREA_MULTIPLIER_MIN,
    multiplier_max=AREA_MULTIPLIER_MAX,
    weight_penalty=1.0,
    out=None,
):
    """
    Solve the area reweighting QP using Clarabel.

    Parameters
    ----------
    slack_penalty : float
        Default penalty for constraint slack (used when
        slack_penalties is None).
    slack_penalties : np.ndarray, optional
        Per-constraint slack penalties (length m). Overrides
        slack_penalty when provided.
    weight_penalty : float
        Penalty weight on (x-1)^2 relative to constraint
        slack. Higher values keep multipliers closer to 1.0
        at the cost of more target violations.

    Returns (x_opt, s_lo, s_hi, info_dict).
    """
    if out is None:
        out = sys.stdout

    m = len(targets)
    n_total = n_records + 2 * m  # x + s_lo + s_hi

    # constraint bounds: t*(1-eps) <= Bx <= t*(1+eps)
    abs_targets = np.abs(targets)
    tol_band = abs_targets * constraint_tol
    cl = targets - tol_band
    cu = targets + tol_band

    # diagonal Hessian: 2*alpha for x, 2*M for slacks
    # Each slack pair (s_lo, s_hi) gets the same penalty
    hess_diag = np.empty(n_total)
    hess_diag[:n_records] = 2.0 * weight_penalty
    if slack_penalties is not None:
        hess_diag[n_records : n_records + m] = 2.0 * slack_penalties
        hess_diag[n_records + m :] = 2.0 * slack_penalties
    else:
        hess_diag[n_records:] = 2.0 * slack_penalty
    P = spdiags(hess_diag, format="csc")

    # linear term: -2*alpha for x
    q = np.zeros(n_total)
    q[:n_records] = -2.0 * weight_penalty

    # extended constraint matrix: [B | I_m | -I_m]
    I_m = speye(m, format="csc")
    A_full = hstack([B_csc, I_m, -I_m], format="csc")

    # constraint scaling for numerical conditioning
    target_scale = np.maximum(np.abs(targets), 1.0)
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

    # Clarabel form: Ax + s = b, s in NonnegativeCone
    I_n = speye(n_total, format="csc")
    A_clar = vstack(
        [A_scaled, -A_scaled, I_n, -I_n],
        format="csc",
    )
    b_clar = np.concatenate([cu_scaled, -cl_scaled, var_ub, -var_lb])

    m_constraints = len(b_clar)
    # pylint: disable=no-member
    cones = [clarabel.NonnegativeConeT(m_constraints)]

    settings = clarabel.DefaultSettings()
    # pylint: enable=no-member
    settings.verbose = False
    settings.max_iter = max_iter
    settings.tol_gap_abs = 1e-7
    settings.tol_gap_rel = 1e-7
    settings.tol_feas = 1e-7

    # solve
    out.write("STARTING CLARABEL SOLVER...\n")
    t_start = time.time()
    solver = clarabel.DefaultSolver(  # pylint: disable=no-member
        P, q, A_clar, b_clar, cones, settings
    )
    result = solver.solve()
    elapsed = time.time() - t_start

    status_str = str(result.status)
    out.write(
        f"Solver status: {status_str}\n"
        f"Iterations: {result.iterations}\n"
        f"Solve time: {elapsed:.2f}s\n"
    )

    # extract solution
    y_opt = np.array(result.x)
    x_opt = y_opt[:n_records]
    s_lo = y_opt[n_records : n_records + m]
    s_hi = y_opt[n_records + m :]

    x_opt = np.clip(x_opt, multiplier_min, multiplier_max)

    info = {
        "status": status_str,
        "iterations": result.iterations,
        "solve_time": elapsed,
        "clarabel_solve_time": result.solve_time,
        "dual": np.array(result.z) if result.z is not None else None,
    }

    return x_opt, s_lo, s_hi, info


def _print_target_diagnostics(
    x_opt, B_csc, targets, labels, constraint_tol, out
):
    """Print target accuracy diagnostics."""
    achieved = np.asarray(B_csc @ x_opt).ravel()
    abs_errors = np.abs(achieved - targets)
    rel_errors = abs_errors / np.maximum(np.abs(targets), 1.0)

    out.write(f"TARGET ACCURACY ({len(targets)} targets):\n")
    out.write(f"  mean |relative error|: {rel_errors.mean():.6f}\n")
    out.write(f"  max  |relative error|: {rel_errors.max():.6f}\n")

    eps = 1e-9
    n_violated = int((rel_errors > constraint_tol + eps).sum())
    n_hit = len(targets) - n_violated
    out.write(
        f"  targets hit: {n_hit}/{len(targets)}"
        f" (tolerance: +/-{constraint_tol * 100:.1f}% + eps)\n"
    )
    if n_violated > 0:
        out.write(f"  VIOLATED: {n_violated} targets\n")
        worst_idx = np.argsort(rel_errors)[::-1]
        for idx in worst_idx[: min(10, n_violated)]:
            out.write(
                f"    {rel_errors[idx] * 100:7.3f}%"
                f" | target={targets[idx]:15.0f}"
                f" | achieved={achieved[idx]:15.0f}"
                f" | {labels[idx]}\n"
            )
    return n_violated


def _print_multiplier_diagnostics(x_opt, out):
    """Print weight multiplier distribution diagnostics."""
    out.write("MULTIPLIER DISTRIBUTION:\n")
    out.write(
        f"  min={x_opt.min():.6f},"
        f" p5={np.percentile(x_opt, 5):.6f},"
        f" median={np.median(x_opt):.6f},"
        f" p95={np.percentile(x_opt, 95):.6f},"
        f" max={x_opt.max():.6f}\n"
    )
    out.write(
        f"  RMSE from 1.0:" f" {np.sqrt(np.mean((x_opt - 1.0) ** 2)):.6f}\n"
    )

    # distribution bins
    bins = [
        0.0,
        1e-6,
        0.1,
        0.5,
        0.8,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.2,
        1.5,
        2.0,
        5.0,
        10.0,
        100.0,
        np.inf,
    ]
    tot = len(x_opt)
    out.write(f"  distribution (n={tot}):\n")
    for i in range(len(bins) - 1):
        count = int(((x_opt >= bins[i]) & (x_opt < bins[i + 1])).sum())
        if count > 0:
            out.write(
                f"    [{bins[i]:10.4f},"
                f" {bins[i + 1]:10.4f}):"
                f" {count:7d}"
                f" ({count / tot:7.2%})\n"
            )


def _print_slack_diagnostics(s_lo, s_hi, targets, labels, out):
    """Print elastic slack diagnostics."""
    total_slack = s_lo + s_hi
    n_active = int(np.sum(total_slack > 1e-6))
    if n_active > 0:
        out.write(
            f"ELASTIC SLACK active on"
            f" {n_active}/{len(targets)} constraints:\n"
        )
        slack_idx = np.where(total_slack > 1e-6)[0]
        for idx in slack_idx[np.argsort(total_slack[slack_idx])[::-1]][:20]:
            out.write(
                f"  slack={total_slack[idx]:12.2f}"
                f" | target={targets[idx]:15.0f}"
                f" | {labels[idx]}\n"
            )
    else:
        out.write("ALL CONSTRAINTS SATISFIED WITHOUT SLACK\n")


def create_area_weights_file(
    area,
    write_log=True,
    write_file=True,
    target_dir=None,
    weight_dir=None,
):
    """
    Create area weights file using Clarabel constrained QP solver.

    Returns 0 on success.
    """
    if target_dir is None:
        target_dir = STATE_TARGET_DIR
    if weight_dir is None:
        weight_dir = STATE_WEIGHT_DIR
    # ensure output directory exists
    weight_dir.mkdir(parents=True, exist_ok=True)

    # remove any existing output files
    awpath = weight_dir / f"{area}_tmd_weights.csv.gz"
    awpath.unlink(missing_ok=True)
    logpath = weight_dir / f"{area}.log"
    logpath.unlink(missing_ok=True)

    # set up output
    if write_log:
        out = open(  # pylint: disable=consider-using-with
            logpath, "w", encoding="utf-8"
        )
    else:
        out = sys.stdout

    if write_file:
        out.write(
            f"CREATING WEIGHTS FILE FOR AREA {area}" " (Clarabel solver) ...\n"
        )
    else:
        out.write(
            f"DOING JUST CALCS FOR AREA {area}" " (Clarabel solver) ...\n"
        )

    # read optional parameters
    params = _read_params(area, out, target_dir=target_dir)
    constraint_tol = params.get(
        "constraint_tol",
        params.get("target_ratio_tolerance", AREA_CONSTRAINT_TOL),
    )
    slack_penalty = params.get("slack_penalty", AREA_SLACK_PENALTY)
    max_iter = params.get("max_iter", AREA_MAX_ITER)
    multiplier_min = params.get("multiplier_min", AREA_MULTIPLIER_MIN)
    multiplier_max = params.get("multiplier_max", AREA_MULTIPLIER_MAX)

    # load data and build constraint matrix
    vdf = _load_taxcalc_data()
    out.write(f"Loaded {len(vdf)} records\n")
    out.write(f"National weight sum: {vdf.s006.sum():.0f}\n")

    B_csc, targets, labels, pop_share = _build_constraint_matrix(
        area, vdf, out, target_dir=target_dir
    )
    out.write(
        f"Built constraint matrix:"
        f" {B_csc.shape[0]} targets x"
        f" {B_csc.shape[1]} records\n"
    )
    out.write(f"Constraint tolerance: +/-{constraint_tol * 100:.1f}%\n")
    out.write(f"Multiplier bounds: [{multiplier_min}, {multiplier_max}]\n")

    # drop impossible targets
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc, targets, labels, out
    )

    # check what x=1 (population-proportional) achieves
    n_records = B_csc.shape[1]
    x_ones = np.ones(n_records)
    achieved_x1 = np.asarray(B_csc @ x_ones).ravel()
    rel_err_x1 = np.abs(achieved_x1 - targets) / np.maximum(
        np.abs(targets), 1.0
    )
    out.write("BEFORE OPTIMIZATION (x=1, population-proportional):\n")
    out.write(f"  mean |relative error|: {rel_err_x1.mean():.6f}\n")
    out.write(f"  max  |relative error|: {rel_err_x1.max():.6f}\n")

    # solve QP
    per_constraint_penalties = _assign_slack_penalties(
        labels, default_penalty=slack_penalty
    )
    x_opt, s_lo, s_hi, _info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        slack_penalties=per_constraint_penalties,
        max_iter=max_iter,
        multiplier_min=multiplier_min,
        multiplier_max=multiplier_max,
        out=out,
    )

    # diagnostics
    _n_violated = _print_target_diagnostics(
        x_opt, B_csc, targets, labels, constraint_tol, out
    )
    _print_multiplier_diagnostics(x_opt, out)
    _print_slack_diagnostics(s_lo, s_hi, targets, labels, out)

    if write_log:
        out.close()
    if not write_file:
        return 0

    # write area weights file with population extrapolation
    w0 = pop_share * vdf.s006.values
    wght_area = x_opt * w0

    with open(POPFILE_PATH, "r", encoding="utf-8") as pf:
        pop = yaml.safe_load(pf.read())

    wdict = {f"WT{FIRST_YEAR}": wght_area}
    cum_pop_growth = 1.0
    for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        annual_pop_growth = pop[year] / pop[year - 1]
        cum_pop_growth *= annual_pop_growth
        wdict[f"WT{year}"] = wght_area * cum_pop_growth

    wdf = pd.DataFrame.from_dict(wdict)
    wdf.to_csv(
        awpath,
        index=False,
        float_format="%.5f",
        compression="gzip",
    )

    return 0
