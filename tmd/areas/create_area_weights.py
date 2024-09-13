"""
Construct AREA_tmd_weights.csv.gz, a Tax-Calculator-style weights file
for 2021+ for the specified sub-national AREA.

AREA prefix for state areas are the two lower-case character postal codes.
AREA prefix for congressional districts are the state prefix followed by
two digits (with a leading zero) identifying the district.  There are no
district files for states with only one congressional district.
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, Bounds
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER
from tmd.areas import AREAS_FOLDER

FIRST_YEAR = 2021
LAST_YEAR = 2074
INFILE_PATH = STORAGE_FOLDER / "output" / "tmd.csv.gz"
WTFILE_PATH = STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"
GFFILE_PATH = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"
POPFILE_PATH = STORAGE_FOLDER / "input" / "cbo_population_forecast.yaml"

REGULARIZATION_DELTA = 1.0e-9
OPTIMIZE_FTOL = 1e-8
OPTIMIZE_GTOL = 1e-8
OPTIMIZE_MAXITER = 5000
OPTIMIZE_IPRINT = 0  # 20 is a good diagnostic value; set to 0 for production
OPTIMIZE_RESULTS = False  # set to True to see complete optimization results
DUMP_ALL_TARGET_DEVIATIONS = False  # set to True only for diagnostic work


def all_taxcalc_variables():
    """
    Return all read and calc Tax-Calculator variables in pd.DataFrame.
    """
    input_data = tc.Records(
        data=pd.read_csv(INFILE_PATH),
        start_year=2021,
        weights=str(WTFILE_PATH),
        gfactors=tc.GrowFactors(growfactors_filename=str(GFFILE_PATH)),
        adjust_ratios=None,
        exact_calculations=True,
    )
    sim = tc.Calculator(records=input_data, policy=tc.Policy())
    sim.calc_all()
    vdf = sim.dataframe([], all_vars=True)
    assert np.all(vdf.s006 > 0), "Not all weights are positive"
    return vdf


def prepared_data(area: str, vardf: pd.DataFrame):
    """
    Construct numpy 2-D target matrix and 1-D target array for
    specified area using specified vardf.  Also, compute initial
    weights scaling factor for specified area.  Return all three
    as a tuple.
    """
    national_population = (vardf.s006 * vardf.XTOT).sum()
    numobs = len(vardf)
    tdf = pd.read_csv(AREAS_FOLDER / "targets" / f"{area}_targets.csv")
    tm_tuple = ()
    ta_list = []
    row_num = 1
    for row in tdf.itertuples(index=False):
        row_num += 1
        line = f"{area}:L{row_num}"
        # construct target amount for this row
        unscaled_target = row.target
        if unscaled_target == 0:
            unscaled_target = 1.0
        scale = 1.0 / unscaled_target
        scaled_target = unscaled_target * scale
        ta_list.append(scaled_target)
        # confirm that row_num 2 contains the area population target
        if row_num == 2:
            bool_list = [
                row.varname == "XTOT",
                row.count == 0,
                row.scope == 0,
                row.agilo < -8e99,
                row.agihi > 8e99,
                row.fstatus == 0,
            ]
            assert all(
                bool_list
            ), f"{line} does not contain the area population target"
            initial_weights_scale = row.target / national_population
        # construct variable array for this target
        assert (
            row.count >= 0 and row.count <= 1
        ), f"count value {row.count} not in [0,1] range on {line}"
        if row.count == 0:
            unmasked_varray = vardf[row.varname].astype(float)
        else:
            unmasked_varray = (vardf[row.varname] > 0).astype(float)
        mask = np.ones(numobs, dtype=int)
        assert (
            row.scope >= 0 and row.scope <= 2
        ), f"scope value {row.scope} not in [0,2] range on {line}"
        if row.scope == 1:
            mask *= vardf.data_source == 1  # PUF records
        elif row.scope == 2:
            mask *= vardf.data_source == 0  # CPS records
        in_bin = (vardf.c00100 >= row.agilo) & (vardf.c00100 < row.agihi)
        mask *= in_bin
        assert (
            row.fstatus >= 0 and row.fstatus <= 5
        ), f"fstatus value {row.fstatus} not in [0,5] range on {line}"
        if row.fstatus > 0:
            mask *= vardf.MARS == row.fstatus
        scaled_masked_varray = mask * unmasked_varray * scale
        tm_tuple = tm_tuple + (scaled_masked_varray,)
    # construct target matrix and target array and return as tuple
    scale_factor = 1.0  # as high as 1e9 works just fine
    target_matrix = np.vstack(tm_tuple).T * scale_factor
    target_array = np.array(ta_list) * scale_factor
    return (
        target_matrix,
        target_array,
        initial_weights_scale,
    )


def target_rmse(wght, target_matrix, target_array):
    """
    Return RMSE of the target deviations given specified arguments.
    """
    act = np.dot(wght, target_matrix)
    act_minus_exp = act - target_array
    ratio = act / target_array
    if DUMP_ALL_TARGET_DEVIATIONS:
        for tnum, ratio_ in enumerate(ratio):
            print(
                f"TARGET{(tnum + 1):03d}:ACT-EXP,ACT/EXP= "
                f"{act_minus_exp[tnum]:16.9e}, {ratio_:.3f}"
            )
    # show distribution of target ratios
    bins = [
        0.0,
        0.4,
        0.8,
        0.9,
        0.99,
        0.9995,
        1.0005,
        1.01,
        1.1,
        1.2,
        1.6,
        2.0,
        3.0,
        4.0,
        5.0,
        np.inf,
    ]
    tot = ratio.size
    print(f"DISTRIBUTION OF TARGET ACT/EXP RATIOS (n={tot}):")
    print(f"  with REGULARIZATION_DELTA= {REGULARIZATION_DELTA:e}")
    header = (
        "low bin ratio    high bin ratio"
        "    bin #    cum #     bin %     cum %"
    )
    print(header)
    out = pd.cut(ratio, bins, right=False, precision=6)
    count = pd.Series(out).value_counts().sort_index().to_dict()
    cum = 0
    for interval, num in count.items():
        cum += num
        if cum == 0:
            continue
        line = (
            f">={interval.left:13.6f}, <{interval.right:13.6f}:"
            f"  {num:6d}   {cum:6d}   {num/tot:7.2%}   {cum/tot:7.2%}"
        )
        print(line)
        if cum == tot:
            break
    # return RMSE of ACT-EXP targets
    return np.sqrt(np.mean(np.square(act_minus_exp)))


def objective_function(x, *args):
    """
    Objective function for minimization.
    Search for NOTE in this file for methodological details.
    https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=320
    """
    A, b, delta = args  # A is a jax sparse matrix
    ssq_target_deviations = jnp.sum(jnp.square(A @ x - b))
    ssq_weight_deviations = jnp.sum(jnp.square(x - 1.0))
    return ssq_target_deviations + delta * ssq_weight_deviations


JIT_FVAL_AND_GRAD = jax.jit(jax.value_and_grad(objective_function))


def weight_ratio_distribution(ratio):
    """
    Print distribution of post-optimized to pre-optimized weight ratios.
    """
    bins = [
        0.0,
        1e-6,
        0.1,
        0.2,
        0.5,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        2.0,
        5.0,
        1e1,
        1e2,
        1e3,
        1e4,
        1e5,
        np.inf,
    ]
    tot = ratio.size
    print(f"DISTRIBUTION OF AREA/US WEIGHT RATIO (n={tot}):")
    print(f"  with REGULARIZATION_DELTA= {REGULARIZATION_DELTA:e}")
    header = (
        "low bin ratio    high bin ratio"
        "    bin #    cum #     bin %     cum %"
    )
    print(header)
    out = pd.cut(ratio, bins, right=False, precision=6)
    count = pd.Series(out).value_counts().sort_index().to_dict()
    cum = 0
    for interval, num in count.items():
        cum += num
        if cum == 0:
            continue
        line = (
            f">={interval.left:13.6f}, <{interval.right:13.6f}:"
            f"  {num:6d}   {cum:6d}   {num/tot:7.2%}   {cum/tot:7.2%}"
        )
        print(line)
        if cum == tot:
            break
    ssqdev = np.sum(np.square(ratio - 1.0))
    print(f"SUM OF SQUARED AREA/US WEIGHT RATIO DEVIATIONS= {ssqdev:e}")


# -- High-level logic of the script:


def create_area_weights_file(area: str, write_file: bool = True):
    """
    Create Tax-Calculator-style weights file for FIRST_YEAR through LAST_YEAR
    for specified area using information in area targets CSV file.
    Return target RMSE using the optimized area weights and optionally write
    the weights file.
    """
    if write_file:
        print(f"CREATING WEIGHTS FILE FOR AREA {area} ...")
    else:
        print(f"DOING JUST WEIGHTS FILE CALCULATIONS FOR AREA {area} ...")
    jax.config.update("jax_platform_name", "cpu")  # ignore GPU/TPU if present
    jax.config.update("jax_enable_x64", True)  # use double precision floats

    # construct variable matrix and target array and weights_scale
    vdf = all_taxcalc_variables()
    target_matrix, target_array, weights_scale = prepared_data(area, vdf)
    wght_us = np.array(vdf.s006 * weights_scale)
    num_weights = len(wght_us)
    num_targets = len(target_array)
    print(f"USING {area}_targets.csv FILE CONTAINING {num_targets} TARGETS")
    rmse = target_rmse(wght_us, target_matrix, target_array)
    print(f"US_PROPORTIONALLY_SCALED_TARGET_RMSE= {rmse:.9e}")
    density = np.count_nonzero(target_matrix) / target_matrix.size
    print(f"target_matrix sparsity ratio = {(1.0 - density):.3f}")

    # optimize weight ratios by minimizing the sum of squared deviations
    # of area-to-us weight ratios from one such that the optimized ratios
    # hit all of the area targets
    #
    # NOTE: This a bi-criterion minimization problem that can be
    #       solved using regularization methods.  For background,
    #       consult Stephen Boyd and Lieven Vandenberghe, Convex
    #       Optimization, Cambridge University Press, 2004, in
    #       particular equation (6.9) on page 306 (see LINK below).
    #       Our problem is exactly the same as (6.9) except that
    #       we measure x deviations from one rather than from zero.
    # LINK: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=320
    #
    A_dense = (target_matrix * wght_us[:, np.newaxis]).T
    A = BCOO.from_scipy_sparse(csr_matrix(A_dense))  # A is JAX sparse matrix
    b = target_array
    print(
        f"OPTIMIZE_WEIGHT_RATIOS: target_matrix.shape= {target_matrix.shape}\n"
        f"REGULARIZATION_DELTA= {REGULARIZATION_DELTA:e}"
    )
    time0 = time.time()
    res = minimize(
        fun=JIT_FVAL_AND_GRAD,  # objective function and its gradient
        x0=np.ones(num_weights),  # initial guess for weight ratios
        jac=True,  # use gradient from JIT_FVAL_AND_GRAD function
        args=(A, b, REGULARIZATION_DELTA),  # fixed arguments of objective func
        method="L-BFGS-B",  # use L-BFGS-B algorithm
        bounds=Bounds(0.0, np.inf),  # consider only non-negative weight ratios
        options={
            "maxiter": OPTIMIZE_MAXITER,
            "ftol": OPTIMIZE_FTOL,
            "gtol": OPTIMIZE_GTOL,
            "iprint": OPTIMIZE_IPRINT,
            "disp": OPTIMIZE_IPRINT != 0,
        },
    )
    time1 = time.time()
    res_summary = (
        f">>> optimization execution time: {(time1-time0):.1f} secs"
        f"  iterations={res.nit}  success={res.success}\n"
        f">>> message: {res.message}\n"
        f">>> L-BFGS-B optimized objective function value: {res.fun:.9e}"
    )
    print(res_summary)
    if OPTIMIZE_RESULTS:
        print(">>> full optimization results:\n", res)
    wght_area = res.x * wght_us
    rmse = target_rmse(wght_area, target_matrix, target_array)
    print(f"AREA-OPTIMIZED_TARGET_RMSE= {rmse:.9e}")
    weight_ratio_distribution(res.x)

    if not write_file:
        return rmse

    # write weights file
    """
    # write annual weights extrapolating using national population forecast
    # get population forecast
    with open(pop_file, "r", encoding="utf-8") as pfile:
        pop = yaml.safe_load(pfile.read())

    # get FIRST_YEAR weights from VARFILE
    vdf = pd.read_csv(VARFILE)
    weights = vdf.s006 * 100  # scale up weights by 100 for Tax-Calculator

    # construct dictionary of scaled-up weights by year
    wdict = {f"WT{FIRST_YEAR}": weights}
    cum_pop_growth = 1.0
    for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        annual_pop_growth = pop[year] / pop[year - 1]
        cum_pop_growth *= annual_pop_growth
        wght = weights.copy() * cum_pop_growth
        wdict[f"WT{year}"] = wght

    # write rounded integer scaled-up weights to CSV-formatted file
    wdf = pd.DataFrame.from_dict(wdict)
    wdf.to_csv(WGTFILE, index=False, float_format="%.0f", compression="gzip")
    """

    return rmse


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(
            "ERROR: exactly one command-line argument is required\n"
        )
        sys.exit(1)
    area_code = sys.argv[1]
    tfile = f"{area_code}_targets.csv"
    target_file = AREAS_FOLDER / "targets" / tfile
    if not target_file.exists():
        sys.stderr.write(
            f"ERROR: {tfile} file not in tmd/areas/targets folder\n"
        )
        sys.exit(1)
    create_area_weights_file(area_code, write_file=True)
