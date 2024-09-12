"""
Construct AREA_tmd_weights.csv.gz, a Tax-Calculator-style weights file
for 2021+ for the specified sub-national AREA.

AREA prefix for state areas are the two lower-case character postal codes.
AREA prefix for congressional districts are the state prefix followed by
two digits (with a leading zero) identifying the district.  Note there
are no district files for states with only one congressional district.
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import Bounds, minimize
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

DUMP_LOSS_FUNCTION_VALUE_COMPONENTS = True
REGULARIZATION_LAMBDA = 1e-9
OPTIMIZE_FTOL = 1e-8
OPTIMIZE_GTOL = 1e-8
OPTIMIZE_MAXITER = 5000
OPTIMIZE_IPRINT = 20  # set to zero for no iteration information
OPTIMIZE_RESULTS = True  # set to True to see complete optimization results


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


def loss_function_value(wght, target_matrix, target_array):
    """
    Return loss function value given specified arguments.
    """
    act = np.dot(wght, target_matrix)
    act_minus_exp = act - target_array
    if DUMP_LOSS_FUNCTION_VALUE_COMPONENTS:
        for tnum, exp in enumerate(target_array):
            print(
                f"TARGET{(tnum + 1):03d}:ACT-EXP,ACT/EXP= "
                f"{act_minus_exp[tnum]:16.9e}, "
                f"{(act[tnum] / exp):.3f}"
            )
    return 0.5 * np.sum(np.square(act_minus_exp))


def objective_function(x, A, b, lambda_):
    """
    Objective function for minimization (sum of squared residuals).
    """
    target_deviation = A @ x - b  # JAX sparse matrix-vector multiplication
    weight_deviation = jnp.sqrt(lambda_) * (x - 1)
    residuals = jnp.concatenate([target_deviation, weight_deviation])
    return jnp.sum(jnp.square(residuals))


def gradient_function(x, A, b, lambda_):
    """
    Define gradient using JAX autodiff.
    """
    grad = jax.grad(objective_function)(x, A, b, lambda_)
    return np.asarray(grad)


def weight_ratio_distribution(ratio):
    """
    Print distribution of post-optimized to pre-optimized weight ratios.
    """
    bins = [
        0.0,
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
    print(f"  with REGULARIZATION_LAMBDA= {REGULARIZATION_LAMBDA:e}")
    header = (
        "low bin ratio    high bin ratio"
        "    bin #    cum #     bin %     cum %"
    )
    print(header)
    out = pd.cut(ratio, bins, right=False, precision=6, duplicates="drop")
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
    Return loss_function_value using the optimized weights and optionally
    write the weights file.
    """
    print(f"CREATING WEIGHTS FILE FOR AREA {area} ...")
    jax.config.update("jax_platform_name", "cpu")  # ignore GPU/TPU if present
    jax.config.update("jax_enable_x64", True)  # use double precision floats

    # construct variable matrix and target array and weights_scale
    vdf = all_taxcalc_variables()
    target_matrix, target_array, weights_scale = prepared_data(area, vdf)
    wght_us = np.array(vdf.s006 * weights_scale)
    num_weights = len(wght_us)
    num_targets = len(target_array)
    print(f"USING {area}_targets.csv FILE CONTAINING {num_targets} TARGETS")
    loss = loss_function_value(wght_us, target_matrix, target_array)
    print(f"US_PROPORTIONALLY_SCALED_LOSS_FUNCTION_VALUE= {loss:.9e}")
    density = np.count_nonzero(target_matrix) / target_matrix.size
    print(f"target_matrix sparsity ratio = {(1.0 - density):.3f}")

    # optimize weight ratios by minimizing the sum of squared deviations
    # of area-to-us weight ratios from one such that the optimized ratios
    # hit all of the areas targets, using traditional Ax = b nomenclature
    A_dense = (target_matrix * wght_us[:, np.newaxis]).T
    A = BCOO.from_scipy_sparse(csr_matrix(A_dense))  # A is JAX sparse matrix
    b = target_array
    print(
        f"OPTIMIZE_RATIOS: target_matrix.shape= {target_matrix.shape}\n"
        f"REGULARIZATION_LAMBDA= {REGULARIZATION_LAMBDA:e}"
    )
    time0 = time.time()
    res = minimize(
        fun=objective_function,  # objective function
        x0=np.ones(num_weights),  # initial wght_ratio guess
        jac=gradient_function,  # objective function gradient
        args=(A, b, REGULARIZATION_LAMBDA),  # fixed arguments
        method="L-BFGS-B",  # use L-BFGS-B solver
        bounds=Bounds(0.0, np.inf),  # consider only non-negative weight ratios
        options={
            "maxiter": OPTIMIZE_MAXITER,
            "ftol": OPTIMIZE_FTOL,
            "gtol": OPTIMIZE_GTOL,
            "iprint": OPTIMIZE_IPRINT,
        },
    )
    time1 = time.time()
    res_summary = (
        f">>> optimization execution time: {(time1-time0):.1f} secs"
        f"  iterations={res.nit}  success={res.success}\n"
        f">>> message: {res.message}\n"
        f">>> L-BFGS-B optimized loss value: {res.fun:.9e}"
    )
    print(res_summary)
    if OPTIMIZE_RESULTS:
        print(">>> full optimization results:\n", res)
    wght_ratio = res.x
    wght_area = res.x * wght_us
    loss = loss_function_value(wght_area, target_matrix, target_array)
    print(f"AREA-OPTIMIZED_LOSS_FUNCTION_VALUE= {loss:.9e}")
    weight_ratio_distribution(wght_ratio)

    if not write_file:
        return loss

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

    return loss


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
