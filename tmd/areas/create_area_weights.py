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
import jax
import jax.numpy as jnp
from scipy.optimize import Bounds, minimize
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
DUMP_MINIMIZE_ITERS = False
DUMP_MINIMIZE_RES = True


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
    return vdf


def prepared_data(area: str, vardf: pd.DataFrame):
    """
    Construct numpy 2-D variable matrix and 1-D targets array for
    specified area using specified vardf.  Also, compute initial
    weights scaling factor for specified area.  Return all three
    as a tuple.
    """
    numobs = len(vardf)
    tdf = pd.read_csv(AREAS_FOLDER / "targets" / f"{area}_targets.csv")
    vm_tuple = ()
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
            uspop = 334.181e6  # tmd/storage/input/cbo_population_forecast.yaml
            initial_weights_scale = row.target / uspop
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
        vm_tuple = vm_tuple + (scaled_masked_varray,)
    # construct variable matrix and target array and return as tuple
    var_matrix = np.vstack(vm_tuple).T
    target_array = np.array(ta_list)
    return (
        var_matrix,
        target_array,
        initial_weights_scale,
    )


def loss_function_value_and_gradient(wght, *args):
    """
    Return loss function value and its gradient as a tuple using jnp functions.
    """
    var, target, var_transpose = args
    act_minus_exp = jnp.dot(wght, var) - target
    fval = jnp.sum(jnp.square(act_minus_exp))
    grad = jnp.dot(2.0 * act_minus_exp, var_transpose)
    return fval, grad


FVAL_AND_FGRAD = jax.jit(loss_function_value_and_gradient)


def loss_function_value(wght, variable_matrix, target_array):
    """
    Return loss function value using numpy functions.
    """
    act_minus_exp = np.dot(wght, variable_matrix) - target_array
    if DUMP_LOSS_FUNCTION_VALUE_COMPONENTS:
        for tnum in range(1, len(target_array) + 1):
            print(f"ACT-EXP:{tnum}= {act_minus_exp[tnum - 1]:16.9e}")
    return np.sum(np.square(act_minus_exp))


# -- High-level logic of the script:


def create_area_weights_file(area: str):
    """
    Create Tax-Calculator-style weights file for FIRST_YEAR through LAST_YEAR
    for specified area using information in area targets CSV file.
    """
    # construct variable matrix and target array and weights_scale
    vdf = all_taxcalc_variables()
    variable_matrix, target_array, weights_scale = prepared_data(area, vdf)
    wght = vdf.s006 * weights_scale

    print(f"NUMBER_OF_TARGETS = {len(target_array)}")
    loss = loss_function_value(wght, variable_matrix, target_array)
    print(f"LOSS_FUNCTION_VALUE= {loss:.9e}")

    # find wght that minimizes mean of squared relative wvar-to-target diffs
    time0 = time.time()
    res = minimize(
        FVAL_AND_FGRAD,
        wght,
        args=(
            variable_matrix,
            target_array,
            variable_matrix.T,
        ),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(lb=0.0, ub=np.inf),
        options={
            "ftol": 1e-30,
            "gtol": 1e-12,
            "maxiter": 1000,
            "disp": DUMP_MINIMIZE_ITERS,
        },
    )
    time1 = time.time()
    res_summary = (
        f">>> scipy.minimize execution: {(time1-time0):.1f} secs"
        f"  iterations={res.nit}  success={res.success}"
    )
    print(res_summary)
    if DUMP_MINIMIZE_RES:
        print(">>> scipy.minimize all results:\n", res)
    wghtx = res.x
    num_neg = (wghtx < 0).sum()
    assert num_neg == 0, f"num_negative_weights= {num_neg}"

    print(f"# units in total is {len(wghtx)}")
    print(f"# units with post weight equal to zero is {(wghtx == 0).sum()}")
    for multiplier in range(1, 5):
        wght_rchg = 2.0 * multiplier
        num_inc = ((wghtx / wght) > wght_rchg).sum()
        print(f"# units with post/pre weight ratio > {wght_rchg} is {num_inc}")
    loss = loss_function_value(wghtx, variable_matrix, target_array)
    print(f"LOSS_FUNCTION_VALUE= {loss:.9e}")

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

    return 0


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
    sys.exit(create_area_weights_file(area_code))
