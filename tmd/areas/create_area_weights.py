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


def variable_matrix_and_target_array(area: str, vardf: pd.DataFrame):
    """
    Construct numpy 2-D variable matrix and 1-D targets array for
    specified area using specified vardf.
    """
    numobs = len(vardf)
    tdf = pd.read_csv(AREAS_FOLDER / "targets" / f"{area}_targets.csv")
    vm_tuple = ()
    ta_list = []
    row_num = 1
    for row in tdf.itertuples(index=False):
        row_num += 1
        line = f"{area}:L{row_num}"
        unscaled_target = row.target
        zero_unscaled_target = bool(unscaled_target == 0)
        if zero_unscaled_target:
            unscaled_target = 1.0
        scale = 1.0 / unscaled_target
        scaled_target = unscaled_target * scale
        ta_list.append(scaled_target)
        # construct variable array for this target
        assert (
            row.count >= 0 and row.count <= 1
        ), f"count value {row.count} not in [0,1] range on {line}"
        if row.count == 0:
            unmasked_varray = vardf[row.varname]
        else:
            unmasked_varray = vardf[row.varname] > 0
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
    )


def loss_function(wght, *args):
    """
    Function to be minimized when creating area weights.
    """
    var, target = args
    return jnp.mean(jnp.square(jnp.dot(wght, var) / target - 1))


FVAL_AND_FGRAD = jax.jit(jax.value_and_grad(loss_function))


# -- High-level logic of the script:


def create_area_weights_file(area: str, initial_weights_scale: float):
    """
    Create Tax-Calculator-style weights file for FIRST_YEAR through LAST_YEAR
    for specified area and initial_weights_scale.
    """
    # compute all Tax-Calculator variables
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

    # construct initial weights array
    wght = vdf.s006 * initial_weights_scale

    # construct variable matrix and target array
    variable_matrix, target_array = variable_matrix_and_target_array(area, vdf)
    # print(variable_matrix.shape)
    # print(target_array.shape)
    # print("dot=", np.dot(wght, variable_matrix) * 1e-6)
    print(f"POP0= {(wght * vdf.XTOT).sum()*1e-6:.3f}")
    print(f"AGI0= {(wght * vdf.c00100).sum()*1e-9:.3f}")

    # find wght that minimizes mean of squared relative wvar-to-target diffs
    time0 = time.time()
    res = minimize(
        FVAL_AND_FGRAD,
        wght,
        args=(
            variable_matrix,
            target_array,
        ),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(lb=0.0, ub=np.inf),
        options={
            "ftol": 1e-12,
            "gtol": 1e-12,
            "maxiter": 200,
            "disp": True,
        },
    )
    time1 = time.time()
    print(f">>> scipy.minimize exectime= {(time1-time0):.1f} secs")
    print(">>> scipy.minimize results:\n", res)
    num_neg = (res.x < 0).sum()
    assert num_neg == 0, f"num_negative_weights= {num_neg}"
    for scale in range(1, 4):
        wght_rchg = 2.0 * scale
        num_inc = ((res.x / wght) > wght_rchg).sum()
        print(f"# units with pst/pre weight ratio > {wght_rchg} is {num_inc}")
    print(f"POP1= {(res.x * vdf.XTOT).sum()*1e-6:.3f}")
    print(f"AGI1= {(res.x * vdf.c00100).sum()*1e-9:.3f}")

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
    sys.exit(create_area_weights_file("xx", 0.1))
