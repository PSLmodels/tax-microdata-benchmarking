"""
Construct AREA_tmd_weights.csv.gz, a Tax-Calculator-style weights file
for 2021+ for the specified sub-national AREA.

AREA prefix for state areas are the two lower-case character postal codes.
AREA prefix for congressional districts are the state prefix followed by
two digits (with a leading zero) identifying the district.  Note there
are no district files for states with only one congressional district.
"""

import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import Bounds, minimize
from tmd.storage import STORAGE_FOLDER
from tmd.areas import AREAS_FOLDER


FIRST_YEAR = 2021
LAST_YEAR = 2074
VARFILE = STORAGE_FOLDER / "output" / "tmd.csv.gz"
POPFILE = STORAGE_FOLDER / "input" / "cbo_population_forecast.yaml"


def variable_matrix_and_target_array(area: str, vardf: pd.DataFrame):
    """
    Construct numpy 2-D variable matrix and 1-D targets array for
    specified area using specified vardf.
    """
    numobs = len(vardf)
    tdf = pd.read_csv(AREAS_FOLDER / "targets" / f"{area}_targets.csv")
    print("tdf.shape=", tdf.shape)
    vm_tuple = ()
    ta_list = []
    for row in tdf.itertuples(index=False):
        ta_list.append(row.target)
        # construct variable array for this target
        if row.count < 0 or row.count > 1:
            raise ValueError(f"count value {row.count} not in [0,1] range")
        if row.count == 0:
            unmasked_varray = vardf[row.varname]
        else:
            unmasked_varray = vardf[row.varname] > 0
        mask = np.ones(numobs, dtype=int)
        if row.scope < 0 or row.scope > 2:
            raise ValueError(f"scope value {row.scope} not in [0,2] range")
        if row.scope == 1:
            mask = mask * (vardf.data_source == 1)  # PUF records
        elif row.scope == 2:
            mask = mask * (vardf.data_source == 0)  # CPS records
        if row.agibin < 0 or row.agibin > 6:
            raise ValueError(f"agibin value {row.agibin} not in [0,6] range")
        if row.agibin > 0:
            mask = mask * (vardf.agi_bin == row.agibin)
        if not row.fstatus in (0, 1, 2, 4):
            raise ValueError(f"fstatus value {row.fstatus} not 1, 2, or 4")
        if row.fstatus > 0 and row.count != 1:
            raise ValueError(f"fstatus {row.fstatus} > 0 when count != 1")
        if row.fstatus == 1:
            mask = mask * (vardf.MARS == 1)  # single filer
        elif row.fstatus == 2:
            mask = mask * (vardf.MARS == 2)  # married filing jointly
        elif row.fstatus == 4:
            mask = mask * (vardf.MARS == 4)  # head of household filer
        masked_varray = mask * unmasked_varray
        vm_tuple = vm_tuple + (masked_varray,)
    # construct variable matrix and target array and return as tuple
    var_matrix = np.vstack(vm_tuple).T
    target_array = np.array(ta_list)
    return (var_matrix, target_array)


def func(wght, *args):
    """
    Function to be minimized when creating area weights.
    """
    var, target = args
    return jnp.mean(jnp.square(jnp.dot(wght, var) / target - 1))


FVAL_AND_FGRAD = jax.jit(jax.value_and_grad(func))


# -- High-level logic of the script:


def create_area_weights_file(area: str, initial_weights_scale: float):
    """
    Create Tax-Calculator-style weights file for FIRST_YEAR through LAST_YEAR
    for specified area and initial_weights_scale.
    """
    vdf = pd.read_csv(VARFILE)

    # construct initial weights array
    wght = vdf.s006 * initial_weights_scale

    # construct variable matrix and target array
    variable_matrix, target_array = variable_matrix_and_target_array(area, vdf)
    print(variable_matrix.shape)
    print(target_array.shape)
    print("dot=", np.dot(wght, variable_matrix) * 1e-6)
    print(f"POP0= {(wght * vdf.XTOT).sum()*1e-6:.3f}")


    ##c00100,     0,    1,     0,      0,  33e6

    # find wght that minimizes mean of squared relative wvar-to-target diffs
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
            "ftol": 1e-9,
            "gtol": 1e-9,
            "maxiter": 10,
            "disp": True,
        },
    )
    print(">>> scipy.minimize results:\n", res)
    num_neg = (res.x < 0).sum()
    assert num_neg == 0, f"num_negative_weights= {num_neg}"
    print(f"POP1= {(res.x * vdf.XTOT).sum()*1e-6:.3f}")

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
