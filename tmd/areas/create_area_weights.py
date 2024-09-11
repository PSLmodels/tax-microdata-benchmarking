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
from scipy.optimize import lsq_linear
from scipy.sparse import vstack, eye, csr_matrix, diags
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
# wide bounds settings:
# OPTIMIZE_LO_BOUND_RATIO = 2e-6  # must be greater than 1e-6
# OPTIMIZE_HI_BOUND_RATIO = 9e+9  # must be less than np.inf
# narrow bounds settings:
OPTIMIZE_LO_BOUND_RATIO = 1e-2  # must be greater than 1e-6
OPTIMIZE_HI_BOUND_RATIO = 1e2  # must be less than np.inf
# tiny bounds settings:
# OPTIMIZE_LO_BOUND_RATIO = 1e-1  # must be greater than 1e-6
# OPTIMIZE_HI_BOUND_RATIO = 1e+1  # must be less than np.inf
OPTIMIZE_RATIOS = True  # True produces much better optimization results
OPTIMIZE_REGULARIZATION = True # add regularization term to lsq_linear to penalize (x - a)^2
REGULARIZATION_LAMBDA = 1.0 # nonnegative: how much emphasis to put on regularization: 0 = none
OPTIMIZE_FTOL = 1e-10
OPTIMIZE_MAXITER = 5000
OPTIMIZE_VERBOSE = 0  # set to zero for no iteration information
OPTIMIZE_RESULTS = False  # set to True to see complete lsq_linear results


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
    Construct numpy 2-D variable matrix and 1-D targets array for
    specified area using specified vardf.  Also, compute initial
    weights scaling factor for specified area.  Return all three
    as a tuple.
    """
    national_population = (vardf.s006 * vardf.XTOT).sum()
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
        vm_tuple = vm_tuple + (scaled_masked_varray,)
    # construct variable matrix and target array and return as tuple
    var_matrix = np.vstack(vm_tuple).T
    target_array = np.array(ta_list)
    return (
        var_matrix,
        target_array,
        initial_weights_scale,
    )


def loss_function_value(wght, variable_matrix, target_array):
    """
    Return loss function value given specified arguments.
    """
    act = np.dot(wght, variable_matrix)
    act_minus_exp = act - target_array
    if DUMP_LOSS_FUNCTION_VALUE_COMPONENTS:
        for tnum in range(1, len(target_array) + 1):
            print(
                f"TARGET{tnum:03d}:ACT-EXP,ACT/EXP= "
                f"{act_minus_exp[tnum - 1]:16.9e}, "
                f"{(act[tnum - 1] / target_array[tnum - 1]):.3f}"
            )
    return 0.5 * np.sum(np.square(act_minus_exp))


def weight_ratio_distribution(ratio):
    """
    Print distribution of post-optimized to pre-optimized weight ratios.
    """
    eps = 1e-6
    bins = [
        0.0,
        OPTIMIZE_LO_BOUND_RATIO - eps,
        OPTIMIZE_LO_BOUND_RATIO + eps,
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
    print(f"DISTRIBUTION OF POST/PRE WEIGHT RATIO (n={tot}):")
    print(f"  with OPTIMIZE_LO_BOUND_RATIO= {OPTIMIZE_LO_BOUND_RATIO:e}")
    print(f"   and OPTIMIZE_HI_BOUND_RATIO= {OPTIMIZE_HI_BOUND_RATIO:e}")
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


# -- High-level logic of the script:


def create_area_weights_file(area: str, write_file: bool = True):
    """
    Create Tax-Calculator-style weights file for FIRST_YEAR through LAST_YEAR
    for specified area using information in area targets CSV file.
    Return loss_function_value using the optimized weights and optionally
    write the weights file.
    """
    # construct variable matrix and target array and weights_scale
    vdf = all_taxcalc_variables()
    variable_matrix, target_array, weights_scale = prepared_data(area, vdf)
    wght = np.array(vdf.s006 * weights_scale)
    num_weights = len(wght)
    num_targets = len(target_array)
    print(f"USING {area}_targets.csv FILE CONTAINING {num_targets} TARGETS")
    loss = loss_function_value(wght, variable_matrix, target_array)
    print(f"US_PROPORTIONALLY_SCALED_LOSS_FUNCTION_VALUE= {loss:.9e}")
    density = np.count_nonzero(variable_matrix) / variable_matrix.size
    print(f"variable_matrix sparsity ratio = {(1.0 - density):.3f}")

    # optimize weights by minimizing sum of squared wght*var-target deviations
    if OPTIMIZE_RATIOS:
        var_matrix = (variable_matrix * wght[:, np.newaxis]).T
        lob = OPTIMIZE_LO_BOUND_RATIO * np.ones(num_weights)
        hib = OPTIMIZE_HI_BOUND_RATIO * np.ones(num_weights)
    else:  # if optimizing the weights
        var_matrix = variable_matrix.T
        lob = OPTIMIZE_LO_BOUND_RATIO * wght
        hib = OPTIMIZE_HI_BOUND_RATIO * wght
    print(
        f"OPTIMIZE_RATIOS= {OPTIMIZE_RATIOS}"
        f"OPTIMIZE_REGULARIZATION= {OPTIMIZE_REGULARIZATION}"
        f"   var_matrix.shape= {var_matrix.shape}\n"
        f"OPTIMIZE_LO_BOUND_RATIO= {OPTIMIZE_LO_BOUND_RATIO:e}\n"
        f"OPTIMIZE_HI_BOUND_RATIO= {OPTIMIZE_HI_BOUND_RATIO:e}"
    )
    time0 = time.time()
    if not OPTIMIZE_REGULARIZATION:
        res = lsq_linear(
            var_matrix,
            target_array,
            bounds=(lob, hib),
            method="bvls",
            tol=OPTIMIZE_FTOL,
            lsq_solver="exact",
            lsmr_tol=None,
            max_iter=OPTIMIZE_MAXITER,
            verbose=OPTIMIZE_VERBOSE,
        )
    elif OPTIMIZE_REGULARIZATION and OPTIMIZE_RATIOS:
        print("adding regularization term")
        # use sparse matrices       
        # res = lsq_linear(
        #     var_matrix,
        #     target_array,
        #     bounds=(lob, hib),
        #     method="bvls",
        #     tol=OPTIMIZE_FTOL,
        #     lsq_solver="exact",
        #     lsmr_tol=None,
        #     max_iter=OPTIMIZE_MAXITER,
        #     verbose=OPTIMIZE_VERBOSE,
        # )        
        
        # Step 1: convert to sparse arrays
        variable_matrix_sparse = csr_matrix(variable_matrix)
        # Perform element-wise multiplication with wght, broadcasting wght appropriately
        A_sparse = csr_matrix(variable_matrix_sparse.multiply(wght[:, np.newaxis]).T)
        # Step 2: Create the regularization matrix and augment A and b
        m = A_sparse.shape[1]  # Number of records
        I = eye(m, format='csr')  # Identity matrix for regularization (size m x m)
        
        # Augment the scaled A and b for regularization (penalizing x - 1)
        A_aug = vstack([csr_matrix(A_sparse), np.sqrt(REGULARIZATION_LAMBDA) * I])  # Augmented matrix
        b_aug = np.hstack([target_array, np.sqrt(REGULARIZATION_LAMBDA) * np.ones(m)])  # Augmented target
        
        res = lsq_linear(
            A_aug, 
            b_aug, 
            bounds=(lob, hib),
            max_iter = OPTIMIZE_MAXITER, 
            lsmr_tol='auto', 
            verbose=OPTIMIZE_VERBOSE)
        
    time1 = time.time()
    res_summary = (
        f">>> scipy.lsq_linear execution: {(time1-time0):.1f} secs"
        f"  iterations={res.nit}  success={res.success}\n"
        f">>> message: {res.message}\n"
        f">>> lsq_linear optimized loss value: {res.cost:.9e}"
    )
    print(res_summary)
    if OPTIMIZE_RESULTS:
        print(">>> scipy.lsq_linear full results:\n", res)
    if OPTIMIZE_RATIOS:
        wghtx = res.x * wght
    else:
        wghtx = res.x
    loss = loss_function_value(wghtx, variable_matrix, target_array)
    print(f"AREA-OPTIMIZED_LOSS_FUNCTION_VALUE= {loss:.9e}")
    weight_ratio_distribution(wghtx / wght)
    print(f"Sum of original weights= {sum(wght):,.0f}")
    print(f"Sum of new weights=      {sum(wghtx):,.0f}")

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
