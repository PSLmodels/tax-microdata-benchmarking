"""
Construct tmd.csv, a Tax-Calculator-style input variable file for TAXYEAR.
"""

import taxcalc as tc
from tmd.datasets.tmd import create_tmd_dataframe
from tmd.imputation_assumptions import (
    TAXYEAR,
    IMPUTATION_RF_RNG_SEED,
    IMPUTATION_BETA_RNG_SEED,
    ITMDED_GROW_RATE,
    W2_WAGES_SCALE,
    CPS_WEIGHTS_SCALE,
    REWEIGHT_DEVIATION_PENALTY,
)
from tmd.storage import STORAGE_FOLDER

DUMP_ALL_UNROUNDED_VARIABLES = False


def create_variable_file(write_file=True):
    """
    Create Tax-Calculator-style input variable file for TAXYEAR.
    """
    # construct dataframe containing input and output variables
    print(f"Creating {TAXYEAR} PUF+CPS file assuming:")
    print(f"  IMPUTATION_RF_RNG_SEED = {IMPUTATION_RF_RNG_SEED}")
    print(f"  IMPUTATION_BETA_RNG_SEED = {IMPUTATION_BETA_RNG_SEED}")
    print(f"  ASSUMED ITMDED_GROW_RATE = {ITMDED_GROW_RATE:.3f}")
    print(f"  ASSUMED W2_WAGES_SCALE = {W2_WAGES_SCALE:.5f}")
    print(f"  WEIGHT_DEVIATION_PENALTY = {REWEIGHT_DEVIATION_PENALTY:.3f}")
    print(f"  ASSUMED CPS_WEIGHTS_SCALE = {CPS_WEIGHTS_SCALE[TAXYEAR]:.2f}")
    vdf = create_tmd_dataframe(TAXYEAR)
    vdf.FLPDYR = TAXYEAR
    vdf.agi_bin = 0
    # optionally dump all input and output variables unrounded
    if write_file and DUMP_ALL_UNROUNDED_VARIABLES:
        fname = STORAGE_FOLDER / "allvars_unrounded_2021.csv"
        print(f"Writing unrounded PUF+CPS file... [{fname}]")
        vdf.to_csv(fname, index=False)
    # streamline dataframe so that it includes only input variables
    print("Removing output variables from PUF+CPS dataframe...")
    rec = tc.Records(
        data=vdf,
        start_year=TAXYEAR,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
        exact_calculations=True,
        weights_scale=1.0,
    )
    vdf.drop(columns=rec.IGNORED_VARS, inplace=True)
    # round all float variables to nearest integer except for weights
    weights = vdf.s006.copy()
    vdf = vdf.astype(int)
    vdf.s006 = weights
    for var in ["e00200", "e00900", "e02100"]:
        vdf[var] = vdf[f"{var}p"] + vdf[f"{var}s"]
    # write input-variables-only dataframe to CSV-formatted file
    if write_file:
        fname = STORAGE_FOLDER / "output" / "tmd.csv.gz"
        print(f"Writing PUF+CPS file... [{fname}]")
        vdf.to_csv(fname, index=False, float_format="%.5f")


if __name__ == "__main__":
    create_variable_file()
