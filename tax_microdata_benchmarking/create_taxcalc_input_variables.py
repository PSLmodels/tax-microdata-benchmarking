"""
Construct tmd.csv, a Tax-Calculator-style input variable file for 2021.
"""

import taxcalc as tc
from tax_microdata_benchmarking.datasets.tmd import create_tmd_2021
from tax_microdata_benchmarking.imputation_assumptions import (
    IMPUTATION_RF_RNG_SEED,
    IMPUTATION_BETA_RNG_SEED,
    W2_WAGES_SCALE,
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    REWEIGHT_DEVIATION_PENALTY,
)
from tax_microdata_benchmarking.storage import STORAGE_FOLDER


TAXYEAR = 2021


def create_variable_file(write_file=True):
    """
    Create Tax-Calculator-style input variable file for TAXYEAR.
    """
    # construct dataframe containing input and output variables
    print(f"Creating {TAXYEAR} PUF+CPS file assuming:")
    print(f"  IMPUTATION_RF_RNG_SEED = {IMPUTATION_RF_RNG_SEED}")
    print(f"  IMPUTATION_BETA_RNG_SEED = {IMPUTATION_BETA_RNG_SEED}")
    print(f"  ASSUMED W2_WAGES_SCALE = {W2_WAGES_SCALE:.5f}")
    print(f"  REWEIGHT_MULTIPLIER_MIN = {REWEIGHT_MULTIPLIER_MIN:.3f}")
    print(f"  REWEIGHT_MULTIPLIER_MAX = {REWEIGHT_MULTIPLIER_MAX:.3f}")
    print(f"  REWEIGHT_DEVIATION_PENALTY = {REWEIGHT_DEVIATION_PENALTY:.3f}")
    vdf = create_tmd_2021()
    vdf.FLPDYR = TAXYEAR
    weights = vdf.s006.copy()
    if write_file:
        # save a copy containing both input and output variables
        fname = STORAGE_FOLDER / "output" / "tmd_2021.csv"
        print(f"Writing PUF+CPS file... [{fname}]")
        vdf.to_csv(fname, index=False)
    # streamline dataframe so that it includes only input variables
    print("Removing output variables from PUF+CPS DataFrame...")
    rec = tc.Records(
        data=vdf,
        start_year=TAXYEAR,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
        exact_calculations=True,
    )
    vdf.drop(columns=rec.IGNORED_VARS, inplace=True)
    # round all float variables to nearest integer except for weights
    vdf = vdf.astype(int)
    vdf.s006 = weights
    for var in ["e00200", "e00900", "e02100"]:
        vdf[var] = vdf[f"{var}p"] + vdf[f"{var}s"]
    # write input-variables-only dataframe to CSV-formatted file
    if write_file:
        fname = STORAGE_FOLDER / "output" / "tmd.csv.gz"
        print(f"Writing PUF+CPS file... [{fname}]")
        vdf.to_csv(fname, index=False, float_format="%.2f")


if __name__ == "__main__":
    create_variable_file()
