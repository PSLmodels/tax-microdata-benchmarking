"""
Generate tmd/storage/output/cached_*.npy files for TAX_YEAR.
"""

import numpy as np
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER, CACHED_TAXCALC_VARIABLES

TAX_YEAR = 2021

INFILE_PATH = STORAGE_FOLDER / "output" / "tmd.csv.gz"
WTFILE_PATH = STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"
GFFILE_PATH = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"


def create_cached_files():
    """
    Create a Numpy binary file containing FIRST_YEAR values
    for each variable in the CACHED_TAXCALC_VARIABLES list.
    """
    # calculate all Tax-Calculator variables for TAX_YEAR
    pol = tc.Policy.tmd_constructor(growfactors=GFFILE_PATH)
    rec = tc.Records.tmd_constructor(
        data_path=INFILE_PATH,
        weights_path=WTFILE_PATH,
        growfactors=GFFILE_PATH,
        exact_calculations=True,
    )
    calc = tc.Calculator(policy=pol, records=rec)
    calc.advance_to_year(TAX_YEAR)
    calc.calc_all()

    # cache all variables to aid in areas examination
    calc.dataframe(variable_list=None, all_vars=True).to_csv(
        STORAGE_FOLDER / "output" / "cached_allvars.csv", index=None
    )

    # write each variable in CACHED_TAXCALC_VARIABLES list to a binary file
    for vname in CACHED_TAXCALC_VARIABLES:
        varray = calc.array(vname)
        fpath = STORAGE_FOLDER / "output" / f"cached_{vname}.npy"
        np.save(fpath, varray, allow_pickle=False)

    # provide timestamp for Makefile
    fpath = STORAGE_FOLDER / "output" / "cached_files"
    with open(fpath, "w", encoding="utf-8") as cfiles:
        cfiles.write("  ")

    return 0


if __name__ == "__main__":
    create_cached_files()
