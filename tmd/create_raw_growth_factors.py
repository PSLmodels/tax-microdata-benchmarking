"""
Construct raw growfactors.csv, a Tax-Calculator-style GrowFactors file.
"""

import pandas as pd
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR

INPUT_FILE = {
    2021: "taxdata25_growfactors.csv",
    2022: "taxdata26_growfactors.csv",
}
INFILE = STORAGE_FOLDER / "input" / INPUT_FILE[TAXYEAR]
OUTFILE = STORAGE_FOLDER / "output" / "growfactors.csv"


def create_raw_factors_file():
    """
    Create Tax-Calculator-style raw growth factors file.
    """
    # read PUF-factors from INFILE
    gfdf = pd.read_csv(INFILE)

    # write gfdf to CSV-formatted file
    gfdf.YEAR = gfdf.YEAR.astype(int)
    gfdf.to_csv(OUTFILE, index=False, float_format="%.6f")


if __name__ == "__main__":
    create_raw_factors_file()
