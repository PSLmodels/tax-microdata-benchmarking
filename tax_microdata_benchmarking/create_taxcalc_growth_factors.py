"""
Construct tmd_growfactors.csv, a Tax-Calculator-style GrowFactors file that
extends through LAST_YEAR from the puf_growfactors.csv file, which is a
copy of the most recent growfactors.csv file in the Tax-Calculator repository.
"""

import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER

FIRST_YEAR = 2021
LAST_YEAR = 2074
PGFFILE = STORAGE_FOLDER / "input" / "puf_growfactors.csv"
TGFFILE = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"


def create_factors_file():
    """
    Create Tax-Calculator-style factors file for FIRST_YEAR through LAST_YEAR.
    """
    # read PUF-factors from PGFFILE
    gfdf = pd.read_csv(PGFFILE)
    first_puf_year = gfdf.YEAR.iat[0]
    last_puf_year = gfdf.YEAR.iat[-1]

    # drop gfdf rows before FIRST_YEAR
    drop_row_index_list = range(0, FIRST_YEAR - first_puf_year)
    gfdf.drop(drop_row_index_list, inplace=True)

    # set all FIRST_YEAR growfactors to one
    gfdf.iloc[0, 1:] = 1.0

    # add rows thru LAST_YEAR by copying values for last year in PUF file
    if LAST_YEAR > last_puf_year:
        last_row = gfdf.iloc[-1, :].copy()
        num_rows = LAST_YEAR - last_puf_year
        added = pd.DataFrame([last_row] * num_rows, columns=gfdf.columns)
        for idx in range(0, num_rows):
            added.YEAR.iat[idx] = last_puf_year + idx + 1
        gfdf = pd.concat([gfdf, added], ignore_index=True)

    # write gfdf to CSV-formatted file
    gfdf.YEAR = gfdf.YEAR.astype(int)
    gfdf.to_csv(TGFFILE, index=False)


if __name__ == "__main__":
    create_factors_file()
