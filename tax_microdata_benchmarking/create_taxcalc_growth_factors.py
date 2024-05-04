"""
Construct tmd_growfactors.csv, a Tax-Calculator-style GrowFactors file for
the years 2021+ from the puf_growfactors.csv file, which is a copy of the
most recent growfactors.csv file in the Tax-Calculator repository.
"""

import pandas as pd

FIRST_YEAR = 2021
LAST_YEAR = 2034
PGFFILE = "puf_growfactors.csv"
TGFFILE = "tmd_growfactors.csv"


def create_factors_file():
    """
    Create Tax-Calculator-style factors file for FIRST_YEAR through LAST_YEAR.
    """
    # read PUF-factors from PGFFILE
    gfdf = pd.read_csv(PGFFILE)
    first_puf_year = gfdf.YEAR.iat[0]
    last_puf_year = gfdf.YEAR.iat[-1]

    # remove rows before FIRST_YEAR
    indexes = [yr - first_puf_year for yr in range(first_puf_year, FIRST_YEAR)]
    gfdf.drop(index=indexes, inplace=True)

    # set all FIRST_YEAR values to one
    gfdf.iloc[0, :] = 1.0
    gfdf.YEAR.iat[0] = FIRST_YEAR

    # add rows thru LAST_YEAR by copying values for last year in PUF file
    if LAST_YEAR > last_puf_year:
        idx = last_puf_year - FIRST_YEAR
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
