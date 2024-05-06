"""
Construct growfactors.csv, a Tax-Calculator-style GrowFactors file that
extends through LAST_YEAR from the puf_growfactors.csv file, which is a
copy of the most recent growfactors.csv file in the Tax-Calculator repository.
"""

import pandas as pd

LAST_YEAR = 2074
PGFFILE = "puf_growfactors.csv"
TGFFILE = "growfactors.csv"


def create_factors_file():
    """
    Create Tax-Calculator-style factors file for FIRST_YEAR through LAST_YEAR.
    """
    # read PUF-factors from PGFFILE
    gfdf = pd.read_csv(PGFFILE)
    last_puf_year = gfdf.YEAR.iat[-1]

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
