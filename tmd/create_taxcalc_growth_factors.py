"""
Construct tmd_growfactors.csv, a Tax-Calculator-style GrowFactors file that
covers the years from FIRST_YEAR through LAST_YEAR.
"""

import pandas as pd
from tmd.storage import STORAGE_FOLDER


FIRST_YEAR = 2021
LAST_YEAR = 2074

AWAGE_INDEX = 6
ASCHCI_INDEX = 7
ASCHEI_INDEX = 9
AINTS_INDEX = 11
ADIVS_INDEX = 12
ACGNS_INDEX = 13
ASOCSEC_INDEX = 14
AUCOMP_INDEX = 15


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

    # adjust some factors in order to calibrate tax revenue after FIRST_YEAR
    # ... adjust 2022 PUF factors to get closer to published 2022 targets:
    gfdf.iat[2022 - FIRST_YEAR, AWAGE_INDEX] += -0.01
    gfdf.iat[2022 - FIRST_YEAR, ADIVS_INDEX] += +0.04
    gfdf.iat[2022 - FIRST_YEAR, ACGNS_INDEX] += +0.04
    gfdf.iat[2022 - FIRST_YEAR, ASCHCI_INDEX] += -0.05
    gfdf.iat[2022 - FIRST_YEAR, ASCHEI_INDEX] += +0.07
    gfdf.iat[2022 - FIRST_YEAR, AUCOMP_INDEX] += -0.01
    gfdf.iat[2022 - FIRST_YEAR, ASOCSEC_INDEX] += +0.10

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
    gfdf.to_csv(TGFFILE, index=False, float_format="%.6f")


if __name__ == "__main__":
    create_factors_file()
