"""
Construct tmd_growfactors.csv, a Tax-Calculator-style GrowFactors file that
covers the years from FIRST_YEAR through LAST_YEAR.
"""

import pandas as pd
import numpy as np
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR

FIRST_YEAR = TAXYEAR
LAST_YEAR = TAXYEAR + 53

AWAGE_INDEX = 6
ASCHCI_INDEX = 7
ASCHEI_INDEX = 9
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

    if TAXYEAR == 2021:
        # Adjust some factors in order to calibrate tax revenue after 2021
        # ... adjust 2022 PUF factors to get closer to published 2022 targets:
        # ...  calendar year 2022 targets from
        # ...  "Individual Income Tax Returns, Preliminary Data, Tax Year 2022"
        # ...  by Michael Parisi, SOI Bulletin (Spring 2024),
        # ...  which is at this URL:
        # ...  https://www.irs.gov/pub/irs-soi/soi-a-inpre-id2401.pdf
        # These adjustments calibrate 2021-to-2022 growth factors.
        # When FIRST_YEAR >= 2022, the 2022 row is the baseline (all ones)
        # and these adjustments don't apply because reweighting handles the
        # 2022 calibration directly via use of 2022 SOI targets).
        gfdf.iat[2022 - FIRST_YEAR, AWAGE_INDEX] += -0.01
        gfdf.iat[2022 - FIRST_YEAR, ADIVS_INDEX] += +0.04
        gfdf.iat[2022 - FIRST_YEAR, ACGNS_INDEX] += +0.04
        gfdf.iat[2022 - FIRST_YEAR, ASCHCI_INDEX] += -0.05
        gfdf.iat[2022 - FIRST_YEAR, ASCHEI_INDEX] += +0.07
        gfdf.iat[2022 - FIRST_YEAR, AUCOMP_INDEX] += -0.01
        gfdf.iat[2022 - FIRST_YEAR, ASOCSEC_INDEX] += +0.10

    # add rows thru LAST_YEAR by copying values for last year in PUF gf file
    if LAST_YEAR > last_puf_year:
        last_row = gfdf.iloc[-1, :].copy()
        num_rows = LAST_YEAR - last_puf_year
        added = pd.DataFrame([last_row] * num_rows, columns=gfdf.columns)
        # ensure shape and index are as expected
        added = added.reset_index(drop=True)  # guardrail
        added.loc[:, "YEAR"] = np.arange(
            last_puf_year + 1, last_puf_year + 1 + num_rows
        )
        gfdf = pd.concat([gfdf, added], ignore_index=True)

    # write gfdf to CSV-formatted file
    gfdf.YEAR = gfdf.YEAR.astype(int)
    gfdf.to_csv(TGFFILE, index=False, float_format="%.6f")


if __name__ == "__main__":
    create_factors_file()
