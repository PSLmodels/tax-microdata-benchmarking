"""
Construct tmd_growfactors.csv, a Tax-Calculator-style GrowFactors file that
covers the years from FIRST_YEAR through LAST_YEAR.
"""

import pandas as pd
import numpy as np
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
    # ...  calendar year 2022 targets from
    # ...  "Individual Income Tax Returns, Preliminary Data, Tax Year 2022"
    # ...  by Michael Parisi, SOI Bulletin (Spring 2024),
    # ...  which is at this URL:
    # ...  https://www.irs.gov/pub/irs-soi/soi-a-inpre-id2401.pdf
    gfdf.iat[2022 - FIRST_YEAR, AWAGE_INDEX] += -0.01
    gfdf.iat[2022 - FIRST_YEAR, ADIVS_INDEX] += +0.04
    gfdf.iat[2022 - FIRST_YEAR, ACGNS_INDEX] += +0.04
    gfdf.iat[2022 - FIRST_YEAR, ASCHCI_INDEX] += -0.05
    gfdf.iat[2022 - FIRST_YEAR, ASCHEI_INDEX] += +0.07
    gfdf.iat[2022 - FIRST_YEAR, AUCOMP_INDEX] += -0.01
    gfdf.iat[2022 - FIRST_YEAR, ASOCSEC_INDEX] += +0.10
    # ... using above adjustments, we have the following results:
    #      2022 AMOUNTS in $B         SOI        TMD   (TMD/SOI-1)*100(%)
    #      wage_and_salary       9648.553   9654.475   +0.1
    #      ordin_dividends        420.403    420.299   -0.0
    #      SchC_net_income        395.136    396.303   +0.3
    #      SchE+partner_Scorp    1108.445   1114.474   +0.5
    #      unemploy_compen         29.554     29.909   +1.2
    #      taxable_soc_sec        471.017    473.755   +0.6
    #      adj_gross_income     15142.763  14851.081   -1.9
    #      taxable_income       11954.522  11842.505   -0.9
    #      refundable_credits     106.380    116.717   +9.7
    #      itax_liability        2285.496   2289.792   +0.2
    # where itax_liability is calculated using only PUF-derived records
    # and where negative itax amounts are ignored

    # add rows thru LAST_YEAR by copying values for last year in PUF file
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
