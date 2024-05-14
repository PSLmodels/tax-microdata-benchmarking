import streamlit as st

st.set_page_config(layout="wide")

from utils import puf_pe_21, pe_21, td_23
from tax_microdata_benchmarking.utils.taxcalc import (
    get_tc_variable_description,
    get_tc_is_input,
)
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import pandas as pd
import numpy as np

st.title("AGI breakdowns")

st.write(
    "This page shows how the datasets compare on the distribution of key variables by AGI band."
)

agi_targets_path = STORAGE_FOLDER / "input" / "agi_targets.csv"

agi_targets = pd.read_csv(agi_targets_path)

# First, just compare AGI totals and nret

INCOME_RANGES = [
    -np.inf,
    1,
    5e3,
    1e4,
    1.5e4,
    2e4,
    2.5e4,
    3e4,
    4e4,
    5e4,
    7.5e4,
    1e5,
    2e5,
    5e5,
    1e6,
    1.5e6,
    2e6,
    5e6,
    1e7,
    np.inf,
]


def agi_comparisons(
    irs_variable: str,
    tc_variable: str,
    is_count: bool,
    taxable: bool = True,
) -> pd.DataFrame:
    """
    Compare the distribution of a variable by AGI band across datasets.

    Args:
        irs_variable (str): The IRS variable to compare.
        tc_variable (str): The Tax-Calculator variable to compare.
        is_count (bool): Whether the variable is a count or a total.

    Returns:
        DataFrame with the comparisons.
    """

    irs_subset = agi_targets[agi_targets.year == 2021][
        agi_targets.vname == irs_variable
    ][agi_targets.datatype == "taxable" if taxable else "filers"]
    divisor = 1e3 if is_count else 1e9
    comparisons = pd.DataFrame()
    if taxable:
        puf_pe_21_ = puf_pe_21[puf_pe_21.iitax - puf_pe_21.c07100 > 0]
        td_23_ = td_23[td_23.iitax - td_23.c07100 > 0]
    for i in list(range(len(INCOME_RANGES) - 1)) + ["all"]:
        if i == "all":
            irs_value = (
                irs_subset[irs_subset.incsort == 1].ptarget.values[0]
                * (1e3 if not is_count else 1)
                / divisor
            )
            lower = -np.inf
            upper = np.inf
        else:
            irs_value = (
                irs_subset[irs_subset.incsort - 2 == i].ptarget.values[0]
                * (1e3 if not is_count else 1)
                / divisor
            )
            lower = INCOME_RANGES[i]
            upper = INCOME_RANGES[i + 1]
        puf_pe_filter = (puf_pe_21_.c00100 >= lower) & (
            puf_pe_21_.c00100 < upper
        )
        td_filter = (td_23_.c00100 >= lower) & (td_23_.c00100 < upper)
        if is_count:
            puf_pe_value = puf_pe_21_[puf_pe_filter].s006.sum() / divisor
            td_value = td_23_[td_filter].s006.sum() / divisor
        else:
            puf_pe_value = (
                puf_pe_21_[puf_pe_filter][tc_variable]
                * puf_pe_21_[puf_pe_filter].s006
            ).sum() / divisor
            td_value = (
                td_23_[td_filter][tc_variable] * td_23_[td_filter].s006
            ).sum() / divisor

        comparisons = pd.concat(
            [
                comparisons,
                pd.DataFrame(
                    {
                        "lower AGI": [lower],
                        "upper AGI": [upper],
                        "irs_21": [irs_value],
                        "puf_pe_21": [puf_pe_value],
                        "td_23": [td_value],
                    }
                ),
            ]
        )

    COLUMNS_TO_ROUND = ["puf_pe_21", "td_23", "irs_21"]

    for column in COLUMNS_TO_ROUND:
        comparisons[column] = comparisons[column].apply(
            lambda x: round(x, 1) if x is not None else None
        )

    comparisons["pufpe / irs (%)"] = (
        (comparisons["puf_pe_21"] / comparisons["irs_21"].fillna(0) - 1)
        .fillna(0)
        .replace(np.inf, 0)
        * 100
    ).round(1)

    comparisons["td / irs (%)"] = (
        (comparisons["td_23"] / comparisons["irs_21"].fillna(0) - 1)
        .fillna(0)
        .replace(np.inf, 0)
        * 100
    ).round(1)

    comparisons["closest to IRS"] = np.select(
        [
            comparisons.irs_21.isna(),
            comparisons["pufpe / irs (%)"].abs()
            < comparisons["td / irs (%)"].abs(),
            comparisons["pufpe / irs (%)"].abs()
            > comparisons["td / irs (%)"].abs(),
            True,
        ],
        [
            "Unknown",
            "puf_pe_21",
            "td_23",
            "Neither",
        ],
    )

    return comparisons


agi = agi_comparisons(
    "agi",
    "c00100",
    False,
)

st.subheader("AGI totals, taxable returns")

st.dataframe(agi)

st.subheader("Number of taxable returns")

nret = agi_comparisons(
    "nret_all",
    "s006",
    True,
)

st.dataframe(nret)
