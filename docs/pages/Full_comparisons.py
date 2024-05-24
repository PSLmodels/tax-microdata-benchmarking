import streamlit as st

st.set_page_config(layout="wide")

from utils import puf_pe_21, pe_21, td_23, TC_NAME_TO_IRS_NAMES, agi_targets
from tax_microdata_benchmarking.utils.taxcalc import (
    get_tc_variable_description,
    get_tc_is_input,
)
import pandas as pd
import numpy as np
from tqdm import tqdm

st.title("Full comparisons")


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

VARIABLE_MAP = {
    "agi": "c00100",
}


def get_dataset_aggregate(
    dataset: pd.DataFrame,
    is_taxable_only: bool = False,
    income_range: int = 0,
    variable: str = None,
):
    if is_taxable_only:
        subset = dataset[dataset.iitax - dataset.c07100 > 0]
    else:
        subset = dataset

    if income_range == 1:
        lower_agi = -np.inf
        upper_agi = np.inf
    else:
        if income_range - 1 >= len(INCOME_RANGES):
            return None
        lower_agi = INCOME_RANGES[income_range - 2]
        upper_agi = INCOME_RANGES[income_range - 1]

    subset = subset[(subset.c00100 >= lower_agi) & (subset.c00100 < upper_agi)]

    if "_single" in variable:
        subset = subset[subset.MARS == 1]
        shortened_variable_name = variable.replace("_single", "")
    elif "_mfjss" in variable:
        subset = subset[subset.MARS.isin([2, 5])]
        shortened_variable_name = variable.replace("_mfjss", "")
    elif "_hoh" in variable:
        subset = subset[subset.MARS == 4]
        shortened_variable_name = variable.replace("_hoh", "")
    elif "_mfs" in variable:
        subset = subset[subset.MARS == 3]
        shortened_variable_name = variable.replace("_mfs", "")
    else:
        shortened_variable_name = variable

    if "nret_" in variable:
        shortened_variable_name = variable.replace("nret_", "")

    dataset_variable_name = VARIABLE_MAP.get(shortened_variable_name)

    if dataset_variable_name is None:
        return None

    weight = dataset.s006

    if "nret_" in variable:
        return round(
            ((subset[dataset_variable_name] > 0) * weight).sum() / 1e3, 1
        )
    else:
        return round((subset[dataset_variable_name] * weight).sum() / 1e9, 1)


with st.expander("Compute selected aggregates"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        is_taxable_only = st.checkbox("Taxable returns only", value=False)
    with col2:
        income_range = st.selectbox(
            "Income range",
            agi_targets.incsort.unique(),
            index=0,
        )
    with col3:
        dataset = st.selectbox(
            "Dataset",
            ["PUF-PE 21", "TD 23"],
            index=0,
        )
    with col4:
        variable = st.selectbox(
            "Variable",
            agi_targets.vname.unique(),
            index=0,
        )

    dataset_aggregate = get_dataset_aggregate(
        puf_pe_21 if dataset == "PUF-PE 21" else td_23,
        is_taxable_only,
        income_range,
        variable,
    )

    st.metric(
        "Aggregate",
        get_dataset_aggregate(puf_pe_21, is_taxable_only, 1, variable),
    )


@st.cache_data
def compute_comparison_df():
    comparison_df = pd.DataFrame()

    for taxable in [True, False]:
        for income_range in [1, 2, 4, 5]:
            for variable in agi_targets.vname.sample(10):
                soi_aggregate = agi_targets[
                    (
                        agi_targets.datatype
                        == ("taxable" if taxable else "filers")
                    )
                    & (agi_targets.vname == variable)
                    & (agi_targets.incsort == income_range)
                    & (agi_targets.year == 2021)
                ].ptarget
                if soi_aggregate.empty:
                    continue
                else:
                    soi_aggregate = round(
                        soi_aggregate.values[0]
                        * (1 if "nret_" in variable else 1e3)
                        / (1e3 if "nret_" in variable else 1e9),
                        1,
                    )
                comparison_df = pd.concat(
                    [
                        comparison_df,
                        pd.DataFrame(
                            {
                                "taxable": [taxable],
                                "dataset": ["SOI"],
                                "aggregate": [soi_aggregate],
                                "variable": variable,
                                "income_range": income_range,
                            }
                        ),
                    ]
                )
                for dataset, dataset_name in [
                    (puf_pe_21, "PUF-PE 21"),
                    (td_23, "TD 23"),
                ]:
                    dataset_aggregate = get_dataset_aggregate(
                        dataset, taxable, income_range, variable
                    )
                    comparison_df = pd.concat(
                        [
                            comparison_df,
                            pd.DataFrame(
                                {
                                    "taxable": [taxable],
                                    "dataset": [dataset_name],
                                    "aggregate": [dataset_aggregate],
                                    "variable": variable,
                                    "income_range": income_range,
                                }
                            ),
                        ]
                    )

    comparison_df = comparison_df.drop_duplicates(
        ["taxable", "variable", "dataset"]
    )

    comparison_df = comparison_df.pivot(
        index=["taxable", "variable"], columns="dataset", values="aggregate"
    ).reset_index()

    comparison_df["PUF-PE 21 - SOI"] = (
        comparison_df["PUF-PE 21"] - comparison_df["SOI"]
    )
    comparison_df["TD 23 - SOI"] = (
        comparison_df["TD 23"] - comparison_df["SOI"]
    )
    comparison_df["Closest to SOI"] = np.select(
        [
            comparison_df["PUF-PE 21 - SOI"].abs()
            < comparison_df["TD 23 - SOI"].abs(),
            comparison_df["PUF-PE 21 - SOI"].abs()
            > comparison_df["TD 23 - SOI"].abs(),
            True,
        ],
        [
            "PUF-PE 21",
            "TD 23",
            "Unknown",
        ],
    )
    return comparison_df


comparison_df = compute_comparison_df()
st.dataframe(comparison_df)
