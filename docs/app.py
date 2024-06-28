import streamlit as st

from tax_microdata_benchmarking.utils.soi_replication import *
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.datasets import *
import pandas as pd

INPUTS = STORAGE_FOLDER / "input"
OUTPUTS = STORAGE_FOLDER / "output"


@st.cache_resource
def generate_comparsions(use_original_weights: bool = False):
    tmd_2021 = pd.read_csv(OUTPUTS / "tmd_2021.csv")
    if use_original_weights:
        tmd_2021.s006 = tmd_2021.s006_original
    soi_from_tmd_2021 = compare_soi_replication_to_soi(
        tc_to_soi(tmd_2021, 2021), 2021
    )
    return soi_from_tmd_2021


def soi_statistic_passes_quality_test(df):
    # Relative error lower than this => OK
    RELATIVE_ERROR_THRESHOLD = 0.05

    # Absolute error lower than this for filer counts => OK
    COUNT_ABSOLUTE_ERROR_THRESHOLD = 1e6

    # Absolute error lower than this for aggregates => OK
    AGGREGATE_ABSOLUTE_ERROR_THRESHOLD = 1e9

    relative_error_ok = (
        df["Absolute relative error"] < RELATIVE_ERROR_THRESHOLD
    )
    absolute_error_threshold = np.where(
        df.Count,
        COUNT_ABSOLUTE_ERROR_THRESHOLD,
        AGGREGATE_ABSOLUTE_ERROR_THRESHOLD,
    )
    absolute_error_ok = df["Absolute error"] < absolute_error_threshold

    return relative_error_ok | absolute_error_ok


# 2021 datasets

comparisons = generate_comparsions()
comparisons_original_weights = generate_comparsions(use_original_weights=True)
comparisons["Original weight value"] = comparisons_original_weights["Value"]
comparisons["Original weight error"] = comparisons_original_weights["Error"]
comparisons["Improved under reweighting"] = (
    comparisons["Absolute error"] < comparisons["Original weight error"].abs()
)
filing_status_all = comparisons["Filing status"] == "All"
filers_only = comparisons["Taxable only"] == False
non_full_range = (comparisons["AGI lower bound"] != -np.inf) | (
    comparisons["AGI upper bound"] != np.inf
)
right_variable = comparisons.Variable.isin(
    ["adjusted_gross_income", "count", "qualified_business_income_deduction"]
)
comparisons["Targeted"] = (
    filing_status_all & filers_only & non_full_range & right_variable
)

st.title("SOI replication results")

st.write(
    """
    This page shows the results of replicating the SOI dataset from the TMD-2021 output data file. It is sorted by absolute error."""
)

st.dataframe(comparisons.sort_values("Absolute error", ascending=False))

import plotly.express as px

histogram = px.histogram(
    comparisons,
    x="Absolute error",
    marginal="rug",
    title="Histogram of absolute relative errors",
)

st.plotly_chart(histogram)

st.subheader("Targets included in reweighting")
