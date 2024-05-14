import streamlit as st
st.set_page_config(layout="wide")

from utils import puf_pe_21, pe_21, td_23
from tax_microdata_benchmarking.utils.taxcalc import get_tc_variable_description, get_tc_is_input
import pandas as pd
import numpy as np

st.title("Aggregates")

# Calculate weighted aggregates

@st.cache_data
def get_aggregate_df(
    nonzero: bool = False,
    taxpayers_only: bool = False,
) -> pd.DataFrame:
    """
    Get aggregate values for each variable in the dataset.

    Args:
        nonzero: If true, returns the number of nonzero filers as opposed to weighted sums.
    
    Returns:
        DataFrame with the aggregate values.
    """

    aggregates_df = pd.DataFrame()

    def get_value(df, column):
        if taxpayers_only:
            df_ = df[df.iitax - df.c07100 > 0]
        else:
            df_ = df

        if nonzero:
            return ((df_[column] > 0) * df_.s006).sum()
        else:
            return (df_[column] * df_.s006).sum()
        
    divisor = 1e9 if not nonzero else 1e6
    
    for variable in td_23.columns:
        if variable in ["RECID", "s006"]:
            continue
        aggregates_df = pd.concat([
            aggregates_df,
            pd.DataFrame({
                "variable": [variable],
                "input": [get_tc_is_input(variable)],
                "description": [get_tc_variable_description(variable)],
                "puf_pe_21": [get_value(puf_pe_21, variable) / divisor],
                "pe_21": [get_value(pe_21, variable) / divisor],
                "td_23": [get_value(td_23, variable) / divisor]
            })
        ])

    aggregates_df = aggregates_df.set_index("variable")

    COLUMNS_TO_ROUND = ["puf_pe_21", "pe_21", "td_23"]

    for column in COLUMNS_TO_ROUND:
        aggregates_df[column] = aggregates_df[column].round(1)

    aggregates_df["pufpe / td23 (%)"] = (aggregates_df["puf_pe_21"] / aggregates_df["td_23"] - 1).fillna(0).replace(np.inf, 0) * 100

    aggregates_df["pufpe / td23 (%)"] = aggregates_df["pufpe / td23 (%)"].round(1)

    return aggregates_df

st.subheader("Total weighted sums")

st.write("This table shows the total weighted sum of each variable in the dataset. Amounts are in billions of dollars.")

aggregates_df = get_aggregate_df(taxpayers_only=True)

only_inputs = st.checkbox("Only show input variables", value=False, key="only_inputs_agg")

if only_inputs:
    aggregates_df = aggregates_df[aggregates_df.input]

st.dataframe(aggregates_df)

st.subheader("Nonzero filers")

st.write("This table shows the number of filers with nonzero values for each variable in the dataset. Numbers are in millions of filers.")

aggregates_df_nonzero = get_aggregate_df(nonzero=True, taxpayers_only=True)

only_inputs = st.checkbox("Only show input variables", value=False, key="only_inputs_agg_nonzero")

if only_inputs:
    aggregates_df_nonzero = aggregates_df_nonzero[aggregates_df_nonzero.input]

st.dataframe(aggregates_df_nonzero)