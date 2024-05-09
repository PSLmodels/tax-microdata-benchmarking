import streamlit as st

# Wide layout
# st.set_page_config(layout="wide")

import taxcalc as tc
import pandas as pd
from tqdm import tqdm
from microdf import MicroDataFrame
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import yaml
import plotly.express as px

st.title("Microdata comparison dashboard")

st.markdown(
    f"This app compares multiple microsimulation model input datasets (after running through Tax-Calculator)."
)


# CSV or CSV.GZ files in the storage folder
storage_files = list((STORAGE_FOLDER / "output").glob("*.csv*"))

# Multi-select box for choosing datasets

datasets = st.multiselect(
    "Select datasets to compare",
    [file.name for file in storage_files],
    default=[file.name for file in storage_files[:2]],
)


def run_through_tc(
    flat_file: pd.DataFrame, time_period: int = 2021
) -> pd.DataFrame:
    """
    Run a flat file through Tax-Calculator.

    Args:
        flat_file (pd.DataFrame): The flat file to run through Tax-Calculator.
        time_period (int): The year to run the simulation for.

    Returns:
        pd.DataFrame: The Tax-Calculator output.
    """
    input_data = tc.Records(data=flat_file, start_year=time_period)
    policy = tc.Policy()
    simulation = tc.Calculator(records=input_data, policy=policy)
    simulation.calc_all()
    output = simulation.dataframe(None, all_vars=True)
    output.s006 = flat_file.s006
    return output


# Load the datasets


def load_dataset(
    dataset: str, run_tax_calculator: bool = True
) -> MicroDataFrame:
    """
    Load a dataset from the storage folder, optionally running it through Tax-Calculator.

    Args:
        dataset (str): The dataset to load.
        run_tax_calculator (bool): Whether to run the dataset through Tax-Calculator.

    Returns:
        MicroDataFrame: The dataset.
    """
    df = pd.read_csv(STORAGE_FOLDER / "output" / dataset)

    # Drop non-numeric columns

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df.drop(column, axis=1, inplace=True)

    return MicroDataFrame(df, weights="s006")


with open(STORAGE_FOLDER / "input" / "taxcalc_variable_metadata.yaml") as f:
    taxcalc_variable_metadata = yaml.safe_load(f)


def add_variable_descriptions(
    df: pd.DataFrame, variable_column=None
) -> pd.DataFrame:
    """
    Add variable descriptions to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add descriptions to.
        variable_column (str): The column in the DataFrame that contains variable names. If None, the index is used.

    Returns:
        pd.DataFrame: The DataFrame with descriptions added.
    """
    if variable_column is not None:
        variables = df[variable_column].values
    else:
        variables = df.index.values

    descriptions = []

    for variable in variables:
        if variable in taxcalc_variable_metadata.get("read", {}):
            descriptions.append(
                taxcalc_variable_metadata["read"][variable]["desc"]
            )
        elif variable in taxcalc_variable_metadata.get("calc", {}):
            descriptions.append(
                taxcalc_variable_metadata["calc"][variable]["desc"]
            )
        else:
            descriptions.append("No description found")

    df["Description"] = descriptions

    return df


dfs = []

for dataset in datasets:
    dfs.append(load_dataset(dataset))

st.write("Variable totals ($bn) by dataset")

totals_df = pd.concat(
    [(df.sum() / 1e9).apply(lambda x: round(x * 10) / 10) for df in dfs],
    axis=1,
)
totals_df.columns = datasets

# Add all intersection differences
from itertools import combinations

for dataset_combo, names_combo in zip(
    combinations(dfs, 2), combinations(datasets, 2)
):
    dataset1, dataset2 = dataset_combo
    dataset1_name, dataset2_name = names_combo

    # Absolute difference
    totals_df[f"{dataset1_name} - {dataset2_name}"] = (
        (dataset1.sum() - dataset2.sum()) / 1e9
    ).fillna(0).apply(lambda x: round(x * 10) / 10)
    # Relative difference
    totals_df[f"{dataset1_name} - {dataset2_name} (%)"] = (
        (dataset1.sum() - dataset2.sum()) / dataset1.sum() * 100
    )

totals_df = add_variable_descriptions(totals_df)
st.dataframe(totals_df)

st.write("Variable distributions by dataset")

variable = st.selectbox(
    "Select a variable to compare distributions", totals_df.index
)

dataset = st.selectbox("Select a dataset", datasets)

fig = px.histogram(
    dfs[datasets.index(dataset)], x=variable, nbins=50, title=f"{variable} distribution in {dataset}"
)

st.plotly_chart(fig)