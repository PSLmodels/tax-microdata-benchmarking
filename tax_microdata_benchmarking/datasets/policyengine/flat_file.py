"""
This module enables transformations of PolicyEngine datasets (in hierarchichal .h5 format) into flat files (in .csv format) for use in Tax-Calculator.
"""

import warnings

warnings.filterwarnings("ignore")
import taxcalc as tc
from policyengine_us import Microsimulation
from policyengine_us.model_api import *
from policyengine_us.system import system
import numpy as np
import pandas as pd
from policyengine_core.periods import instant
from scipy.optimize import minimize
from tax_microdata_benchmarking.utils.qbi import add_pt_w2_wages
from microdf import MicroDataFrame
import numpy as np
from .data_reform import taxcalc_extension, UPRATING_VARIABLES
from tax_microdata_benchmarking.storage import STORAGE_FOLDER


def create_flat_file(
    source_dataset: str = "enhanced_cps_2022",
    target_year: int = 2024,
) -> pd.DataFrame:
    sim = Microsimulation(reform=taxcalc_extension, dataset=source_dataset)
    original_year = sim.dataset.time_period

    for variable in UPRATING_VARIABLES:
        original_value = sim.calculate(variable, original_year)
        uprating_factor = get_variable_uprating(
            variable,
            source_time_period=original_year,
            target_time_period=target_year,
        )
        try:
            sim.set_input(
                variable, original_year, original_value * uprating_factor
            )
        except Exception as e:
            print(f"Error uprating {variable}: {e}")

    df = pd.DataFrame()

    for variable in sim.tax_benefit_system.variables:
        if variable.startswith("tc_"):
            df[variable[3:]] = sim.calculate(
                variable, original_year
            ).values.astype(np.float64)

        if variable == "is_tax_filer":
            df[variable] = sim.calculate(
                variable, original_year
            ).values.astype(np.float64)

    # Extra quality-control checks to do with different data types, nothing major
    FILER_SUM_COLUMNS = [
        "e00200",
        "e00900",
        "e02100",
    ]
    for column in FILER_SUM_COLUMNS:
        df[column] = df[column + "p"] + df[column + "s"]

    df.e01700 = np.minimum(df.e01700, df.e01500)
    df.e00650 = np.minimum(df.e00650, df.e00600)

    df.RECID = df.RECID.astype(int)
    df.MARS = df.MARS.fillna(1).astype(int)
    df.FLPDYR = target_year

    # Drop duplicate columns

    df = df.loc[:, ~df.columns.duplicated()]

    assert_no_duplicate_columns(df)

    return df


def get_variable_uprating(
    variable: str, source_time_period: str, target_time_period: str
) -> str:
    """
    Get the uprating factor for a given variable between two time periods.

    Args:
        variable (str): The variable to uprate.
        source_time_period (str): The source time period.
        target_time_period (str): The target time period.

    Returns:
        str: The uprating factor.
    """

    population = system.parameters.calibration.gov.census.populations.total

    calibration = system.parameters.calibration
    if variable in calibration.gov.irs.soi.children:
        parameter = calibration.gov.irs.soi.children[variable]
    else:
        parameter = calibration.gov.cbo.income_by_source.adjusted_gross_income
    source_value = parameter(source_time_period)
    target_value = parameter(target_time_period)

    population_change = population(target_time_period) / population(
        source_time_period
    )

    uprating_factor = target_value / source_value
    return uprating_factor / population_change


def assert_no_duplicate_columns(df):
    """
    Assert that there are no duplicate columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check for duplicates.
    """
    assert len(df.columns) == len(set(df.columns))


population = system.parameters.calibration.gov.census.populations.total


def get_population_growth(target_year: int, source_year: int):
    return population(f"{target_year}-01-01") / population(
        f"{source_year}-01-01"
    )
