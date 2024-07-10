"""
This module provides utilities for working with Tax-Calculator.
"""

import pathlib
import yaml
import numpy as np
import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import taxcalc as tc


with open(STORAGE_FOLDER / "input" / "taxcalc_variable_metadata.yaml") as f:
    taxcalc_variable_metadata = yaml.safe_load(f)


def get_tc_variable_description(variable: str) -> str:
    """
    Get the description of a Tax-Calculator variable.

    Args:
        variable (str): The name of the variable.

    Returns:
        str: The description of the variable.
    """
    if variable in taxcalc_variable_metadata.get("read", {}):
        return taxcalc_variable_metadata["read"][variable]["desc"]
    elif variable in taxcalc_variable_metadata.get("calc", {}):
        return taxcalc_variable_metadata["calc"][variable]["desc"]


def get_tc_is_input(variable: str) -> bool:
    """
    Get the type (whether input or not) of a Tax-Calculator variable.

    Args:
        variable (str): The name of the variable.

    Returns:
        bool: Whether the variable is an input.
    """
    if variable in taxcalc_variable_metadata.get("read", {}):
        return True
    elif variable in taxcalc_variable_metadata.get("calc", {}):
        return False


def add_taxcalc_outputs(
    flat_file: pd.DataFrame,
    input_data_year: int,
    simulation_year: int,
    reform: dict = None,
    weights=None,
    growfactors=None,
) -> pd.DataFrame:
    """
    Run a flat file through Tax-Calculator.

    Args:
        flat_file (pd.DataFrame): The flat file to run through Tax-Calculator.
        time_period (int): The year to run the simulation for.
        reform (dict, optional): The reform to apply. Defaults to None.

    Returns:
        pd.DataFrame: The Tax-Calculator output.
    """
    if isinstance(weights, pathlib.PosixPath):
        wghts = str(weights)
    else:
        wghts = weights
    if isinstance(growfactors, pathlib.PosixPath):
        growf = tc.GrowFactors(growfactors_filename=str(growfactors))
    else:
        growf = growfactors
    input_data = tc.Records(
        data=flat_file,
        start_year=input_data_year,
        gfactors=growf,
        weights=wghts,
        adjust_ratios=None,
        exact_calculations=True,
    )
    policy = tc.Policy()
    if reform:
        policy.implement_reform(reform)
    simulation = tc.Calculator(records=input_data, policy=policy)
    simulation.advance_to_year(simulation_year)
    simulation.calc_all()
    output = simulation.dataframe(None, all_vars=True)
    if weights is None and growfactors is None:
        assert np.allclose(output.s006, flat_file.s006)
    return output


te_reforms = {
    "cg_tax_preference": {"CG_nodiff": {"2023": True}},
    "ctc": {"CTC_c": {"2023": 0}, "ODC_c": {"2023": 0}, "ACTC_c": {"2023": 0}},
    "eitc": {"EITC_c": {"2023": [0, 0, 0, 0]}},
    "niit": {"NIIT_rt": {"2023": 0}},
    "qbid": {"PT_qbid_rt": {"2023": 0}},
    "salt": {"ID_AllTaxes_hc": {"2023": 1}},
    "social_security_partial_taxability": {"SS_all_in_agi": {"2023": True}},
}


def get_tax_expenditure_results(
    flat_file: pd.DataFrame,
    input_data_year: int,
    simulation_year: int,
    weights_file_name: str,
    growfactors_file_name: str,
) -> dict:
    baseline = add_taxcalc_outputs(
        flat_file,
        input_data_year,
        simulation_year,
        reform=None,
        weights=weights_file_name,
        growfactors=growfactors_file_name,
    )
    tax_revenue_baseline = (baseline.combined * baseline.s006).sum() / 1e9

    te_results = {}
    for reform_name, reform in te_reforms.items():
        reform_results = add_taxcalc_outputs(
            flat_file,
            input_data_year,
            simulation_year,
            reform,
            weights=weights_file_name,
            growfactors=growfactors_file_name,
        )
        tax_revenue_reform = (
            reform_results.combined * reform_results.s006
        ).sum() / 1e9
        revenue_effect = tax_revenue_baseline - tax_revenue_reform
        te_results[reform_name] = round(-revenue_effect, 1)

    taxexp_path = STORAGE_FOLDER / "output" / "tax_expenditures"
    year = simulation_year
    with open(taxexp_path, "w") as tefile:
        for reform, estimate in te_results.items():
            res = f"YEAR,KIND,ESTIMATE= {year} {reform} {estimate}\n"
            tefile.write(res)

    return te_results
