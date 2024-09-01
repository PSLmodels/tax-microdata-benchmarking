"""
This module provides utilities for working with Tax-Calculator.
"""

import pathlib
import yaml
import numpy as np
import pandas as pd
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER


with open(STORAGE_FOLDER / "input" / "tc_variable_metadata.yaml") as f:
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
    "ctc": {"CTC_c": {"2023": 0}, "ODC_c": {"2023": 0}, "ACTC_c": {"2023": 0}},
    "eitc": {"EITC_c": {"2023": [0, 0, 0, 0]}},
    "social_security_partial_taxability": {"SS_all_in_agi": {"2023": True}},
    "niit": {"NIIT_rt": {"2023": 0}},
    "cgqd_tax_preference": {"CG_nodiff": {"2023": True}},
    "qbid": {"PT_qbid_rt": {"2023": 0}},
    "salt": {"ID_AllTaxes_hc": {"2023": 1}},
}


def get_tax_expenditure_results(
    flat_file: pd.DataFrame,
    input_data_year: int,
    simulation_year: int,
    weights_file_path: pathlib.Path,
    growfactors_file_path: pathlib.Path,
) -> dict:
    assert input_data_year == 2021
    assert simulation_year in [2023, 2026]
    baseline = add_taxcalc_outputs(
        flat_file,
        input_data_year,
        simulation_year,
        reform=None,
        weights=weights_file_path,
        growfactors=growfactors_file_path,
    )
    ptax_baseline = (baseline.payrolltax * baseline.s006).sum() / 1e9
    itax_baseline = (baseline.iitax * baseline.s006).sum() / 1e9
    itax_baseline_refcredits = (baseline.refund * baseline.s006).sum() / 1e9

    te_results = {}
    for reform_name, reform in te_reforms.items():
        reform_results = add_taxcalc_outputs(
            flat_file,
            input_data_year,
            simulation_year,
            reform,
            weights=weights_file_path,
            growfactors=growfactors_file_path,
        )
        tax_revenue_reform = (
            reform_results.iitax * reform_results.s006
        ).sum() / 1e9
        revenue_effect = itax_baseline - tax_revenue_reform
        te_results[reform_name] = round(-revenue_effect, 1)

    taxexp_path = STORAGE_FOLDER / "output" / "tax_expenditures"
    if simulation_year == 2023:
        open_mode = "w"
    else:
        open_mode = "a"
    year = simulation_year
    with open(taxexp_path, open_mode) as tefile:
        res = f"YR,KIND,EST= {year} paytax {ptax_baseline:.1f}\n"
        tefile.write(res)
        omb_itax_revenue = itax_baseline + itax_baseline_refcredits
        res = f"YR,KIND,EST= {year} iitax {omb_itax_revenue:.1f}\n"
        tefile.write(res)
        for reform, estimate in te_results.items():
            res = f"YR,KIND,EST= {year} {reform} {estimate}\n"
            tefile.write(res)

    return te_results
