"""
This module provides a function that adds Tax-Calculator output variables to
a Tax-Calculator input DataFrame.
"""

import pathlib
import numpy as np
import pandas as pd
import taxcalc
from tmd.imputation_assumptions import CREDIT_CLAIMING


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
        growf = taxcalc.GrowFactors(growfactors_filename=str(growfactors))
    else:
        growf = growfactors
    rec = taxcalc.Records(
        data=flat_file,
        start_year=input_data_year,
        gfactors=growf,
        weights=wghts,
        adjust_ratios=None,
        exact_calculations=True,
        weights_scale=1.0,
    )
    pol = taxcalc.Policy()
    pol.implement_reform(CREDIT_CLAIMING)
    if reform:
        pol.implement_reform(reform)
    simulation = taxcalc.Calculator(records=rec, policy=pol)
    simulation.advance_to_year(simulation_year)
    simulation.calc_all()
    output = simulation.dataframe(None, all_vars=True)
    if weights is None and growfactors is None:
        assert np.allclose(output.s006, flat_file.s006)
    return output
