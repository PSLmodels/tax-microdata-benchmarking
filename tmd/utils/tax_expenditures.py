"""
This module provides a utility function that calculates
selected 2023 tax expenditue estimates using Tax-Calculator.
"""

import pathlib
import pandas as pd
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR
from tmd.utils.taxcalc_output import add_taxcalc_outputs

TAX_EXPENDITURE_REFORMS = {
    "ctc": {"CTC_c": {"2023": 0}, "ODC_c": {"2023": 0}, "ACTC_c": {"2023": 0}},
    "eitc": {"EITC_c": {"2023": [0, 0, 0, 0]}},
    "social_security_partial_taxability": {"SS_all_in_agi": {"2023": True}},
    "niit": {"NIIT_rt": {"2023": 0}},
    "cgqd_tax_preference": {"CG_nodiff": {"2023": True}},
    "qbid": {"PT_qbid_rt": {"2023": 0}},
    "salt": {"ID_AllTaxes_hc": {"2023": 1}},
}
TAX_EXPENDITURE_PATH = STORAGE_FOLDER / "output" / "tax_expenditures"


def get_tax_expenditure_results(
    flat_file: pd.DataFrame,
    input_data_year: int,
    simulation_year: int,
    weights_file_path: pathlib.Path,
    growfactors_file_path: pathlib.Path,
) -> dict:
    """
    Returns a dictionary containing tax expenditure estimates and
    writes estimates in the TAX_EXPENDITURE_PATH file.
    """
    assert input_data_year == TAXYEAR
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

    taxexp_results = {}
    for reform_name, reform in TAX_EXPENDITURE_REFORMS.items():
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
        taxexp_results[reform_name] = round(-revenue_effect, 1)

    if simulation_year == 2023:
        open_mode = "w"
    else:
        open_mode = "a"
    year = simulation_year
    with open(TAX_EXPENDITURE_PATH, open_mode, encoding="utf-8") as tefile:
        res = f"YR,KIND,EST= {year} paytax {ptax_baseline:.1f}\n"
        tefile.write(res)
        omb_itax_revenue = itax_baseline + itax_baseline_refcredits
        res = f"YR,KIND,EST= {year} iitax {omb_itax_revenue:.1f}\n"
        tefile.write(res)
        for reform, estimate in taxexp_results.items():
            res = f"YR,KIND,EST= {year} {reform} {estimate}\n"
            tefile.write(res)

    return taxexp_results
