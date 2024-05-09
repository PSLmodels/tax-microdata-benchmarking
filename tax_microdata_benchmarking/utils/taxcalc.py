import taxcalc as tc
import pandas as pd


def add_taxcalc_outputs(
    flat_file: pd.DataFrame,
    time_period: int,
    reform: dict = None,
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
    input_data = tc.Records(data=flat_file, start_year=time_period)
    policy = tc.Policy()
    if reform is not None:
        policy.implement_reform(reform)
    simulation = tc.Calculator(records=input_data, policy=policy)
    simulation.calc_all()
    output = simulation.dataframe(None, all_vars=True)
    output.s006 = flat_file.s006  # Tax-Calculator seems to change the weights.
    return output


te_reforms = {
    "cg_tax_preference": {"CG_nodiff": {"2015": True}},
    "ctc": {"CTC_c": {"2015": 0}, "ODC_c": {"2015": 0}, "ACTC_c": {"2015": 0}},
    "eitc": {"EITC_c": {"2015": [0, 0, 0, 0]}},
    "niit": {"NIIT_rt": {"2015": 0}},
    "qbid": {"PT_qbid_rt": {"2015": 0}},
    "salt": {"ID_AllTaxes_hc": {"2015": 1}},
    "social_security_partial_taxability": {"SS_all_in_agi": {"2015": True}},
}


def get_tax_expenditure_results(
    flat_file: pd.DataFrame,
    time_period: int,
) -> dict:
    baseline = add_taxcalc_outputs(flat_file, time_period)

    tax_revenue_baseline = (baseline.combined * baseline.s006).sum() / 1e9

    te_results = {}
    for reform_name, reform in te_reforms.items():
        reform_results = add_taxcalc_outputs(flat_file, time_period, reform)
        tax_revenue_reform = (
            reform_results.combined * reform_results.s006
        ).sum() / 1e9
        revenue_effect = tax_revenue_baseline - tax_revenue_reform
        te_results[reform_name] = round(revenue_effect, 1)

    return te_results
