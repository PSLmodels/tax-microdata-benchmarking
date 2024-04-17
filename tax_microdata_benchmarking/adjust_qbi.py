import numpy as np
import pandas as pd
from scipy.optimize import minimize, bisect
import taxcalc as tc


def add_pt_w2_wages(df, time_period: int, verbose: bool = True):
    """
    Add pass-through W2 wages to the flat file.

    Args:
        df (pd.DataFrame): The DataFrame to add W2 wages to.

    Returns:
        pd.DataFrame: The DataFrame with W2 wages added.
    """
    qbid_tax_expenditures = {  # From JCT TE reports 2018- and 2023-
        2015: 0,
        2016: 0,
        2017: 0,
        2018: 33.2,
        2019: 48.6,
        2020: 56.3,
        2021: 59.0,
        2022: 61.9,
        2023: 55.7,
        2024: 57.6,
        2025: 60.9,
        2026: 24.9,
        2027: 0,
    }

    QBID_TOTAL_21 = 205.8

    target = (
        QBID_TOTAL_21
        * qbid_tax_expenditures[
            time_period + 1
        ]  # JCT figures are one year behind TC (check!)
        / qbid_tax_expenditures[2021]
    )

    qbi = np.maximum(0, df.e00900 + df.e26270 + df.e02100 + df.e27200)

    if target == 0:
        df["PT_binc_w2_wages"] = qbi * 0

        return df

    # Solve for scale to match the tax expenditure

    def expenditure_loss(scale):
        input_data = df.copy()
        input_data["PT_binc_w2_wages"] = qbi * scale
        input_data = tc.Records(data=input_data, start_year=time_period)
        policy = tc.Policy()
        simulation = tc.Calculator(records=input_data, policy=policy)
        simulation.calc_all()
        taxcalc_qbided_sum = (
            simulation.dataframe(["qbided"]).qbided * df.s006
        ).sum() / 1e9
        deviation = taxcalc_qbided_sum - target
        if verbose:
            print(
                f"scale: {scale}, deviation: {deviation}, total: {taxcalc_qbided_sum}"
            )
        return deviation

    scale = bisect(expenditure_loss, 0, 2, rtol=0.01)

    df["PT_binc_w2_wages"] = qbi * scale

    return df
