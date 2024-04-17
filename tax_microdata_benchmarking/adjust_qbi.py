import numpy as np
import pandas as pd
from scipy.optimize import minimize

def add_pt_w2_wages(df, time_period: int, verbose: bool = True):
    """
    Add pass-through W2 wages to the flat file.

    Args:
        df (pd.DataFrame): The DataFrame to add W2 wages to.

    Returns:
        pd.DataFrame: The DataFrame with W2 wages added.
    """
    qbid_tax_expenditures = { # From JCT TE reports 2018- and 2023-
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
    }

    QBID_TOTAL_21 = 205.8

    target = QBID_TOTAL_21 * qbid_tax_expenditures[time_period] / qbid_tax_expenditures[2021]

    qbi = np.maximum(0, df.e00900 + df.e26270 + df.e02100 + df.e27200)

    # Solve for scale to match the tax expenditure

    def expenditure_loss(scale):
        res = (qbi * df.s006).sum()/1e9
        deviation = (res - target)
        print(f"Scale: {scale}, expenditure: {res}, deviation: {deviation}")
        return deviation ** 2
    
    
    scale = minimize(
        expenditure_loss,
        1,
        tol=1,
    ).x[0]

    df["PT_binc_w2_wages"] = qbi * scale
    
    return df