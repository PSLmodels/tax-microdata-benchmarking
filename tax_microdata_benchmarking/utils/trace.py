"""
This module provides tracing utilities for working with the repository.
"""

import pandas as pd


def trace1(loc: str, vdf: pd.DataFrame) -> None:
    """
    Write to stdout loc and trace1 tabulation of specified DataFrame.

    Args:
        loc (str): Identifies location of call to trace1.
        vdf (DataFrame):  Contains variable to tabulate.

    Returns:
        None
    """
    tracing = True
    if not tracing:
        return
    # weight tabulations
    wght = vdf.s006
    filer = vdf.data_source == 1
    wtot = wght.sum() * 1e-6
    wpuf = (wght * filer).sum() * 1e-6
    wcps = (wght * ~filer).sum() * 1e-6
    print(f">{loc} weights tot,puf,cps (#M)= {wtot:.3f} {wpuf:.3f} {wcps:.3f}")
    # PT_binc_w2_wages tabulations
    w2wages = vdf.PT_binc_w2_wages
    wages_min = w2wages.min()
    wages_max = w2wages.max()
    wages_wtot = (wght * w2wages).sum() * 1e-9
    print(f">{loc} W2_wages min,max ($)= {wages_min:.0f} {wages_max:.0f}")
    print(f">{loc} total weighted W2_wages ($B)= {wages_wtot:.3f}")
