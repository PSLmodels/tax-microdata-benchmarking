"""
This module provides tracing utilities for working with this repository.
"""

import pandas as pd


def trace1(loc: str, vdf: pd.DataFrame) -> None:
    """
    Write to stdout loc and trace1 tabulation of specified DataFrame.

    Args:
        loc (str): Identifies location of call to trace1.
        vdf (DataFrame):  Contains variables to tabulate.

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
    wght_min = wght.min()
    wght_max = wght.max()
    print(f">{loc} weights all,puf,cps (#M)= {wtot:.3f} {wpuf:.3f} {wcps:.3f}")
    print(f">{loc} weights all_min,all_max (#)= {wght_min:.1f} {wght_max:.1f}")
    # PT_binc_w2_wages tabulations
    w2wages = vdf.PT_binc_w2_wages
    wages_min = w2wages.min()
    wages_max = w2wages.max()
    wages_wtot = (wght * w2wages).sum() * 1e-9
    print(f">{loc} W2_wages min,max ($)= {wages_min:.0f} {wages_max:.0f}")
    print(f">{loc} total weighted W2_wages ($B)= {wages_wtot:.3f}")
    # QBID tabulations
    if "qbided" in vdf:
        qbid = vdf.qbided
        qbid_wtot = (wght * qbid).sum() * 1e-9
        print(f">{loc} total weighted QBID ($B)= {qbid_wtot:.3f}")
    else:
        print(f">{loc} QBID not in DataFrame")
