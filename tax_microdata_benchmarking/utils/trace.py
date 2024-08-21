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
    filer = vdf.data_source == 1
    # unweighted tabulations
    utot = len(filer)
    upuf = filer.sum()
    ucps = (~filer).sum()
    print(f">{loc} count all,puf,cps (#)= " f"{utot} {upuf} {ucps}")
    # weight tabulations
    wght = vdf.s006
    wtot = wght.sum() * 1e-6
    wpuf = (wght * filer).sum() * 1e-6
    wcps = (wght * ~filer).sum() * 1e-6
    wght_min = wght.min()
    wght_max = wght.max()
    wght_results = (
        f">{loc} weights all,puf,cps (#M)= "
        f"{wtot:.3f} {wpuf:.3f} [160.8] {wcps:.3f}"
    )
    print(wght_results)
    print(f">{loc} weights all_min,all_max (#)= {wght_min:.1f} {wght_max:.1f}")
    # CTC tabulations
    if "ctc_total" in vdf:
        ctc = vdf.ctc_total * filer
        ctc_amt = (wght * ctc).sum() * 1e-9
        ctc_num = (wght * (ctc > 0)).sum() * 1e-6
        print(f">{loc} weighted puf CTC ($B)= {ctc_amt:.3f} [124.6]")
        print(f">{loc} weighted puf CTC (#M)= {ctc_num:.3f} [36.5...47.4]")
    else:
        print(f">{loc} CTC not in DataFrame")
    # SALT tabulations
    salt = (vdf.e18400 + vdf.e18500) * filer
    salt_amt = (wght * salt).sum() * 1e-9
    salt_num = (wght * (salt > 0)).sum() * 1e-6
    print(f">{loc} weighted puf SALT ($B)= {salt_amt:.3f} [?]")
    print(f">{loc} weighted puf SALT (#M)= {salt_num:.3f} [14.3...27.1]")
    # PT_binc_w2_wages tabulations
    w2wages = vdf.PT_binc_w2_wages * filer
    wages_min = w2wages.min()
    wages_max = w2wages.max()
    wages_wtot = (wght * w2wages).sum() * 1e-9
    print(f">{loc} W2_wages min,max ($)= {wages_min:.0f} {wages_max:.0f}")
    print(f">{loc} weighted puf W2_wages ($B)= {wages_wtot:.3f}")
    # QBID tabulations
    if "qbided" in vdf:
        qbid = vdf.qbided * filer
        qbid_wtot = (wght * qbid).sum() * 1e-9
        print(f">{loc} weighted puf QBID ($B)= {qbid_wtot:.3f} [205.8]")
    else:
        print(f">{loc} QBID not in DataFrame")
