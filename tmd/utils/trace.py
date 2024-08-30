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
        f">{loc} WEIGHT all,puf,cps (#M)= "
        f"{wtot:.3f} {wpuf:.3f} [160.8] {wcps:.3f}"
    )
    print(wght_results)
    print(f">{loc} WEIGHT all_min,all_max= {wght_min:.1f} {wght_max:.1f}")
    # population tabulations
    people = vdf.XTOT
    pop_tot = (wght * people).sum() * 1e-6
    pop_puf = (wght * filer * people).sum() * 1e-6
    pop_cps = (wght * ~filer * people).sum() * 1e-6
    pop_results = (
        f">{loc} POPULATION all,puf,cps (#M)= "
        f"{pop_tot:.3f} [334.181] {pop_puf:.3f} {pop_cps:.3f}"
    )
    print(pop_results)
    # pencon tabulations
    pencon = vdf.pencon_p + vdf.pencon_s
    pencon_amt = (wght * filer * pencon).sum() * 1e-9
    pencon_num = (wght * filer * (pencon > 0)).sum() * 1e-6
    print(f">{loc} weighted puf PENCON ($B)= {pencon_amt:.3f}")
    print(f">{loc} weighted puf PENCON>0 (#M)= {pencon_num:.3f}")
    pencon_amt = (wght * ~filer * pencon).sum() * 1e-9
    pencon_num = (wght * ~filer * (pencon > 0)).sum() * 1e-6
    print(f">{loc} weighted cps PENCON ($B)= {pencon_amt:.3f}")
    print(f">{loc} weighted cps PENCON>0 (#M)= {pencon_num:.3f}")
    # CTC tabulations
    if "ctc_total" in vdf:
        ctc = vdf.ctc_total * filer
        ctc_amt = (wght * ctc).sum() * 1e-9
        ctc_num = (wght * (ctc > 0)).sum() * 1e-6
        print(f">{loc} weighted puf CTC ($B)= {ctc_amt:.3f} [124.6]")
        print(f">{loc} weighted puf CTC>0 (#M)= {ctc_num:.3f} [36.5...47.4]")
    else:
        print(f">{loc} CTC not in DataFrame")
    # SALT tabulations
    salt = (vdf.e18400 + vdf.e18500) * filer
    salt_amt = (wght * salt).sum() * 1e-9
    salt_num = (wght * (salt > 0)).sum() * 1e-6
    print(f">{loc} weighted puf SALT ($B)= {salt_amt:.3f} [?]")
    print(f">{loc} weighted puf SALT>0 (#M)= {salt_num:.3f} [14.3...27.1]")
    # PT_binc_w2_wages tabulations
    w2wages = vdf.PT_binc_w2_wages * filer
    wages_min = w2wages.min()
    wages_max = w2wages.max()
    wages_wtot = (wght * w2wages).sum() * 1e-9
    print(f">{loc} W2_WAGES min,max ($)= {wages_min:.0f} {wages_max:.0f}")
    print(f">{loc} weighted puf W2_WAGES ($B)= {wages_wtot:.3f}")
    # QBID tabulations
    if "qbided" in vdf:
        qbid = vdf.qbided * filer
        qbid_wtot = (wght * qbid).sum() * 1e-9
        print(f">{loc} weighted puf QBID ($B)= {qbid_wtot:.3f} [205.8]")
    else:
        print(f">{loc} QBID not in DataFrame")
    # IITAX tabulations
    if "iitax" in vdf:
        wiitax = wght * vdf.iitax
        itax_tot = (wiitax).sum() * 1e-9
        itax_puf = (wiitax * filer).sum() * 1e-9
        itax_cps = (wiitax * ~filer).sum() * 1e-9
        itax_results = (
            f">{loc} weighted all,puf,cps IITAX ($B)= "
            f"{itax_tot:.3f} {itax_puf:.3f} {itax_cps:.3f}"
        )
        print(itax_results)
        pos_itax_cps = (wght * (vdf.iitax > 0) * ~filer).sum() * 1e-6
        print(f">{loc} weighted cps IITAX>0 (#M)= {pos_itax_cps:.3f}")
    else:
        print(f">{loc} IITAX not in DataFrame")
