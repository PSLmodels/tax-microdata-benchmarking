"""
Generate sample precision estimates using bootstrap sampling methods.
"""

import sys
import numpy as np
import pandas as pd

USAGE = "USAGE: python bootstrap_sampling.py tc_dump_output_csv_file_name\n"

SS_FRAC = 1.00  # 0.1
SS_RNSEED = 902345678

BS_SAMPLES = 1000
BS_RNSEED = 192837465
BS_DUMP = False
BS_CI = True


def bootstrap_sampling(outfile):
    """
    High-level logic of the script.
    """
    # read odf from outfile and possibly sample it
    odf = pd.read_csv(outfile)
    gdf = odf[(odf["data_source"] == 1) & (odf["iitax"] > 0)]
    print("len(odf),len(gdf) =", len(odf), len(gdf))
    if SS_FRAC < 1.0:
        ss_rng = np.random.default_rng(seed=SS_RNSEED)
        fdf = gdf.sample(frac=SS_FRAC, replace=True, random_state=ss_rng)
    else:
        fdf = gdf
    print(f"SS_FRAC = {SS_FRAC:.2f}")
    print(f"SS:wght(#M) = {fdf['s006'].sum() * 1e-6:.3f}")
    print(f"SS:itax($B) = {(fdf['s006'] * fdf['iitax']).sum() * 1e-9:.3f}")

    # compute sum of wght and wght*itax for each bootstrap sample
    xdf = pd.DataFrame({"wght": fdf["s006"], "itax": fdf["iitax"]})
    bs_rng = np.random.default_rng(seed=BS_RNSEED)
    wght_list = []
    itax_list = []
    for bss in range(1, BS_SAMPLES + 1):
        bsdf = xdf.sample(n=len(xdf), replace=True, random_state=bs_rng)
        wght_value = bsdf["wght"].sum() * 1e-6
        itax_value = (bsdf["wght"] * bsdf["itax"]).sum() * 1e-9
        if BS_DUMP:
            print(f"{itax_value:9.3f} {wght_value:7.3f} {bss:4d}")
        wght_list.append(wght_value)
        itax_list.append(itax_value)
    wght = np.sort(np.array(wght_list))
    itax = np.sort(np.array(itax_list))
    wght_mean = wght.mean()
    wght_stdv = wght.std()
    wght_cv = wght_stdv / wght_mean
    print(
        f"BS:wght num,mean,stdev,cv(%) = {BS_SAMPLES:4d}  "
        f"{wght_mean:9.3f}  {wght_stdv:7.3f}  {100 * wght_cv:8.4f}"
    )
    if BS_CI and BS_SAMPLES == 1000:
        print(f"BS:wght median = {wght[499]:9.3f}")
        print(f"BS:wght 95%_conf_intv = {wght[24]:9.3f}  , {wght[974]:9.3f}")
    itax_mean = itax.mean()
    itax_stdv = itax.std()
    itax_cv = itax_stdv / itax_mean
    print(
        f"BS:itax num,mean,stdev,cv(%) = {BS_SAMPLES:4d}  "
        f"{itax_mean:9.3f}  {itax_stdv:7.3f}  {100 * itax_cv:8.4f}"
    )
    if BS_CI and BS_SAMPLES == 1000:
        print(f"BS:itax median = {itax[499]:9.3f}")
        print(f"BS:itax 95%_conf_intv = {itax[24]:9.3f}  , {itax[974]:9.3f}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) - 1 != 1:
        sys.stderr.write("ERROR: one command-line argument not specified\n")
        sys.stderr.write(USAGE)
        sys.exit(1)
    sys.exit(bootstrap_sampling(sys.argv[1]))
