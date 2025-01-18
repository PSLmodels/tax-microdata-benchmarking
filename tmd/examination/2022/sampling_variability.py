"""
Calculate hypothetical TMD-vs-SOI itax percentage differences
using bootstrap sampling methods.
"""

import sys
import numpy as np
import pandas as pd

BS_SAMPLES = 1000
BS_RNSEED = 192837465

ITX_MEAN = 2300
TMD_CV = 0.0034
SOI_CV = 0.0102


def sampling_variability():
    """
    High-level logic of the script.
    """
    # specify rng and draw samples
    rng = np.random.default_rng(seed=BS_RNSEED)
    tmd = rng.normal(ITX_MEAN, TMD_CV * ITX_MEAN, BS_SAMPLES)
    soi = rng.normal(ITX_MEAN, SOI_CV * ITX_MEAN, BS_SAMPLES)
    pctdiff = 100 * (tmd / soi - 1)

    # show results
    print(f"ITX_MEAN,TMD_CV,SOI_CV = {ITX_MEAN:.1f} {TMD_CV:.4f} {SOI_CV:.4f}")
    pd_mean = pctdiff.mean()
    pd_stdv = pctdiff.std()
    pd_cv = pd_stdv / pd_mean
    print(
        f"BS:pctdiff num,mean,stdev,cv(%) = {BS_SAMPLES:4d}  "
        f"{pd_mean:9.3f}  {pd_stdv:7.3f}  {100 * pd_cv:6.2f}"
    )
    if BS_SAMPLES == 1000:
        pdiff = np.sort(pctdiff)
        print(f"BS:pctdiff median = {pdiff[499]:9.3f}")
        print(f"BS:pctdiff 95%_ci = {pdiff[24]:9.3f} , {pdiff[974]:9.3f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(sampling_variability())
