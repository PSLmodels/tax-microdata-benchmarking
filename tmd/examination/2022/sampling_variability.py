"""
Calculate hypothetical TMD-vs-SOI itax percentage differences
using resampling methods.
"""

import sys
import yaml
import numpy as np

BS_SAMPLES = 1000
BS_RNSEED = 192837465

ITX_MEAN = 2300
TMD_CV = 0.0034
SOI_CV = 0.0102

USAGE = "USAGE: python sampling_variability.py ASSUMPTIONS_YAML_FILE_NAME\n"


def sampling_variability(yamlfilename):
    """
    High-level logic of the script.
    """
    #
    with open(yamlfilename, "r", encoding="utf-8") as yamlfile:
        assumptions = yaml.safe_load(yamlfile)
    assert isinstance(assumptions, dict)

    # specify rng and draw samples
    for area, asmp in assumptions.items():
        print(f"Generating results for {area} ...")
        rng = np.random.default_rng(seed=BS_RNSEED)
        mean = asmp["mean"]
        cv_tmd = asmp["cv_tmd"]
        cv_soi = asmp["cv_soi"]
        tmd = rng.normal(mean, cv_tmd * mean, BS_SAMPLES)
        soi = rng.normal(mean, cv_soi * mean, BS_SAMPLES)
        del rng
        pctdiff = 100 * (tmd / soi - 1)
        print(f"mean,cv_tmd,cv_soi = {mean:.3f} {cv_tmd:.6f} {cv_soi:.6f}")
        pd_mean = pctdiff.mean()
        pd_stdv = pctdiff.std()
        print(
            f"BS:pctdiff num,mean,stdev = {BS_SAMPLES:4d}  "
            f"{pd_mean:9.3f}  {pd_stdv:7.3f}"
        )
        if BS_SAMPLES == 1000:
            pdiff = np.sort(pctdiff)
            print(f"BS:pctdiff median = {pdiff[499]:9.3f}")
            print(f"BS:pctdiff 95%_ci = {pdiff[24]:9.3f} , {pdiff[974]:9.3f}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) - 1 != 1:
        sys.stderr.write("ERROR: one command-line argument not specified\n")
        sys.stderr.write(USAGE)
        sys.exit(1)
    sys.exit(sampling_variability(sys.argv[1]))
