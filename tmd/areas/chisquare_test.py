"""
Conduct chi-square test of the similarity of two distributions of weights by
income tax liability categories for the same area that are generated using
different targets.  The two area weights files are assumed to be located in
the tmd/areas/weights folder.
All test output is written to stdout; error messages are written to stderr.

For background information on the chi-square test that compares two
categorical (that is, binned) data sets, see Section 14.3: Are Two
Distributions Different? in Press, et al., Numerical Recipies in C:
The Art of Scientific Computing, Second Edition (Cambridge University
Press, 1992).  This script uses the chi2_contingency function in the
Python scipy package, which is documented at the following URL:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

Critical chi-square statistic values (say, at p=0.05) for large
degrees-of-freedom values (which equals numbins-1 in this test)
are available from the following online calculator:
https://www.danielsoper.com/statcalc/calculator.aspx?id=12
For example: Xsq(p=0.05,dof= 100)=  124.34
             Xsq(p=0.05,dof= 200)=  233.99
             Xsq(p=0.05,dof=1000)= 1074.68
             Xsq(p=0.05,dof=2000)= 2105.15

USAGE: python chisquare_test.py WGHT1 WGHT2 [numbins] [dump]

EXAMPLE using default numbins (equal to 200) and with no details dump:
areas% python chisquare_test.py pa08 pa08A
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from tmd.areas import AREAS_FOLDER
from tmd.storage import STORAGE_FOLDER

USAGE = "USAGE: python chisquare_test.py WGHT1 WGHT2 [numbins] [dump]"
CACHED_ITAX_PATH = STORAGE_FOLDER / "output" / "cached_iitax.npy"
TAX_YEAR = 2021
MINIMUM_NUM_BINS = 50
DEFAULT_NUM_BINS = 200
ITXBINS_DEFINED_USING_AREA_WEIGHTS = False  # default is using national weights


def check_arguments():
    """
    Check validity of arguments and return the following tuple:
    (wname1, wpath1, wname2, wpath2, numbins, dump)
    containing name of and path to the WGHT1 and WGHT2 weights files,
    plus requested number of income tax categories (or bins) and
    whether or not detailed dump information is included in output.
    """
    numargs = len(sys.argv) - 1
    if not 2 <= numargs <= 4:
        sys.stderr.write(
            f"ERROR: number of arguments not in [2,4] range\n{USAGE}\n"
        )
        sys.exit(1)
    all_ok = True
    wname1 = sys.argv[1]
    wpath1 = AREAS_FOLDER / "weights" / f"{wname1}_tmd_weights.csv.gz"
    if not wpath1.exists():
        sys.stderr.write(f"ERROR: WGHT1 {str(wpath1)} file does not exist\n")
        all_ok = False
    wname2 = sys.argv[2]
    wpath2 = AREAS_FOLDER / "weights" / f"{wname2}_tmd_weights.csv.gz"
    if not wpath2.exists():
        sys.stderr.write(f"ERROR: WGHT2 {str(wpath2)} file does not exist\n")
        all_ok = False
    if not CACHED_ITAX_PATH.exists():
        sys.stderr.write(
            f"ERROR: {str(CACHED_ITAX_PATH)} file does not exist\n"
        )
        all_ok = False
    numbins = DEFAULT_NUM_BINS
    if numargs >= 3:
        numbins = int(sys.argv[3])
        if numbins < MINIMUM_NUM_BINS:
            sys.stderr.write(
                f"ERROR: numbins must be no less than {MINIMUM_NUM_BINS}\n"
            )
            all_ok = False
    dump = False
    if numargs >= 4:
        if sys.argv[4] == "dump":
            dump = True
        else:
            sys.stderr.write("ERROR: optional fourth argument must be dump\n")
            all_ok = False
    if not all_ok:
        sys.stderr.write(f"{USAGE}\n")
        sys.exit(1)
    return (wname1, wpath1, wname2, wpath2, numbins, dump)


def weights_array(wghts_path: Path):
    """
    Return array containing TAX_YEAR weights in area weights file with wpath.
    """
    wdf = pd.read_csv(wghts_path)
    warray = wdf[f"WT{TAX_YEAR}"]
    return warray


def weighted_qcut_variable(values, weights, numbins):
    """
    Return weighted quantile bin variable for specified values.
    """
    quantiles = np.linspace(0, 1, numbins + 1)
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, right=True, labels=False)
    return bins.sort_index()


def sorted_vdf_with_itxbin(vdf: pd.DataFrame, numbins: int):
    """
    Expects vdf to include two area weights arrays and an itax array.
    Returns itax-sorted vdf with added itxbin given NUM_BINS environ variable.
    """
    # check contents of vdf
    assert "wght1" in vdf, "wght1 variable not in vdf"
    assert "wght2" in vdf, "wght2 variable not in vdf"
    assert "itax" in vdf, "itax variable not in vdf"
    # add itxbin variable to vdf
    if ITXBINS_DEFINED_USING_AREA_WEIGHTS:
        wght = 0.5 * (vdf.wght1 + vdf.wght2)
    else:  # using national weights
        wght = weights_array(STORAGE_FOLDER / "output" / "tmd_weights.csv.gz")
    vdf["itxbin"] = weighted_qcut_variable(vdf.itax, wght, numbins)
    # return sorted vdf
    vdf.sort_values("itax", inplace=True)
    return vdf


# -- High-level logic of the script:


def main(
    wname1: str,
    wpath1: Path,
    wname2: str,
    wpath2: Path,
    numbins: int,
    dump: bool,
):
    """
    Conduct chi-square two-variable test using the WGHT1 weights (wpath1)
    and the WGHT2 weights (wpath2), which are weights for the same area
    generated using different targets, using specified income tax numbins.
    """
    # read cached iitax amounts, which are the same for each area weights file
    itax = np.load(CACHED_ITAX_PATH)

    # construct vdf DataFrame and sort it and add itxbin variable to it
    wght1 = weights_array(wpath1)
    wght2 = weights_array(wpath2)
    assert len(wght1) == len(wght2) == len(itax)
    vdf = pd.DataFrame({"wght1": wght1, "wght2": wght2, "itax": itax})
    vdf = sorted_vdf_with_itxbin(vdf, numbins)
    if dump:
        print("***** vdf=\n", vdf)
        pct = np.linspace(0.1, 0.9, num=9)
        print("***** vdf.describe=\n", vdf.describe(percentiles=pct))

    # prepare frequency data for chi-square test
    unweighted_count = len(vdf)
    wght1_sum = wght1.sum()
    wght2_sum = wght2.sum()
    if dump:
        print(f"***** {wname1:>5}_weight_total= {wght1_sum:.5f}")
        print(f"***** {wname2:>5}_weight_total= {wght2_sum:.5f}")
        ratio = wght2_sum / wght1_sum
        print(f"*****       ==> ratio= {ratio:.8f}")
    gvdf = vdf.groupby(by=["itxbin"], sort=False, observed=True)
    if dump:
        itxtop = gvdf["itax"].max()
        print("***** itax_at_bin_top=\n", itxtop)
    freq1 = (unweighted_count / wght1_sum) * gvdf["wght1"].sum()
    freq2 = (unweighted_count / wght2_sum) * gvdf["wght2"].sum()
    if dump:
        fdf = pd.DataFrame({"freq1": freq1, "freq2": freq2})
        pct = np.linspace(0.1, 0.9, num=9)
        print("***** freq.describe=\n", fdf.describe(percentiles=pct))
    assert len(freq1) == len(freq2)
    assert np.allclose([freq1.sum()], [freq2.sum()])
    witax1 = (wght1 * itax).sum() * 1e-9
    witax2 = (wght2 * itax).sum() * 1e-9
    witax_ratio = witax2 / witax1
    print(
        f"{wname1},{wname2}_weighted_iitax($B)= {witax1:.3f} {witax2:.3f}"
        f"  ===>  ratio= {witax_ratio:.4f}"
    )
    min_bin_cnt = min(np.min(freq1), np.min(freq2))
    print(f"numbins,minimum_bin_freqency_count= {numbins} {min_bin_cnt:.1f}")
    if min_bin_cnt < 5:
        if numbins > MINIMUM_NUM_BINS:
            print("WARNING: reduce value of numbins command-line argument")
            print("         to get minimum_bin_fredquency_count above five")
        else:
            print("WARNING: minimum_bin_fredquency_count is below five")
            print("         so use TMD_AREA_ITXBINS=1 environment variable")
        print(USAGE)
        return 1

    # conduct chi-square test
    print("Conducting chi-square two-variable independence test:")
    ctable = pd.DataFrame({"1": freq1, "2": freq2})
    res = chi2_contingency(ctable)
    print(f"Xsq_statistic= {res.statistic:.3f}")
    print(f"Xsq_test_pval= {res.pvalue:.3f}   where dof= {res.dof}")
    if dump:
        print("***** Xsq(pval=0.05,dof= 100)=  124.34")
        print("***** Xsq(pval=0.05,dof= 200)=  233.99")
        print("***** Xsq(pval=0.05,dof=1000)= 1074.68")
        print("***** Xsq(pval=0.05,dof=2000)= 2105.15")

    return 0


if __name__ == "__main__":
    if "TMD_AREA_ITXBINS" in os.environ:
        ITXBINS_DEFINED_USING_AREA_WEIGHTS = True
    awname1, awpath1, awname2, awpath2, num_bins, dump_ = check_arguments()
    RCODE = main(awname1, awpath1, awname2, awpath2, num_bins, dump_)
    sys.exit(RCODE)
