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

USAGE: python chisquare_test.py WGHT1 WGHT2
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from tmd.areas import AREAS_FOLDER
from tmd.storage import STORAGE_FOLDER

B_N1 = [b * -1e3 for b in range(20, 4, -2)]
B_N2 = [-4.0e3, -3.5e3, -3.0e3, -2.5e3, -2.0e3, -1.5e3, -1.0e3, -0.5e3]
B_LO = [b * 1e2 for b in range(0, 100, 5)]
B_H1 = [b * 1e3 for b in range(10, 20, 1)]
B_H2 = [20e3, 25e3] + [b * 1e4 for b in range(3, 29)]
ITAX_BINS = [-9e99] + B_N1 + B_N2 + B_LO + B_H1 + B_H2 + [9e99]

CACHED_ITAX_PATH = STORAGE_FOLDER / "output" / "cached_iitax.npy"

TAX_YEAR = 2021

DEFAULT_NUM_BINS = 2000

DUMP_DETAILS = False


def check_command_line_arguments():
    """
    Check validity of arguments and return (wname1, wpath1, wname2, wpath2)
    tuple containing name of and path to the WGHT1 and WGHT2 weights files.
    """
    usage = "USAGE: python chisquare_test.py WGHT1 WGHT2\n"
    if len(sys.argv) != 3:
        sys.stderr.write(
            f"ERROR: two command-line arguments are required\n{usage}"
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
    if not all_ok:
        sys.stderr.write(f"{usage}")
        sys.exit(1)
    return (wname1, wpath1, wname2, wpath2)


def weights_array(wghts_path: Path):
    """
    Return array containing TAX_YEAR weights in area weights file with wpath.
    """
    wdf = pd.read_csv(wghts_path)
    warray = wdf[f"WT{TAX_YEAR}"]
    return warray


def weighted_qcut(values, weights, numbins):
    """
    Return weighted quantile cuts from a given series, values.
    """
    quantiles = np.linspace(0, 1, numbins + 1)
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, right=True, labels=False)
    return bins.sort_index()


def sorted_vdf_with_itxbin(vdf: pd.DataFrame):
    """
    Expects vdf to include two area weights arrays and an itax array.
    Returns itax-sorted vdf with added itxbin given NUM_BINS environ variable.
    """
    # determine number of itax bins to construct
    numbins = DEFAULT_NUM_BINS
    if "NUM_BINS" in os.environ:
        numbins = int(os.environ["NUM_BINS"])
    # check contents of vdf
    assert "wght1" in vdf, "wght1 variable not in vdf"
    assert "wght2" in vdf, "wght2 variable not in vdf"
    assert "itax" in vdf, "itax variable not in vdf"
    # add itxbin variable to vdf
    avg_wght = 0.5 * (vdf.wght1 + vdf.wght2)
    vdf["itxbin"] = weighted_qcut(vdf.itax, avg_wght, numbins)
    # return sorted vdf
    vdf.sort_values("itax", inplace=True)
    return vdf


# -- High-level logic of the script:


def main(wname1: str, wpath1: Path, wname2: str, wpath2: Path):
    """
    Conduct chi-square two-variable test using the WGHT1 weights (wpath1)
    and the WGHT2 weights (wpath2), which are weights for the same area
    generated using different targets.
    """
    # read cached iitax amounts, which are the same for each area weights file
    itax = np.load(CACHED_ITAX_PATH)

    # construct vdf DataFrame and sort it and add itxbin variable to it
    wght1 = weights_array(wpath1)
    wght2 = weights_array(wpath2)
    assert len(wght1) == len(wght2) == len(itax)
    witax1 = (wght1 * itax).sum() * 1e-9
    witax2 = (wght2 * itax).sum() * 1e-9
    witax_ratio = witax2 / witax1
    vdf = pd.DataFrame({"wght1": wght1, "wght2": wght2, "itax": itax})
    vdf = sorted_vdf_with_itxbin(vdf)

    # prepare frequency data for chi-square test
    unweighted_count = len(vdf)
    wght1_sum = wght1.sum()
    wght2_sum = wght2.sum()
    weight_ratio = wght2_sum / wght1_sum
    print(f"{wname2}/{wname1}_weight_total_ratio= {weight_ratio:.4f}")
    print(
        f"{wname1},{wname2}_weighted_iitax($B)= {witax1:.3f} {witax2:.3f}"
        f"  ===>  ratio= {witax_ratio:.4f}"
    )
    gvdf = vdf.groupby(by=["itxbin"], sort=False, observed=True)
    freq1 = (unweighted_count / wght1_sum) * gvdf["wght1"].sum()
    freq2 = (unweighted_count / wght2_sum) * gvdf["wght2"].sum()
    assert len(freq1) == len(freq2)
    assert np.allclose([freq1.sum()], [freq2.sum()])
    min_cell_count = min(np.min(freq1), np.min(freq2))
    print(f"minimum_bin_freqency_count= {min_cell_count:.1f}")
    if min_cell_count < 5:
        print("WARNING: reduce value of NUM_BINS environment variable")
        print("         to get minimum_bin_fredquency_count above five")

    # conduct chi-square test
    print("Conducting chi-square two-variable independence test:")
    ctable = pd.DataFrame({"1": freq1, "2": freq2})
    res = chi2_contingency(ctable)
    print(f"Xsq_statistic= {res.statistic:.3f}")
    print(f"Xsq_test_pval= {res.pvalue:.3f}   where dof= {res.dof}")

    return 0


if __name__ == "__main__":
    awname1, awpath1, awname2, awpath2 = check_command_line_arguments()
    RCODE = main(awname1, awpath1, awname2, awpath2)
    sys.exit(RCODE)
