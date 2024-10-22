"""
Conduct chi-square test of the similarity of two weights distributions
for the same area that are generated using different targets.  The two area
weights files are assumed to be located in the tmd/areas/weights folder.
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

DUMP_DETAILS = False


def check_command_line_arguments():
    """
    Check validity of arguments and return (wpath1, wpath2) tuple
    containing path to WGHT1 and WGHT2 weights files, respectively.
    """
    usage = "USAGE: python chisquare_test.py WGHT1 WGHT2\n"
    if len(sys.argv) != 3:
        sys.stderr.write(
            f"ERROR: two command-line arguments are required\n{usage}"
        )
        sys.exit(1)
    all_ok = True
    wpath1 = AREAS_FOLDER / "weights" / f"{sys.argv[1]}_tmd_weights.csv.gz"
    if not wpath1.exists():
        sys.stderr.write(
            f"ERROR: WGHT1 {str(wpath1)} file does not exist\n"
        )
        all_ok = False
    wpath2 = AREAS_FOLDER / "weights" / f"{sys.argv[2]}_tmd_weights.csv.gz"
    if not wpath2.exists():
        sys.stderr.write(
            f"ERROR: WGHT2 {str(wpath2)} file does not exist\n"
        )
        all_ok = False
    if not CACHED_ITAX_PATH.exists():
        sys.stderr.write(
            f"ERROR: {str(CACHED_ITAX_PATH)} file does not exist\n"
        )
        all_ok = False
    if not all_ok:
        sys.stderr.write(f"{usage}")
        sys.exit(1)
    return (wpath1, wpath2)


def weights_array(wghts_path: Path):
    """
    Return array containing TAX_YEAR weights in area weights file with wpath.
    """
    wdf = pd.read_csv(wghts_path)
    warray = wdf[f"WT{TAX_YEAR}"] * 0.01
    return warray


# -- High-level logic of the script:


def main(wpath1: Path, wpath2: Path):
    """
    Conduct chi-square two-variable test using the WGHT1 weights (wpath1)
    and the WGHT2 weights (wpath2), which are weights for the same area
    generated using different targets.
    """
    if DUMP_DETAILS:
        for idx, low_edge in enumerate(ITAX_BINS):
            print(idx, low_edge)

    # read cached iitax amounts, which are the same for each area weights file
    iitax = np.load(CACHED_ITAX_PATH)
    wght1 = weights_array(wpath1)
    wght2 = weights_array(wpath2)
    assert len(wght1) == len(wght2) == len(iitax)
    witax1 = (wght1 * iitax).sum() * 1e-9
    witax2 = (wght2 * iitax).sum() * 1e-9
    witax_ratio = witax2 / witax1

    # create iitax bin variable called itxbin
    itxbin = pd.cut(iitax, bins=ITAX_BINS, right=False)
    assert len(itxbin) == len(wght1)

    # compute total weight in each iitax bin
    df1 = pd.DataFrame({"iitax": iitax, "s006": wght1, "itxbin": itxbin})
    df2 = pd.DataFrame({"iitax": iitax, "s006": wght2, "itxbin": itxbin})
    wght1_bin = df1.groupby("itxbin", observed=True)["s006"].sum()
    wght2_bin = df2.groupby("itxbin", observed=True)["s006"].sum()
    assert len(wght1_bin) == len(wght2_bin)
    num_bins = len(wght1_bin)
    if DUMP_DETAILS:
        wght1_tot = df1.s006.sum()
        for idx, wght in enumerate(wght1_bin):
            pct = 100 * wght / wght1_tot
            print(f"{idx:3d}  {wght:10.2f}  {pct:5.2f}")

    # prepare frequency data for chi-square test
    unweighted_count = len(df1)
    wght1_sum = df1.s006.sum()
    wght2_sum = df2.s006.sum()
    weight_ratio = wght2_sum / wght1_sum
    print(f"WGHT2/WGHT1_weight_total_ratio= {weight_ratio:.4f}")
    print(
        f"WGHT1,WGHT2_weighted_iitax($B)= {witax1:.3f} {witax2:.3f}"
        f"  ===>  ratio= {witax_ratio:.4f}"
    )
    freq1 = unweighted_count * wght1_bin / wght1_sum
    freq2 = unweighted_count * wght2_bin / wght2_sum
    assert len(freq1) == len(freq2)
    assert np.allclose([freq1.sum()], [freq2.sum()])
    min_cell_count = min(np.min(freq1), np.min(freq2))
    print(f"number_of_iitax_bins= {num_bins}")
    print(f"minimum_bin_freqency_count= {min_cell_count:.1f}")

    # conduct chi-square test
    print("Conducting chi-square two-variable independence test:")
    ctable = pd.DataFrame({"1": freq1, "2": freq2})
    res = chi2_contingency(ctable)
    print(f"Xsq_statistic= {res.statistic:.3f}")
    print(f"Xsq_test_pval= {res.pvalue:.3f}   where dof= {res.dof}")

    return 0


if __name__ == "__main__":
    awpath1, awpath2 = check_command_line_arguments()
    RCODE = main(awpath1, awpath2)
    sys.exit(RCODE)
