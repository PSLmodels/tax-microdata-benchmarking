"""
Conduct chi-square test of the independence of two area weights distributions
for the same area but generated using different targets.  The two area weights
files are assumed to be located in the tmd/areas/weights folder and the
national data files are assumed to be in the tmd/storage/output folder.  
All test output is written to stdout.

USAGE: python chisquare_test.py AREA1 AREA2
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from tmd.areas import AREAS_FOLDER
from tmd.storage import STORAGE_FOLDER
import taxcalc as tc

INFILE_PATH = STORAGE_FOLDER / "output" / "tmd.csv.gz"
GFFILE_PATH = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"

TAX_YEAR = 2021

B_N1 = [b * -1e3 for b in range(20, 4, -2)]
B_N2 = [-4.0e3, -3.5e3, -3.0e3, -2.5e3, -2.0e3, -1.5e3, -1.0e3, -0.5e3]
B_LO = [b * 1e2 for b in range(0, 100, 5)]
B_H1 = [b * 1e3 for b in range(10, 20, 1)]
B_H2 = [20e3, 25e3] + [b * 1e4 for b in range(3, 29)]
ITAX_BINS = [-9e99] + B_N1 + B_N2 + B_LO + B_H1 + B_H2 + [9e99]

DUMP_DETAILS = False
TWO_VARIABLES_TEST = True


def check_command_line_arguments():
    """
    Check validity of arguments and return (wpath1, wpath2) tuple
    containing path to AREA1 and AREA2 weights files, respectively.
    """
    usage = "USAGE: python chisquare_test.py AREA1 AREA2\n"
    if len(sys.argv) != 3:
        sys.stderr.write(
            f"ERROR: two command-line arguments are required\n{usage}"
        )
        sys.exit(1)
    all_ok = True
    area1 = sys.argv[1]
    wpath1 = AREAS_FOLDER / "weights" / f"{area1}_tmd_weights.csv.gz"
    if not wpath1.exists():
        sys.stderr.write(
            f"ERROR: AREA1 {str(wpath1)} file does not exist\n"
        )
        all_ok = False
    area2 = sys.argv[2]
    wpath2 = AREAS_FOLDER / "weights" / f"{area2}_tmd_weights.csv.gz"
    if not wpath2.exists():
        sys.stderr.write(
            f"ERROR: AREA2 {str(wpath2)} file does not exist\n"
        )
        all_ok = False
    # check existence of national input and growfactors files
    if not INFILE_PATH.exists():
        sys.stderr.write(
            f"ERROR: national {str(INFILE_PATH)} file does not exist\n"
        )
        all_ok = False
    if not GFFILE_PATH.exists():
        sys.stderr.write(
            f"ERROR: national {str(GFFILE_PATH)} file does not exist\n"
        )
        all_ok = False
    if not all_ok:
        sys.stderr.write(f"{usage}")
        sys.exit(1)
    return (wpath1, wpath2)


# -- High-level logic of the script:


def main(wpath1: Path, wpath2: Path):
    """
    Conduct chi-square independence test using the AREA1 weights (wpath1)
    and the AREA2 weights (wpath2), which are weights for the same area
    generated with different targets.
    """
    if DUMP_DETAILS:
        for idx, low_edge in enumerate(ITAX_BINS):
            print(idx, low_edge)

    # use Tax-Calculator to generate iitax amounts for each area weights file
    pol = tc.Policy()
    rec1 = tc.Records.tmd_constructor(
        data_path=INFILE_PATH,
        weights_path=wpath1,
        growfactors_path=GFFILE_PATH,
        exact_calculations=True,
    )
    calc = tc.Calculator(policy=pol, records=rec1)
    calc.advance_to_year(TAX_YEAR)
    calc.calc_all()
    df1 = calc.dataframe(["iitax", "s006"])
    del calc
    rec2 = tc.Records.tmd_constructor(
        data_path=INFILE_PATH,
        weights_path=wpath2,
        growfactors_path=GFFILE_PATH,
        exact_calculations=True,
    )
    calc = tc.Calculator(policy=pol, records=rec2)
    calc.advance_to_year(TAX_YEAR)
    calc.calc_all()
    df2 = calc.dataframe(["iitax", "s006"])
    del calc
    assert len(df1) == len(df2)
    assert np.allclose(df1.iitax, df2.iitax)

    # create iitax bin variable in the two dataframes
    itxbin = pd.cut(df1.iitax, bins=ITAX_BINS)
    assert len(itxbin) == len(df1)
    df1["itxbin"] = itxbin
    df2["itxbin"] = itxbin

    # compute total weight in each iitax bin
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
    print(f"AREA2/AREA1_weight_total_ratio= {weight_ratio:.4f}")
    freq1 = unweighted_count * wght1_bin / wght1_sum
    freq2 = unweighted_count * wght2_bin / wght2_sum
    assert len(freq1) == len(freq2)
    assert np.allclose([freq1.sum()], [freq2.sum()])
    min_cell_count = min(np.min(freq1), np.min(freq2))
    print(f"number_of_weight_bins= {num_bins}")
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
