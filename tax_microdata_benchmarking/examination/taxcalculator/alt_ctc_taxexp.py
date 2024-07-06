"""
Tabulate alternative CTC tax-expenditure estimate.
"""

import os
import sys
import argparse
import pandas as pd


def main():
    """
    High-level script logic.
    """
    # parse command-line arguments
    usage_str = "python alt_ctc_taxexp.py INPUT YEAR [--help]"
    parser = argparse.ArgumentParser(
        prog="", usage=usage_str, description=__doc__
    )
    parser.add_argument(
        "INPUT",
        help="Name of CSV-formatted file containing the input dataset",
        nargs="?",
        default="",
    )
    parser.add_argument(
        "YEAR",
        help="Tax policy calendar year of tax-expenditure results",
        type=int,
        nargs="?",
        default=0,
    )
    args = parser.parse_args()

    # check command-line argument values
    args_ok = True
    if args.INPUT.endswith(".csv"):
        sys.stderr.write(f"ERROR: {args.INPUT} ends with .csv\n")
        args_ok = False
    if args.YEAR not in [23, 26]:
        sys.stderr.write(f"ERROR: YEAR {args.YEAR} is neither 23 nor 26\n")
        args_ok = False
    if not args_ok:
        sys.stderr.write(f"USAGE: {usage_str}\n")
        return 1

    # construct baseline and tax-expenditure reform dump output file names
    generic = f"{args.INPUT}-{args.YEAR}-#-xxx-#.csv"
    bas_fname = generic.replace("xxx", "clp")
    ref_fname = generic.replace("xxx", "ctc")
    args_ok = True
    if not os.path.isfile(bas_fname):
        sys.stderr.write(f"ERROR: {bas_fname} file does not exist\n")
        args_ok = False
    if not os.path.isfile(ref_fname):
        sys.stderr.write(f"ERROR: {ref_fname} file does not exist\n")
        args_ok = False
    if not args_ok:
        sys.stderr.write(f"USAGE: {usage_str}\n")
        return 1

    # read base and reform .csv dump output files
    bdf = pd.read_csv(bas_fname)
    rdf = pd.read_csv(ref_fname)

    # standard tax-expenditure tabulation using iitax
    btax = (bdf.s006 * bdf.iitax).sum() * 1e-9
    rtax = (rdf.s006 * rdf.iitax).sum() * 1e-9
    std_te = rtax - btax
    print(f"STD tax-expenditure($B)= {std_te:.3f}")

    # alternative tax-expenditure tabulation using ctc_total
    bctc = (bdf.s006 * bdf.ctc_total).sum() * 1e-9
    rctc = (rdf.s006 * rdf.ctc_total).sum() * 1e-9
    alt_te = bctc - rctc
    print(f"ALT tax-expenditure($B)= {alt_te:.3f}")

    # return no-error exit code
    return 0


if __name__ == "__main__":
    sys.exit(main())
