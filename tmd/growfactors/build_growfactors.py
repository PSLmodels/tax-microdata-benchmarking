"""
Stage 2 of the growth factors pipeline.

Transforms data/Stage_I_factors.csv (written by build_stage1_factors.py)
and data/benefit_growth_rates.csv into a Tax-Calculator-style growfactors
file written to tmd/storage/input.  That file is the input read by
tmd/create_raw_growth_factors.py during `make data`.

This logic was migrated from the TaxData repository
(puf_stage1/factors_finalprep.py).
"""

import argparse
import sys

import pandas as pd

from tmd.growfactors import DATA_FOLDER
from tmd.storage import STORAGE_FOLDER

# pylint: disable=invalid-name

FIRST_BENEFIT_YEAR = 2014
FIRST_DATA_YEAR = 2011

# vintage of the source data currently in tmd/growfactors/data;
# also the default used by the tmd/growfactors/Makefile
DEFAULT_VINTAGE = "26"

STAGE1_FILE = DATA_FOLDER / "Stage_I_factors.csv"

BENEFIT_NAMES = [
    "mcare",
    "mcaid",
    "ssi",
    "snap",
    "wic",
    "housing",
    "tanf",
    "vet",
]
GF_BENEFIT_NAMES = [f"ABEN{bname.upper()}" for bname in BENEFIT_NAMES]

# variables used by Tax-Calculator; all others are dropped from the output
TC_USED_VARS = set(
    [
        "ABOOK",
        "ACGNS",
        "ACPIM",
        "ACPIU",
        "ADIVS",
        "AINTS",
        "AIPD",
        "ASCHCI",
        "ASCHCL",
        "ASCHEI",
        "ASCHEL",
        "ASCHF",
        "ASOCSEC",
        "ATXPY",
        "AUCOMP",
        "AWAGE",
        "ABENOTHER",
    ]
    + GF_BENEFIT_NAMES
)


def growfactors_path(vintage):
    """
    Return path of the growfactors file produced for the specified vintage.
    """
    return STORAGE_FOLDER / "input" / f"taxdata{vintage}_growfactors.csv"


def format_factor(value):
    """
    Return CSV representation of a growth factor rounded to six decimals
    with insignificant trailing zeros removed, so that an exact 1.0 is
    written as "1" rather than "1.0".
    """
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def benefit_growth_factors(years):
    """
    Return DataFrame of "one plus annual proportion change" benefit
    factors covering exactly the specified years.
    """
    bgr_all = pd.read_csv(
        DATA_FOLDER / "benefit_growth_rates.csv", index_col="YEAR"
    )
    keep_cols = [f"{bname}_average_benefit" for bname in BENEFIT_NAMES]
    bgr_raw = bgr_all[keep_cols]
    bgr_raw.columns = GF_BENEFIT_NAMES
    bgf = 1.0 + bgr_raw.astype("float64").pct_change()

    # specify first row values because pct_change leaves first year undefined
    bgf.loc[FIRST_BENEFIT_YEAR, :] = 1.0

    # add rows of ones for years from FIRST_DATA_YEAR thru first benefit year
    ones = [1.0] * len(BENEFIT_NAMES)
    for year in range(FIRST_DATA_YEAR, FIRST_BENEFIT_YEAR):
        row = pd.DataFrame(data=[ones], columns=GF_BENEFIT_NAMES, index=[year])
        bgf = pd.concat([bgf, row], verify_integrity=True)
    bgf.sort_index(inplace=True)

    # Pad with rows of ones for any year beyond the last year in
    # benefit_growth_rates.csv.  Without this, benefit factors for those
    # years are silently NaN in the output file, which is what TaxData's
    # factors_finalprep.py does now that its benefit_growth_rates.csv
    # ends before the last Stage_I_factors.csv year.
    bgf = bgf.reindex(years, fill_value=1.0)

    # round converted factors to six decimal digits of accuracy
    return bgf.round(6)


def create_growfactors_file(vintage):
    """
    Create the growfactors file for the specified data vintage.
    """
    # read the level indexes written by build_stage1_factors.py
    data = pd.read_csv(STAGE1_FILE, index_col="YEAR")

    bgf = benefit_growth_factors(data.index)

    # convert some aggregate factors into per-capita factors
    elderly_pop = data["APOPSNR"]
    data["ASOCSEC"] = data["ASOCSEC"] / elderly_pop
    pop = data["APOPN"]
    for var in [
        "AWAGE",
        "ATXPY",
        "ASCHCI",
        "ASCHCL",
        "ASCHF",
        "AINTS",
        "ADIVS",
        "ASCHEI",
        "ASCHEL",
        "ACGNS",
        "ABOOK",
        "ABENEFITS",
    ]:
        data[var] = data[var] / pop
    data.rename(columns={"ABENEFITS": "ABENOTHER"}, inplace=True)

    # convert factors into "one plus annual proportion change" format
    data = 1.0 + data.pct_change()

    # specify first row values because pct_change leaves first year undefined
    data.loc[FIRST_DATA_YEAR, :] = 1.0

    # round converted factors to six decimal digits of accuracy
    data = data.round(6)

    # combine the two DataFrames and drop variables Tax-Calculator ignores
    gfdf = pd.concat([data, bgf], axis="columns", verify_integrity=True)
    gfdf = gfdf.drop(set(list(gfdf)) - TC_USED_VARS, axis=1)

    outfile = growfactors_path(vintage)
    gfdf.to_csv(outfile, index_label="YEAR", float_format=format_factor)
    print(f"Wrote {outfile}")


def main():
    """
    High-level script logic.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Write a Tax-Calculator-style growfactors file for the "
            "specified data vintage to tmd/storage/input."
        )
    )
    parser.add_argument(
        "--vintage",
        default=DEFAULT_VINTAGE,
        help=(
            "data vintage used in the output file name "
            f"(default: {DEFAULT_VINTAGE})"
        ),
    )
    args = parser.parse_args()
    create_growfactors_file(args.vintage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
