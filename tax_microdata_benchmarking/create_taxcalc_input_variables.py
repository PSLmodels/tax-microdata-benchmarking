"""
Construct tmd.csv.gz, a Tax-Calculator-style input variable file for 2021.
"""

from tax_microdata_benchmarking.create_flat_file import (
    create_stacked_flat_file,
)
from tax_microdata_benchmarking.adjust_qbi import (
    add_pt_w2_wages,
)
import taxcalc as tc


TAXYEAR = 2021
INITIAL_PT_W2_WAGES_SCALE = 0.31738


def create_variable_file():
    """
    Create Tax-Calculator-style input variable file for TAXYEAR.
    """
    # construct dataframe containing input and output variables
    vdf = create_stacked_flat_file(
        target_year=TAXYEAR,
        pt_w2_wages_scale=INITIAL_PT_W2_WAGES_SCALE,
    )
    vdf.FLPDYR = TAXYEAR
    (vdf, pt_w2_wages_scale) = add_pt_w2_wages(vdf)
    abs_diff = abs(pt_w2_wages_scale - INITIAL_PT_W2_WAGES_SCALE)
    if abs_diff > 1e-6:
        print(f"WARNING: FINAL vs INITIAL scale diff = {abs_diff:.6f}")
        print(f"  INITIAL pt_w2_wages_scale = {INITIAL_PT_W2_WAGES_SCALE:.6f}")
        print(f"    FINAL pt_w2_wages_scale = {pt_w2_wages_scale:.6f}")
    # streamline variables dataframe
    rec = tc.Records(
        data=vdf,
        start_year=TAXYEAR,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
    )
    vdf.drop(columns=rec.IGNORED_VARS, inplace=True)
    vdf.e00200p = vdf.e00200p.to_numpy().round()
    vdf.e00200s = vdf.e00200s.to_numpy().round()
    vdf.e00200 = vdf.e00200p + vdf.e00200s
    vdf.e00900p = vdf.e00900p.to_numpy().round()
    vdf.e00900s = vdf.e00900s.to_numpy().round()
    vdf.e00900 = vdf.e00900p + vdf.e00900s
    vdf.e02100p = vdf.e02100p.to_numpy().round()
    vdf.e02100s = vdf.e02100s.to_numpy().round()
    vdf.e02100 = vdf.e02100p + vdf.e02100s
    # write streamlined variables dataframe to CSV-formatted file
    vdf.to_csv(
        "tmd.csv.gz",
        index=False,
        float_format="%.0f",
        compression="gzip",
    )


if __name__ == "__main__":
    create_variable_file()
