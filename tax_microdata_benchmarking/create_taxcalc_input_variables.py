"""
Construct tmd.csv, a Tax-Calculator-style input variable file for 2021.
"""

TAXYEAR = 2021
DO_REWEIGHTING = True
INITIAL_W2_WAGES_SCALE = 0.19980
INCLUDE_ORIGINAL_WEIGHTS = True


def create_variable_file(write_file=True):
    """
    Create Tax-Calculator-style input variable file for TAXYEAR.
    """
    import taxcalc as tc
    from tax_microdata_benchmarking.datasets.tmd import create_tmd_2021
    from tax_microdata_benchmarking.utils.qbi import (
        add_pt_w2_wages,
    )
    from tax_microdata_benchmarking.storage import STORAGE_FOLDER

    # construct dataframe containing input and output variables
    print(f"Creating {TAXYEAR} PUF+CPS file assuming:")
    print(f"  DO_REWEIGHTING = {DO_REWEIGHTING}")
    print(f"  INITIAL_W2_WAGES_SCALE = {INITIAL_W2_WAGES_SCALE:.5f}")
    print(f"  INCLUDE_ORIGINAL_WEIGHTS = {INCLUDE_ORIGINAL_WEIGHTS}")
    vdf = create_tmd_2021()
    vdf.FLPDYR = TAXYEAR
    (vdf, pt_w2_wages_scale) = add_pt_w2_wages(vdf)
    abs_diff = abs(pt_w2_wages_scale - INITIAL_W2_WAGES_SCALE)
    msg = (
        f"  FINAL vs INITIAL scale diff = {abs_diff:.6f}\n"
        f"    INITIAL pt_w2_wages_scale = {INITIAL_W2_WAGES_SCALE:.6f}\n"
        f"      FINAL pt_w2_wages_scale = {pt_w2_wages_scale:.6f}"
    )
    print(msg)
    if abs_diff > 1e-6:
        raise ValueError("INITIAL and FINAL scale values are inconsistent")
    # streamline dataframe so that it includes only input variables
    rec = tc.Records(
        data=vdf,
        start_year=TAXYEAR,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
    )
    weights = vdf.s006.copy()
    if DO_REWEIGHTING and write_file:
        original_weights = vdf.s006_original.copy()
    else:
        original_weights = vdf.s006.copy()
    vdf.drop(columns=rec.IGNORED_VARS, inplace=True)
    # round all float variables to nearest integer except for weights
    vdf = vdf.astype(int)
    vdf.s006 = weights
    if INCLUDE_ORIGINAL_WEIGHTS:
        vdf["s006_original"] = original_weights
    for var in ["e00200", "e00900", "e02100"]:
        vdf[var] = vdf[f"{var}p"] + vdf[f"{var}s"]
    # write streamlined variables dataframe to CSV-formatted file
    if write_file:
        tmd_csv_fname = STORAGE_FOLDER / "output" / "tmd.csv.gz"
        print(f"Writing PUF+CPS file named {tmd_csv_fname}")
        vdf.to_csv(tmd_csv_fname, index=False, float_format="%.2f")


if __name__ == "__main__":
    create_variable_file()
