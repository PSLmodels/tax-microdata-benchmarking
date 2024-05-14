"""
Construct tmd.csv, a Tax-Calculator-style input variable file for 2021.
"""


def create_variable_file(
    initial_pt_w2_wages_scale=0.320,
    create_from_scratch=False,
    write_file=True,
):
    """
    Create Tax-Calculator-style input variable file for 2021.
    """
    import taxcalc as tc
    from tax_microdata_benchmarking.datasets.policyengine.puf_ecps import (
        create_puf_ecps_flat_file,
    )
    from tax_microdata_benchmarking.utils.qbi import (
        add_pt_w2_wages,
    )
    from tax_microdata_benchmarking.storage import STORAGE_FOLDER

    taxyear = 2021
    # construct dataframe containing input and output variables
    print(f"Creating {taxyear} PUF-ECPS file using initial pt_w2_wages_scale")
    vdf = create_puf_ecps_flat_file(
        target_year=taxyear,
        pt_w2_wages_scale=initial_pt_w2_wages_scale,
        from_scratch=create_from_scratch,
    )
    vdf.FLPDYR = taxyear
    (vdf, pt_w2_wages_scale) = add_pt_w2_wages(vdf)
    abs_diff = abs(pt_w2_wages_scale - initial_pt_w2_wages_scale)
    if abs_diff > 1e-6:
        msg = (
            f"\nFINAL vs INITIAL scale diff = {abs_diff:.6f}"
            f"\n  INITIAL pt_w2_wages_scale = {initial_pt_w2_wages_scale:.6f}"
            f"\n    FINAL pt_w2_wages_scale = {pt_w2_wages_scale:.6f}"
        )
        if abs_diff < 1e-3:
            print("WARNING:", msg[1:])
        else:
            raise ValueError(msg)
    # streamline dataframe so that it includes only input variables
    rec = tc.Records(
        data=vdf,
        start_year=taxyear,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
    )
    vdf.drop(columns=rec.IGNORED_VARS, inplace=True)
    # round all float variables to nearest integer except for weights
    weights = vdf.s006.copy()
    vdf = vdf.astype(int)
    vdf.s006 = weights
    for var in ["e00200", "e00900", "e02100"]:
        vdf[var] = vdf[f"{var}p"] + vdf[f"{var}s"]
    # write streamlined variables dataframe to CSV-formatted file
    if write_file:
        tmd_csv_fname = STORAGE_FOLDER / "output" / "tmd.csv.gz"
        vdf.to_csv(tmd_csv_fname, index=False, float_format="%.2f")


if __name__ == "__main__":
    create_variable_file()
