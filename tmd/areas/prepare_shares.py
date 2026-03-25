# pylint: disable=import-outside-toplevel
"""
Pre-compute SOI geographic shares for area target preparation.

Shares are the stable component — they depend only on SOI data
and geographic crosswalks, which change rarely. TMD national sums
(the volatile component) are applied later in prepare_targets.

The shares file stores each area's fraction of the national total
for every (varname, count, fstatus, agistub) combination defined
in ALL_SHARING_MAPPINGS.  XTOT (population) is stored as a fixed
target value rather than a share.

Usage:
    python -m tmd.areas.prepare_shares --scope cds
    python -m tmd.areas.prepare_shares --scope states
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.areas.prepare.constants import (
    ALL_SHARING_MAPPINGS,
    CD_AGI_CUTS,
    STATE_AGI_CUTS,
    AreaType,
)

_SHARES_DIR = Path(__file__).parent / "prepare" / "data"

# Extended target mappings: (tmd_varname, soi_base, count, fstatus, desc)
# These supplement ALL_SHARING_MAPPINGS with additional variables
# for extended area targets.
EXTENDED_SHARING_MAPPINGS = [
    # SOI-shared income/deduction variables
    ("e01700", "01700", 0, 0, "Taxable pensions"),
    ("c02500", "02500", 0, 0, "Taxable Social Security"),
    ("e01400", "01400", 0, 0, "Taxable IRA distributions"),
    ("capgains_net", "01000", 0, 0, "Net capital gains"),
    ("e00600", "00600", 0, 0, "Ordinary dividends"),
    ("e00900", "00900", 0, 0, "Business/professional income"),
    ("c19200", "19300", 0, 0, "Interest deduction (mortgage)"),
    ("c19700", "19700", 0, 0, "Charitable contributions"),
    # SALT components (SOI proxy for Census)
    ("e18400", "18425", 0, 0, "SALT income/sales"),
    ("e18500", "18500", 0, 0, "SALT real estate"),
    # Credits — amounts
    ("eitc", "59660", 0, 0, "EITC amount"),
    ("ctc_total", "07225", 0, 0, "CTC+ACTC amount"),
    # Credits — nonzero counts
    ("eitc", "59660", 2, 0, "EITC nz-count"),
    ("ctc_total", "07225", 2, 0, "CTC+ACTC nz-count"),
]


def _add_ctc_total(base_targets):
    """
    Add derived CTC total rows (07225 + 11070) to base_targets.

    The SOI CD file has a07225 (nonrefundable CTC/ODC) and a11070
    (refundable ACTC) separately. We sum them for ctc_total shares.
    """
    ctc_parts = base_targets[
        base_targets["basesoivname"].isin(["07225", "11070"])
    ].copy()
    if ctc_parts.empty:
        return base_targets

    # Group by all geographic + classification columns and sum
    # the two CTC components (07225 + 11070).
    # Include both 'area' and 'stabbr' if present so neither is lost.
    group_candidates = [
        "stabbr",
        "area",
        "count",
        "scope",
        "fstatus",
        "agistub",
        "agilo",
        "agihi",
    ]
    group_cols = [c for c in group_candidates if c in ctc_parts.columns]

    ctc_sum = ctc_parts.groupby(group_cols)["target"].sum().reset_index()
    # Copy non-grouped metadata from a sample row
    sample = ctc_parts.iloc[0].copy()
    existing = set(ctc_sum.columns)
    for col in ctc_parts.columns:
        if col not in existing and col != "target":
            ctc_sum[col] = sample.get(col)
    ctc_sum["basesoivname"] = "07225"  # use 07225 as share basis

    # Remove original 07225 and 11070 rows — they are now replaced
    # by the combined ctc_total rows.  Without this, downstream
    # share computation finds two conflicting rows for basesoivname
    # "07225": one with just nonrefundable CTC, one with the correct
    # CTC + ACTC total.
    base_targets = base_targets[
        ~base_targets["basesoivname"].isin(["07225", "11070"])
    ]

    return pd.concat([base_targets, ctc_sum], ignore_index=True)


def _agi_bin_boundaries(agistub, agi_cuts):
    """Convert agistub number to (agilo, agihi) boundaries."""
    if agistub == 0:
        return -9e99, 9e99
    lo = agi_cuts[agistub - 1]
    hi = agi_cuts[agistub]
    # Replace inf with 9e99 to match target file convention
    if lo == float("-inf"):
        lo = -9e99
    if hi == float("inf"):
        hi = 9e99
    return lo, hi


def compute_shares(area_type, area_data_year=2022):
    """
    Compute SOI geographic shares for all areas.

    Returns a DataFrame with columns:
        area, varname, count, scope, fstatus, agistub, agilo, agihi,
        soi_share, fixed_target, description

    For XTOT rows: soi_share is NaN, fixed_target is population.
    For shared rows: fixed_target is NaN, soi_share is the fraction.
    """
    repo_root = Path(__file__).parent.parent.parent

    if area_type == AreaType.CD:
        return _compute_cd_shares(repo_root, area_data_year)
    if area_type == AreaType.STATE:
        return _compute_state_shares(repo_root, area_data_year)
    raise ValueError(f"Unsupported area_type: {area_type}")


def _compute_cd_shares(repo_root, area_data_year):
    """Compute shares for congressional districts."""
    from tmd.areas.prepare.soi_cd_data import (
        apply_crosswalk,
        compute_cd_population,
        create_cd_base_targets,
        create_cd_soilong,
        load_crosswalk,
    )
    from tmd.areas.prepare.target_sharing import (
        compute_cd_soi_shares,
    )

    agi_cuts = CD_AGI_CUTS
    soi_dir = repo_root / "tmd" / "areas" / "prepare" / "data" / "soi_cds"

    # SOI data ingestion + crosswalk
    cd117_long = create_cd_soilong(soi_dir, years=[area_data_year])
    crosswalk = load_crosswalk()
    cd118_long = apply_crosswalk(cd117_long, crosswalk)
    cd_pop = compute_cd_population()
    base_targets = create_cd_base_targets(cd118_long, cd_pop, area_data_year)

    # Add derived CTC total (07225 + 11070) to base_targets
    # so that shares can be computed for ctc_total
    base_targets = _add_ctc_total(base_targets)

    # Compute SOI shares for base + extended mappings
    all_mappings = ALL_SHARING_MAPPINGS + EXTENDED_SHARING_MAPPINGS
    soi_shares = compute_cd_soi_shares(base_targets, all_mappings)

    # Filter to exact mapping combos
    wanted = pd.DataFrame(
        [
            {
                "basesoivname": m[1],
                "count": m[2],
                "fstatus": m[3],
            }
            for m in all_mappings
        ]
    ).drop_duplicates()
    soi_shares = soi_shares.merge(
        wanted,
        on=["basesoivname", "count", "fstatus"],
        how="inner",
    )

    return _format_shares(
        soi_shares,
        base_targets,
        agi_cuts,
        area_col="area",
        all_mappings=all_mappings,
    )


def _compute_state_shares(repo_root, area_data_year):
    """Compute shares for states."""
    from tmd.areas.prepare.census_population import (
        get_state_population,
    )
    from tmd.areas.prepare.soi_state_data import (
        create_soilong,
        create_state_base_targets,
    )
    from tmd.areas.prepare.target_sharing import (
        compute_soi_shares,
    )

    agi_cuts = STATE_AGI_CUTS
    soi_dir = repo_root / "tmd" / "areas" / "prepare" / "data" / "soi_states"

    soilong = create_soilong(soi_dir, years=[area_data_year])
    pop_df = get_state_population(area_data_year)
    base_targets = create_state_base_targets(soilong, pop_df, area_data_year)

    # Add derived CTC total (07225 + 11070) to base_targets
    base_targets = _add_ctc_total(base_targets)

    # Compute SOI shares for base + extended mappings
    all_mappings = ALL_SHARING_MAPPINGS + EXTENDED_SHARING_MAPPINGS
    soi_shares = compute_soi_shares(base_targets, all_mappings)

    # Filter to exact mapping combos
    wanted = pd.DataFrame(
        [
            {
                "basesoivname": m[1],
                "count": m[2],
                "fstatus": m[3],
            }
            for m in all_mappings
        ]
    ).drop_duplicates()
    soi_shares = soi_shares.merge(
        wanted,
        on=["basesoivname", "count", "fstatus"],
        how="inner",
    )

    # States use stabbr; add area column
    _EXCLUDE = {"US", "OA", "PR"}
    soi_shares = soi_shares[~soi_shares["stabbr"].isin(_EXCLUDE)].copy()
    soi_shares["area"] = soi_shares["stabbr"]

    return _format_shares(
        soi_shares,
        base_targets,
        agi_cuts,
        area_col="area",
        all_mappings=all_mappings,
    )


def _format_shares(
    soi_shares, base_targets, agi_cuts, area_col, all_mappings=None
):
    """
    Format shares into the output schema.

    Maps SOI base variable names to TMD variable names, adds AGI
    bin boundaries, and appends XTOT (population) rows as fixed
    targets.
    """
    if all_mappings is None:
        all_mappings = ALL_SHARING_MAPPINGS

    # Build SOI-to-TMD mapping (one-to-many: multiple TMD vars
    # can share the same SOI base, e.g., e01500 and e01700 both
    # use SOI 01700)
    from collections import defaultdict

    soi_to_tmd_list = defaultdict(list)
    descriptions = {}
    for tmdvar, soi_base, cnt, fs, desc in all_mappings:
        soi_to_tmd_list[(soi_base, cnt, fs)].append(tmdvar)
        descriptions[(tmdvar, cnt, fs)] = desc

    # Map to TMD variable names (one share row → multiple output
    # rows if multiple TMD vars use the same SOI base)
    rows = []
    for _, r in soi_shares.iterrows():
        key = (r["basesoivname"], r["count"], r["fstatus"])
        tmdvars = soi_to_tmd_list.get(key, [])
        if not tmdvars:
            continue
        stub = int(r["agistub"])
        lo, hi = _agi_bin_boundaries(stub, agi_cuts)
        for tmdvar in tmdvars:
            desc = descriptions.get((tmdvar, r["count"], r["fstatus"]), "")
            rows.append(
                {
                    "area": r[area_col],
                    "varname": tmdvar,
                    "count": int(r["count"]),
                    "scope": 1,
                    "fstatus": int(r["fstatus"]),
                    "agistub": stub,
                    "agilo": lo,
                    "agihi": hi,
                    "soi_share": r["soi_share"],
                    "fixed_target": np.nan,
                    "description": desc,
                }
            )

    shares_df = pd.DataFrame(rows)

    # Add XTOT (population) as fixed targets
    xtot = base_targets.loc[base_targets["basesoivname"] == "XTOT"].copy()
    xtot_rows = []
    for _, r in xtot.iterrows():
        area = r.get("area", r.get("stabbr", ""))
        if area in ("US", "OA", "PR"):
            continue
        xtot_rows.append(
            {
                "area": area,
                "varname": "XTOT",
                "count": 0,
                "scope": 0,
                "fstatus": 0,
                "agistub": 0,
                "agilo": -9e99,
                "agihi": 9e99,
                "soi_share": np.nan,
                "fixed_target": r["target"],
                "description": "Population",
            }
        )

    xtot_df = pd.DataFrame(xtot_rows)
    result = pd.concat([xtot_df, shares_df], ignore_index=True)
    result = result.sort_values(
        ["area", "varname", "count", "fstatus", "agistub"]
    ).reset_index(drop=True)

    return result


def save_shares(area_type, area_data_year=2022):
    """Compute and save shares to CSV."""
    shares = compute_shares(area_type, area_data_year)

    suffix = "cds" if area_type == AreaType.CD else "states"
    outpath = _SHARES_DIR / f"{suffix}_shares.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    shares.to_csv(outpath, index=False, float_format="%.10g")

    n_areas = shares["area"].nunique()
    n_rows = len(shares)
    print(f"Saved {n_rows:,} rows for {n_areas} areas" f" to {outpath}")
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute SOI geographic shares",
    )
    parser.add_argument(
        "--scope",
        default="states",
        help="'states' or 'cds' (default: states)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="SOI data year (default: 2022)",
    )
    args = parser.parse_args()

    scope = args.scope.lower().strip()
    if scope == "cds":
        save_shares(AreaType.CD, args.year)
    elif scope == "states":
        save_shares(AreaType.STATE, args.year)
    else:
        print(f"Unknown scope: {args.scope}")


if __name__ == "__main__":
    main()
