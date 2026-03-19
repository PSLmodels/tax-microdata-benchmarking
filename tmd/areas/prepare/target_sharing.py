"""
Target sharing — derive state targets from TMD national totals.

Every targeted variable uses TMD national totals scaled by SOI
geographic shares:

    area_target = TMD_national_sum × (area_SOI / national_SOI)

This ensures area targets sum exactly to national TMD totals.

Year flexibility:
    The area_data_year (SOI geographic shares) and national_data_year
    (TMD levels from cached_allvars.csv) can differ. The convenience
    function ``prepare_area_targets()`` handles path resolution and
    loading for both years.

Note: Congressional District (CD) support will be added in a future PR.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from tmd.areas.prepare.constants import (
    ALL_SHARING_MAPPINGS,
    STATE_AGI_CUTS,
    AreaType,
)
from tmd.areas.prepare.census_population import (
    get_state_population,
)
from tmd.areas.prepare.soi_state_data import (
    create_soilong,
    create_state_base_targets,
)

# ---- TMD national sums --------------------------------


def compute_tmd_national_sums(
    cached_allvars_path: Path,
    all_mappings: List[Tuple[str, str, int, int, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Compute TMD national sums for all targeted variable combos.

    Handles three count types:
      - count=0 (amounts): sum(s006 * var) per AGI bin
      - count=1 (allcounts): weighted count, optionally by MARS
      - count=2 (nonzero counts): sum(s006 * (var != 0))

    Parameters
    ----------
    cached_allvars_path : Path
        Path to ``cached_allvars.csv``.
    all_mappings : list of (tmdvar, soi_base, count, fstatus, desc)
        All variable/count/fstatus combinations.
    agi_cuts : list of float
        AGI bin cut points.

    Returns
    -------
    pd.DataFrame
        Columns: tmdvar, basesoivname, count, fstatus, agistub,
        scope, tmdsum.
    """
    tmd = pd.read_csv(cached_allvars_path)
    tmd = tmd.loc[tmd["data_source"] == 1].copy()

    tmd["agistub"] = (
        pd.cut(
            tmd["c00100"],
            bins=agi_cuts,
            right=False,
            labels=False,
        ).astype(int)
        + 1
    )

    records = []
    stubs = sorted(tmd["agistub"].unique())

    for tmdvar, soi_base, count_type, fstatus, _ in all_mappings:
        for stub in stubs:
            mask = tmd["agistub"] == stub
            wts = tmd.loc[mask, "s006"]

            if count_type == 0:
                vals = tmd.loc[mask, tmdvar]
                tmdsum = (wts * vals).sum()
            elif count_type == 1:
                if fstatus == 0:
                    tmdsum = wts.sum()
                else:
                    mars_mask = tmd.loc[mask, "MARS"] == fstatus
                    tmdsum = wts[mars_mask].sum()
            elif count_type == 2:
                vals = tmd.loc[mask, tmdvar]
                tmdsum = (wts * (vals != 0)).sum()
            else:
                tmdsum = 0.0

            records.append(
                {
                    "tmdvar": tmdvar,
                    "basesoivname": soi_base,
                    "count": count_type,
                    "fstatus": fstatus,
                    "agistub": stub,
                    "tmdsum": tmdsum,
                }
            )

    sums_by_stub = pd.DataFrame(records)

    # Add totals (agistub=0)
    group = ["tmdvar", "basesoivname", "count", "fstatus"]
    totals = sums_by_stub.groupby(group)[["tmdsum"]].sum().reset_index()
    totals["agistub"] = 0
    sums_all = pd.concat([sums_by_stub, totals], ignore_index=True)
    sums_all["scope"] = 1
    return sums_all


# ---- SOI geographic shares ----------------------------


def compute_soi_shares(
    base_targets: pd.DataFrame,
    sharing_mappings: List[Tuple],
) -> pd.DataFrame:
    """
    Compute each area's share of the US total for sharer variables.

    For each (basesoivname, count, scope, fstatus, agistub):
      soi_share = area_target / US_target

    Shares are rescaled so 51 states (excl US, OA, PR) sum to 1.0.

    Parameters
    ----------
    base_targets : pd.DataFrame
        Base targets from SOI data.
    sharing_mappings : list
        5-tuple mappings. Element [1] is soi_base.

    Returns
    -------
    pd.DataFrame
        base_targets rows for sharer variables with added columns:
        soi_ussum, soi_share.
    """
    soi_bases = [m[1] for m in sharing_mappings]
    df = base_targets.loc[base_targets["basesoivname"].isin(soi_bases)].copy()
    group_cols = [
        "basesoivname",
        "count",
        "scope",
        "fstatus",
        "agistub",
    ]
    us_totals = df.loc[df["stabbr"] == "US", group_cols + ["target"]].rename(
        columns={"target": "soi_ussum"}
    )
    df = df.merge(us_totals, on=group_cols, how="left")
    df["soi_share"] = np.where(
        df["soi_ussum"] == 0, 0, df["target"] / df["soi_ussum"]
    )

    # Rescale shares so 51 states (excl US, OA, PR) sum to 1.0.
    # Raw SOI shares sum to ~99.5% because "Other Areas" (OA)
    # gets ~0.5%. Since the PUF represents only 51-state filers,
    # we redistribute proportionally.
    _EXCLUDE = {"US", "OA", "PR"}
    state_mask = ~df["stabbr"].isin(_EXCLUDE)
    group_sums = (
        df.loc[state_mask].groupby(group_cols)["soi_share"].transform("sum")
    )
    df.loc[state_mask, "soi_share"] = np.where(
        group_sums > 0,
        df.loc[state_mask, "soi_share"] / group_sums,
        0,
    )
    return df


# ---- Build shared targets --------------------------------


def _apply_sharing(
    soi_shares: pd.DataFrame,
    tmd_sums: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join SOI shares with TMD sums and compute targets.

    For non-total stubs: target = tmdsum × soi_share
    For agistub=0: target = sum of bin targets
    """
    join_keys = [
        "basesoivname",
        "scope",
        "fstatus",
        "count",
        "agistub",
    ]
    joined = soi_shares.merge(
        tmd_sums[join_keys + ["tmdvar", "tmdsum"]],
        on=join_keys,
        how="left",
    )

    # Compute bin-level targets
    joined["target"] = np.where(
        joined["agistub"] != 0,
        joined["tmdsum"] * joined["soi_share"],
        np.nan,
    )

    # Compute totals (agistub=0) as sum of bin targets
    group_cols = [
        "stabbr",
        "tmdvar",
        "basesoivname",
        "scope",
        "fstatus",
        "count",
    ]
    group_cols = [c for c in group_cols if c in joined.columns]

    bin_sums = (
        joined.loc[joined["agistub"] != 0]
        .groupby(group_cols)["target"]
        .sum()
        .reset_index()
        .rename(columns={"target": "target_total"})
    )
    joined = joined.merge(bin_sums, on=group_cols, how="left")
    joined.loc[joined["agistub"] == 0, "target"] = joined.loc[
        joined["agistub"] == 0, "target_total"
    ]
    joined = joined.drop(columns=["target_total"])
    return joined


def build_all_shares_targets(
    base_targets: pd.DataFrame,
    cached_allvars_path: Path,
    all_mappings: List[Tuple[str, str, int, int, str]],
    agi_cuts: List[float],
) -> pd.DataFrame:
    """
    Build enhanced targets where ALL variables use TMD x SOI shares.

    Every targeted variable (except XTOT population) uses:
      area_target = TMD_national_sum × (area_SOI / national_SOI)

    This ensures area targets sum exactly to national TMD totals.

    Parameters
    ----------
    base_targets : pd.DataFrame
        Base targets from SOI data (used for geographic shares).
    cached_allvars_path : Path
        Path to ``cached_allvars.csv``.
    all_mappings : list of (tmdvar, soi_base, count, fstatus, desc)
        All variable/count/fstatus combinations to share.
    agi_cuts : list of float
        AGI bin cut points.

    Returns
    -------
    pd.DataFrame
        Enhanced targets with sort column.
    """
    tmd_sums = compute_tmd_national_sums(
        cached_allvars_path, all_mappings, agi_cuts
    )

    soi_shares = compute_soi_shares(base_targets, all_mappings)

    # Filter to only the exact combos in all_mappings
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

    shared = _apply_sharing(soi_shares, tmd_sums)

    # Construct shared variable names
    shared["basesoivname"] = (
        "tmd"
        + shared["tmdvar"].str[1:]
        + "_shared_by_soi"
        + shared["basesoivname"]
    )
    shared["soivname"] = np.where(
        shared["count"] == 0,
        "a" + shared["basesoivname"],
        "n" + shared["basesoivname"],
    )

    if "area" not in shared.columns:
        shared["area"] = shared["stabbr"]

    xtot = base_targets.loc[base_targets["basesoivname"] == "XTOT"].copy()

    out_cols = [
        "stabbr",
        "area",
        "count",
        "scope",
        "agilo",
        "agihi",
        "fstatus",
        "target",
        "basesoivname",
        "soivname",
        "agistub",
        "agilabel",
    ]
    shared_out = shared[[c for c in out_cols if c in shared.columns]]
    xtot_out = xtot[[c for c in out_cols if c in xtot.columns]]

    stack = pd.concat([xtot_out, shared_out], ignore_index=True)

    is_xtot = (stack["basesoivname"] == "XTOT") & (stack["scope"] == 0)
    stack["_xtot"] = is_xtot.astype(int)
    stack = stack.sort_values(
        [
            "stabbr",
            "_xtot",
            "scope",
            "fstatus",
            "basesoivname",
            "count",
            "agistub",
        ],
        ascending=[True, False, True, True, True, True, True],
    ).reset_index(drop=True)
    stack["sort"] = stack.groupby("stabbr").cumcount() + 1
    stack = stack.drop(columns=["_xtot"])
    return stack


# ---- Convenience orchestrator ----------------------------


def prepare_area_targets(
    area_type: "AreaType",
    area_data_year: int,
    national_data_year: int = 0,
    pop_year: int = 0,
    cached_allvars_path: Path = None,
    soi_raw_data_dir: Path = None,
) -> pd.DataFrame:
    """
    End-to-end: SOI data -> base targets -> all-shares targets.

    Supports flexible year pairing: the SOI year (geographic
    shares), the TMD year (national levels), and the population
    year can each be set independently.

    Parameters
    ----------
    area_type : AreaType
        Currently only STATE is supported.
    area_data_year : int
        Year for SOI data (geographic distribution).
    national_data_year : int, optional
        Year for TMD national data. Defaults to area_data_year.
    pop_year : int, optional
        Year for population data. Defaults to area_data_year.
    cached_allvars_path : Path, optional
        Path to cached_allvars.csv.
    soi_raw_data_dir : Path, optional
        Directory containing raw SOI state CSV files.

    Returns
    -------
    pd.DataFrame
        Enhanced targets ready for target_file_writer.
    """
    if area_type != AreaType.STATE:
        raise ValueError(
            f"Only STATE is supported in this PR. Got: {area_type}"
        )

    if national_data_year == 0:
        national_data_year = area_data_year
    if pop_year == 0:
        pop_year = area_data_year

    repo_root = Path(__file__).parent.parent.parent.parent
    if cached_allvars_path is None:
        cached_allvars_path = (
            repo_root / "tmd" / "storage" / "output" / "cached_allvars.csv"
        )

    agi_cuts = STATE_AGI_CUTS
    if soi_raw_data_dir is None:
        soi_raw_data_dir = (
            repo_root / "tmd" / "areas" / "prepare" / "data" / "soi_states"
        )

    soilong = create_soilong(soi_raw_data_dir, years=[area_data_year])
    pop_df = get_state_population(pop_year)
    base_targets = create_state_base_targets(soilong, pop_df, area_data_year)

    return build_all_shares_targets(
        base_targets=base_targets,
        cached_allvars_path=cached_allvars_path,
        all_mappings=ALL_SHARING_MAPPINGS,
        agi_cuts=agi_cuts,
    )
