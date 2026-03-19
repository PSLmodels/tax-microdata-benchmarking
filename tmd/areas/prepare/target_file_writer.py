"""
Target file writer — generates per-area target CSV files.

Reads:
  - A JSON recipe specifying which targets to include.
  - A variable mapping CSV (varname → basesoivname).
  - An enhanced_targets DataFrame (base + shared targets).

Writes:
  - One ``{area}_targets.csv`` per area, with columns:
    varname, count, scope, agilo, agihi, fstatus, target.
"""

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd

from tmd.areas.prepare.constants import (
    ALLCOUNT_VARS,
    STATE_NUM_AGI_STUBS,
)

# ---- JSON recipe parsing -------------------------------------------


def load_recipe(path: Path) -> dict:
    """
    Load a JSON recipe file, stripping ``//`` comments.

    Parameters
    ----------
    path : Path
        Path to the JSON file (may contain ``//`` line comments).

    Returns
    -------
    dict
        Parsed recipe.
    """
    text = path.read_text(encoding="utf-8")
    # Remove // comments (not inside strings — good enough for our files)
    cleaned = re.sub(r"//[^\n]*", "", text)
    return json.loads(cleaned)


# ---- Variable mapping -----------------------------------------------


def load_variable_mapping(path: Path) -> pd.DataFrame:
    """
    Read a variable mapping CSV.

    Expected columns: varname, basesoivname, description, fstatus.
    """
    return pd.read_csv(path, dtype={"fstatus": int})


# ---- Build match frame -----------------------------------------------


def _build_match_frame(
    recipe: dict,
    vmap: pd.DataFrame,
    top_agistub: int,
) -> pd.DataFrame:
    """
    Build a DataFrame of all (varname, count, scope, fstatus,
    agistub) combinations requested by the recipe.

    Steps:
      1. Cross recipe targets × agistubs 1..top_agistub.
      2. Drop agi_exclude stubs.
      3. Add XTOT row (agistub=0).
      4. Join with variable mapping to get basesoivname.
      5. Filter invalid count/basesoivname combos.
    """
    # Parse target specs from recipe
    target_rules = pd.DataFrame(recipe["targets"])
    # Build (varname, scope, count, fstatus) × agistub combos
    stubs = pd.DataFrame({"agistub": range(1, top_agistub + 1)})
    target_stubs = (
        target_rules[["varname", "scope", "count", "fstatus"]]
        .drop_duplicates()
        .merge(stubs, how="cross")
    )

    # Drop excluded AGI stubs
    if "agi_exclude" in target_rules.columns:
        drops = target_rules.dropna(subset=["agi_exclude"])
        if not drops.empty:
            # agi_exclude may be a list per row
            drop_rows = []
            for _, row in drops.iterrows():
                excl = row["agi_exclude"]
                if isinstance(excl, (list, tuple)):
                    for stub in excl:
                        drop_rows.append(
                            {
                                "varname": row["varname"],
                                "scope": row["scope"],
                                "count": row["count"],
                                "fstatus": row["fstatus"],
                                "agistub": int(stub),
                            }
                        )
                elif excl is not None:
                    drop_rows.append(
                        {
                            "varname": row["varname"],
                            "scope": row["scope"],
                            "count": row["count"],
                            "fstatus": row["fstatus"],
                            "agistub": int(excl),
                        }
                    )
            if drop_rows:
                drop_df = pd.DataFrame(drop_rows)
                keys = [
                    "varname",
                    "scope",
                    "count",
                    "fstatus",
                    "agistub",
                ]
                target_stubs = target_stubs.merge(
                    drop_df, on=keys, how="left", indicator=True
                )
                target_stubs = target_stubs.loc[
                    target_stubs["_merge"] == "left_only"
                ].drop(columns=["_merge"])

    # Add sort numbers and XTOT row
    target_stubs["sort"] = range(2, len(target_stubs) + 2)
    xtot = pd.DataFrame(
        [
            {
                "varname": "XTOT",
                "scope": 0,
                "count": 0,
                "fstatus": 0,
                "agistub": 0,
                "sort": 1,
            }
        ]
    )
    target_stubs = pd.concat([xtot, target_stubs], ignore_index=True)

    # Cross variable mapping with count values 0-4
    counts_df = pd.DataFrame({"count": range(5)})
    vmap2 = vmap.merge(counts_df, how="cross")
    # Filter invalid combos (same logic as R)
    # XTOT only with count=0, fstatus=0
    vmap2 = vmap2.loc[
        ~(
            (vmap2["basesoivname"] == "XTOT")
            & ((vmap2["count"] != 0) | (vmap2["fstatus"] != 0))
        )
    ]
    # count=1 only for allcount_vars
    # Handle both direct SOI names ("n1", "mars1") and shared names
    # ("tmd00100_shared_by_soin1") by extracting the SOI part
    allcount = [v for v in ALLCOUNT_VARS if v != "n2"]

    def _is_allcount(name):
        if name in allcount:
            return True
        if "_shared_by_soi" in str(name):
            soi_part = str(name).rsplit("_shared_by_soi", maxsplit=1)[-1]
            return soi_part in allcount
        return False

    is_ac = vmap2["basesoivname"].apply(_is_allcount)
    vmap2 = vmap2.loc[~((vmap2["count"] == 1) & ~is_ac)]
    # allcount_vars can only have count=1
    vmap2 = vmap2.loc[~(is_ac & (vmap2["count"] != 1))]

    # Join target stubs with mapping to get basesoivname
    match_frame = target_stubs.merge(
        vmap2[["varname", "basesoivname", "fstatus", "count"]],
        on=["varname", "fstatus", "count"],
        how="inner",
    )
    match_frame = match_frame.sort_values("sort").reset_index(drop=True)
    return match_frame


# ---- Write target files ----------------------------------------------


def write_area_target_files(
    recipe_path: Path,
    enhanced_targets: pd.DataFrame,
    variable_mapping_path: Path,
    output_dir: Path,
    extra_targets: Dict[str, "pd.DataFrame"] = None,
) -> Dict[str, int]:
    """
    Write per-area target CSV files from a recipe.

    Parameters
    ----------
    recipe_path : Path
        Path to JSON recipe file.
    enhanced_targets : pd.DataFrame
        Combined base + additional targets with columns including:
        stabbr, area, basesoivname, scope, count, fstatus, agistub,
        agilo, agihi, target, sort.
    variable_mapping_path : Path
        Path to variable mapping CSV.
    output_dir : Path
        Directory to write target files to.
    extra_targets : dict, optional
        Mapping of area code → DataFrame of additional target rows
        (e.g., from ``build_extended_targets()``).  These are appended
        after the recipe-based targets in a single write pass.

    Returns
    -------
    dict
        Mapping of area code → number of targets written.
    """
    recipe = load_recipe(recipe_path)
    vmap = load_variable_mapping(variable_mapping_path)

    # Determine area type settings
    areatype = recipe["areatype"]
    if areatype == "state":
        top_agistub = STATE_NUM_AGI_STUBS
    else:
        raise ValueError(f"Unknown areatype: {areatype}")

    suffix = recipe.get("suffix", "")
    notzero = recipe.get("notzero", False)
    notnegative = recipe.get("notnegative", False)
    arealist = recipe.get("arealist", "all")

    # Build the match frame
    match_frame = _build_match_frame(recipe, vmap, top_agistub)

    # Apply filters to enhanced_targets
    stack = enhanced_targets.copy()

    # Area filter
    if arealist != "all":
        if isinstance(arealist, str):
            arealist = [arealist]
        stack = stack.loc[stack["area"].isin(arealist)]

    # Zero filter
    if notzero:
        stack = stack.loc[stack["target"] != 0]

    # Negative filter
    if notnegative:
        stack = stack.loc[stack["target"] >= 0]

    # Join: match_frame inner join filtered stack
    mapped = match_frame.merge(
        stack[
            [
                "area",
                "basesoivname",
                "scope",
                "count",
                "fstatus",
                "agistub",
                "agilo",
                "agihi",
                "target",
            ]
        ],
        on=["basesoivname", "scope", "count", "fstatus", "agistub"],
        how="inner",
    )
    mapped = mapped.sort_values(["area", "sort"]).reset_index(drop=True)

    # Write per-area files
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    out_cols = [
        "varname",
        "count",
        "scope",
        "agilo",
        "agihi",
        "fstatus",
        "target",
    ]
    for area_code, group in mapped.groupby("area"):
        area_upper = str(area_code).upper()
        area_lower = str(area_code).lower()
        fname = f"{area_lower}{suffix}_targets.csv"
        fpath = output_dir / fname
        base = group[out_cols]
        # Append extended targets if provided
        if extra_targets and area_upper in extra_targets:
            extra = extra_targets[area_upper][out_cols]
            base = pd.concat([base, extra], ignore_index=True)
        base.to_csv(fpath, index=False)
        result[str(area_code)] = len(base)

    return result
