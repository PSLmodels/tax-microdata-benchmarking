"""
Congressional District SOI data ingestion, crosswalk, and base targets.

Pipeline:
  1. Read raw SOI CD CSV files for one or more years.
  2. Build area codes (e.g., "AL01") from state + district number.
     At-large states (CONG_DISTRICT=0) are recoded to district 1.
  3. Pivot to long format (one row per area/agistub/variable).
  4. Classify variables, create derived variables, scale amounts.
  5. Apply 117th→{118th,119th} Congress crosswalk using population-weighted
     allocation factors. Choice of target Congress is an explicit argument.
  6. Compute XTOT population targets from N2 (exemptions/dependents).
  7. Produce ``base_targets`` DataFrame.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from tmd.areas.prepare.constants import (
    ALLCOUNT_VARS,
    AT_LARGE_STATES,
    AreaType,
    SOI_CD_CSV_PATTERNS,
    build_agi_labels,
)
from tmd.areas.prepare.soi_state_data import (
    annotate_variables,
    scale_amounts,
)

# Default paths
_DATA_DIR = Path(__file__).parent / "data"
_SOI_CD_DIR = _DATA_DIR / "soi_cds"

# Crosswalk files by target Congressional session.  Each file is the
# population-weighted geocorr 2022 crosswalk from 117th Congress CDs
# (the vintage of all SOI CD data) to the target session boundaries.
SUPPORTED_CONGRESSES = (118, 119)
_CROSSWALK_PATHS = {
    118: _DATA_DIR / "geocorr2022_cd117_to_cd118.csv",
    119: _DATA_DIR / "geocorr2022_cd117_to_cd119.csv",
}


def _validate_congress(congress: int) -> int:
    if congress not in _CROSSWALK_PATHS:
        raise ValueError(
            f"Unsupported Congress session: {congress}. "
            f"Supported: {SUPPORTED_CONGRESSES}"
        )
    return congress


# ---- Read raw CD CSV files ----------------------------------------


def read_soi_cd_csv(
    raw_data_dir: Path,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Read and stack raw SOI CD CSV files.

    Returns wide-format DataFrame with columns lowercased, plus
    ``stabbr``, ``district_num``, ``area``, and ``year`` columns.
    The ``area`` column has the 118th-Congress-style code (e.g.,
    "AL01") even though at this stage the district numbers still
    reflect the 117th Congress boundaries.

    At-large states (CONG_DISTRICT=0) are recoded to district 1.
    State-total rows (non-at-large states with CONG_DISTRICT=0)
    and the US-total row are excluded.
    """
    if years is None:
        years = sorted(SOI_CD_CSV_PATTERNS.keys())
    frames = []
    for yr in years:
        fname = SOI_CD_CSV_PATTERNS.get(yr)
        if fname is None:
            raise FileNotFoundError(
                f"No SOI CD CSV pattern for year {yr}. "
                f"Available: {sorted(SOI_CD_CSV_PATTERNS.keys())}"
            )
        fpath = raw_data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"SOI CD CSV not found: {fpath}")
        df = pd.read_csv(fpath, thousands=",")
        df.columns = [c.lower() for c in df.columns]

        # SOI CD data bug (confirmed in 2022 file): column a59664
        # (EITC amount for filers with 3+ qualifying children) is
        # published in dollars, while all other amount columns
        # (a59660, a59661, a59662, a59663) are in $1,000s.  Evidence:
        # at stub 0, a59661+a59662+a59663+a59664 overshoots a59660
        # by ~234x, but a59661+a59662+a59663+(a59664/1000) matches
        # a59660 within 0.04% (normal SOI rounding).  The state SOI
        # file does NOT have this error.  Reported to SOI 2026-03-24.
        # See session_notes/soi_a59664_unit_error_email.md.
        if "a59664" in df.columns:
            df["a59664"] = df["a59664"] / 1000
        df = df.rename(
            columns={
                "state": "stabbr",
                "agi_stub": "agistub",
                "cong_district": "cd117_district",
            }
        )
        year_col = pd.Series(yr, index=df.index, name="year")
        df = pd.concat([df, year_col], axis=1)

        # Drop US-total row
        df = df[df["stabbr"] != "US"].copy()

        # Recode at-large states: district 0 → 1
        at_large_mask = df["stabbr"].isin(AT_LARGE_STATES) & (
            df["cd117_district"] == 0
        )
        df.loc[at_large_mask, "cd117_district"] = 1

        # Drop state-total rows (non-at-large states with district 0)
        df = df[df["cd117_district"] > 0].copy()

        # Build area code: "AL01", "CA52", etc.
        df["district_num"] = df["cd117_district"]
        df["area"] = df["stabbr"] + df["district_num"].apply(
            lambda d: f"{d:02d}"
        )

        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- Pivot CD data to long format ---------------------------------


def _cd_pivot_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot wide CD SOI data to long format.

    Like the state version but uses area (CD code) instead of stabbr
    as the geographic identifier.
    """
    id_cols = ["stabbr", "area", "year", "agistub"]
    # Drop ID columns that shouldn't be melted
    drop_cols = ["statefips", "cd117_district", "district_num"]
    value_cols = [
        c for c in wide_df.columns if c not in id_cols and c not in drop_cols
    ]
    long = wide_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="soivname",
        value_name="value",
    )
    long = long.dropna(subset=["value"]).copy()
    return long


# ---- Derived variables (CD-aware) ---------------------------------


def _create_cd_derived_variables(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived SOI variables for CD data.

    Same as state version (a18400 = a18425 + a18450) but groups
    by ``area`` as well as ``stabbr``.
    """
    mask = long_df["soivname"].str[1:].isin(["18425", "18450"])
    components = long_df.loc[mask].copy()
    if components.empty:
        return long_df
    components["soivname"] = components["soivname"].str[0] + "18400"
    derived = components.groupby(
        ["stabbr", "area", "year", "agistub", "soivname"],
        as_index=False,
    )["value"].sum()
    return pd.concat([long_df, derived], ignore_index=True)


# ---- Full pipeline: raw CSV to annotated long DataFrame -----------


def create_cd_soilong(
    raw_data_dir: Path = None,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Full pipeline from raw SOI CD CSVs to annotated long DataFrame.

    Steps: read → pivot → derive → classify → scale → add labels.

    Returns DataFrame with columns:
      stabbr, area, soivname, basesoivname, vtype, agistub, year,
      value, agilo, agihi, agilabel
    """
    if raw_data_dir is None:
        raw_data_dir = _SOI_CD_DIR
    wide = read_soi_cd_csv(raw_data_dir, years)
    long = _cd_pivot_to_long(wide)
    long = _create_cd_derived_variables(long)
    long = annotate_variables(long)
    long = scale_amounts(long)
    # Add AGI labels
    agi_labels = build_agi_labels(AreaType.CD)
    long = long.merge(
        agi_labels[["agistub", "agilo", "agihi", "agilabel"]],
        on="agistub",
        how="left",
    )
    long = long.sort_values(
        ["area", "soivname", "basesoivname", "vtype", "agistub", "year"]
    ).reset_index(drop=True)
    return long


# ---- Crosswalk: 117th → 118th Congress ---------------------------


def load_crosswalk(
    congress: int = 118,
    crosswalk_path: Path = None,
) -> pd.DataFrame:
    """
    Load the geocorr 117th→{target}th Congress crosswalk.

    Parameters
    ----------
    congress : int
        Target Congress session (118 or 119).  Selects the default
        crosswalk file when ``crosswalk_path`` is not given.
    crosswalk_path : Path, optional
        Explicit path to a crosswalk CSV.  Overrides the default.

    Returns DataFrame with columns:
      stabbr, cd117, cd_target, afact2 (allocation factor cd117→cd_target)

    The target-CD column (``cd118`` or ``cd119``) is renamed to a
    neutral ``cd_target`` for uniform downstream handling.

    The crosswalk has a label/header row at line 2 that is skipped.
    District codes are zero-padded 2-digit strings.
    """
    _validate_congress(congress)
    if crosswalk_path is None:
        crosswalk_path = _CROSSWALK_PATHS[congress]
    df = pd.read_csv(crosswalk_path, dtype=str)
    # Row 0 after header is a label row — drop it
    df = df.iloc[1:].copy()
    df["afact2"] = df["afact2"].astype(float)
    # Detect and rename the target-CD column (cd118 or cd119) to
    # the neutral name ``cd_target``.
    target_col_candidates = [c for c in df.columns if c in ("cd118", "cd119")]
    if not target_col_candidates:
        raise ValueError(
            f"Crosswalk {crosswalk_path} has no cd118 or cd119 column; "
            f"columns are {list(df.columns)}"
        )
    target_col = target_col_candidates[0]
    # Ensure district codes are zero-padded strings
    df["cd117"] = df["cd117"].str.zfill(2)
    df[target_col] = df[target_col].str.zfill(2)
    df = df.rename(columns={"stab": "stabbr", target_col: "cd_target"})
    # Strip whitespace from stabbr
    df["stabbr"] = df["stabbr"].str.strip()

    # At-large states use "00" for cd117 in the crosswalk.
    # Recode cd117 to "01" to match the SOI data recode.
    # Do NOT recode cd_target here unless the state is still at-large
    # — MT, for example, goes from 1 district (117th) to 2 (118th/119th).
    at_large_mask = df["stabbr"].isin(AT_LARGE_STATES)
    df.loc[at_large_mask, "cd117"] = "01"
    # For states that remain at-large in the target Congress,
    # cd_target is "00" (or "98" for DC non-voting delegate) → recode
    # to "01" to match SOI-style area codes.
    still_at_large = at_large_mask & df["cd_target"].isin(["00", "98"])
    df.loc[still_at_large, "cd_target"] = "01"

    return df[["stabbr", "cd117", "cd_target", "afact2"]].copy()


def apply_crosswalk(
    cd117_long: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert 117th Congress CD targets to the target Congress boundaries.

    For each (cd_target, variable, agistub) combination:
      value_target = sum over cd117 contributors:
          value_cd117 * afact2

    Parameters
    ----------
    cd117_long : pd.DataFrame
        Long-format CD data with 117th Congress area codes.
    crosswalk : pd.DataFrame
        Crosswalk with stabbr, cd117, cd_target, afact2 (as returned
        by ``load_crosswalk``).  Works for any target Congress.

    Returns
    -------
    pd.DataFrame
        Long-format data with target-Congress area codes.
    """
    # Extract state and district from area code
    df = cd117_long.copy()
    df["cd117"] = df["area"].str[2:].str.zfill(2)

    # Merge with crosswalk to get target-CD mapping and allocation factors
    merged = df.merge(
        crosswalk,
        on=["stabbr", "cd117"],
        how="inner",
    )

    # Apply allocation factor
    merged["value"] = merged["value"] * merged["afact2"]

    # Build new area code with target-Congress district
    merged["area"] = merged["stabbr"] + merged["cd_target"]

    # Aggregate: sum values for same (area, soivname, agistub, year, ...)
    group_cols = [
        "stabbr",
        "area",
        "year",
        "agistub",
        "soivname",
        "basesoivname",
        "vtype",
        "agilo",
        "agihi",
        "agilabel",
    ]
    result = (
        merged.groupby(group_cols, as_index=False)["value"]
        .sum()
        .sort_values(["area", "soivname", "basesoivname", "vtype", "agistub"])
        .reset_index(drop=True)
    )
    return result


# ---- XTOT population from N2 -------------------------------------


def compute_cd_population(congress: int = 118) -> pd.DataFrame:
    """
    Compute target-Congress CD population from the geocorr crosswalk.

    Uses the geocorr crosswalk ``pop20`` column (2020 Census population)
    aggregated to the target-Congress districts.  This matches the state
    pipeline's use of Census population for XTOT targets, ensuring
    pop_share denominators are consistent.

    Parameters
    ----------
    congress : int
        Target Congress session (118 or 119).

    Returns
    -------
    pd.DataFrame
        Columns (stabbr, area, population) for each CD.
    """
    _validate_congress(congress)
    crosswalk_path = _CROSSWALK_PATHS[congress]
    xw = pd.read_csv(crosswalk_path, dtype=str)
    xw = xw.iloc[1:]  # skip label row
    xw["pop20"] = xw["pop20"].astype(int)
    xw["stabbr"] = xw["stab"].str.strip()

    # Exclude territories (PR, etc.) — not in PUF/TMD
    xw = xw[~xw["stabbr"].isin(["PR", "GU", "VI", "AS", "MP"])].copy()

    # Detect target-CD column (cd118 or cd119) and normalize
    target_col_candidates = [c for c in xw.columns if c in ("cd118", "cd119")]
    if not target_col_candidates:
        raise ValueError(
            f"Crosswalk {crosswalk_path} has no cd118 or cd119 column"
        )
    target_col = target_col_candidates[0]
    xw[target_col] = xw[target_col].str.zfill(2)

    # Recode at-large districts to 01
    # Most at-large states use "00"; DC uses "98" (non-voting delegate)
    at_large_mask = xw["stabbr"].isin(AT_LARGE_STATES)
    recode_mask = at_large_mask & xw[target_col].isin(["00", "98"])
    xw.loc[recode_mask, target_col] = "01"

    # Aggregate population by target-Congress district
    xw["area"] = xw["stabbr"] + xw[target_col]
    cd_pop = (
        xw.groupby(["stabbr", "area"])["pop20"]
        .sum()
        .reset_index()
        .rename(columns={"pop20": "population"})
    )
    return cd_pop.sort_values("area").reset_index(drop=True)


# ---- Base targets construction ------------------------------------


def create_cd_base_targets(
    cd_long: pd.DataFrame,
    cd_pop: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Create base_targets from CD long data and CD population.

    Mirrors ``create_state_base_targets`` but for CDs.

    Parameters
    ----------
    cd_long : pd.DataFrame
        Output of ``apply_crosswalk()`` (118th Congress long data).
    cd_pop : pd.DataFrame
        CD population from ``compute_cd_population()`` with columns
        (stabbr, area, population).
    year : int
        Year to filter SOI data to.

    Returns
    -------
    pd.DataFrame
        Base targets with columns: stabbr, area, count, scope,
        agilo, agihi, fstatus, target, basesoivname, soivname,
        agistub, agilabel.
    """
    # 1. Filter to year
    soi = cd_long.loc[cd_long["year"] == year].copy()

    # 2. Add scope (always 1 for SOI = PUF-derived)
    soi["scope"] = 1

    # 3. Assign count type
    soi["count"] = np.where(
        soi["vtype"] == "amount",
        0,
        np.where(
            soi["soivname"].isin(ALLCOUNT_VARS),
            1,
            2,
        ),
    )

    # 4. Assign filing status
    mars_mask = soi["soivname"].str.startswith("mars")
    soi["fstatus"] = 0
    soi.loc[mars_mask, "fstatus"] = (
        soi.loc[mars_mask, "soivname"].str[-1].astype(int)
    )

    # 5. Rename value → target
    soi = soi.rename(columns={"value": "target"})
    soi = soi[
        [
            "stabbr",
            "area",
            "soivname",
            "basesoivname",
            "count",
            "scope",
            "agilo",
            "agihi",
            "fstatus",
            "target",
            "agistub",
            "agilabel",
        ]
    ].copy()

    # 6. Create XTOT (population) records from N2
    agi_labels = build_agi_labels(AreaType.CD)
    agi0 = agi_labels.loc[agi_labels["agistub"] == 0].iloc[0]
    pop_recs = cd_pop.copy()
    pop_recs["soivname"] = "XTOT"
    pop_recs["basesoivname"] = "XTOT"
    pop_recs["agistub"] = 0
    pop_recs["count"] = 0
    pop_recs["scope"] = 0
    pop_recs["fstatus"] = 0
    pop_recs["target"] = pop_recs["population"]
    pop_recs["agilo"] = agi0["agilo"]
    pop_recs["agihi"] = agi0["agihi"]
    pop_recs["agilabel"] = agi0["agilabel"]
    pop_recs = pop_recs[soi.columns].copy()

    # 7. Combine and sort
    base_targets = pd.concat([pop_recs, soi], ignore_index=True)
    base_targets = base_targets.sort_values(
        [
            "area",
            "scope",
            "fstatus",
            "basesoivname",
            "count",
            "agistub",
        ]
    ).reset_index(drop=True)

    return base_targets
