"""
State SOI data ingestion and base targets construction.

Pipeline:
  1. Read raw SOI state CSV files for one or more years.
  2. Pivot to long format (one row per state/year/agistub/variable).
  3. Classify variables (vtype, basesoivname).
  4. Create derived variables (18400 = 18425 + 18450).
  5. Multiply amount values by 1000 (SOI stores in thousands).
  6. Add AGI labels.
  7. For a chosen year, add count/scope/fstatus metadata.
  8. Append XTOT (population) records.
  9. Produce ``base_targets`` DataFrame.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from tmd.areas.prepare.constants import (
    ALLCOUNT_VARS,
    AreaType,
    SOI_STATE_CSV_PATTERNS,
    build_agi_labels,
)

# ---- Variable classification ------------------------------------------------


def classify_soi_variable(soivname: str):
    """
    Derive (basesoivname, vtype) from a lowercase SOI variable name.

    Rules:
      - 6-char names starting with 'a': amount, base = chars [1:6]
      - 6-char names starting with 'n': count,  base = chars [1:6]
      - 'numdep': count, base = 'numdep'
      - everything else (mars1, mars2, mars4, elf, prep, ...): count,
        base = full name

    Returns
    -------
    (basesoivname, vtype) : tuple[str, str]
    """
    if soivname == "numdep":
        return "numdep", "count"
    if len(soivname) == 6:
        first = soivname[0]
        if first == "a":
            return soivname[1:], "amount"
        if first == "n":
            return soivname[1:], "count"
    # Short names (mars1, elf, cprep, prep, schf, etc.) are counts
    return soivname, "count"


# ---- Read raw CSV files ---------------------------------------------


def read_soi_state_csvs(
    raw_data_dir: Path,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Read and stack raw SOI state CSV files.

    Parameters
    ----------
    raw_data_dir : Path
        Directory containing the raw CSV files.
    years : list of int, optional
        Years to read.  Defaults to all years in
        ``SOI_STATE_CSV_PATTERNS``.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns lowercased, STATE→stabbr,
        AGI_STUB→agistub, plus a ``year`` column.
    """
    if years is None:
        years = sorted(SOI_STATE_CSV_PATTERNS.keys())
    frames = []
    for yr in years:
        fname = SOI_STATE_CSV_PATTERNS.get(yr)
        if fname is None:
            raise FileNotFoundError(
                f"No SOI CSV pattern for year {yr}. "
                f"Available: {sorted(SOI_STATE_CSV_PATTERNS.keys())}"
            )
        fpath = raw_data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"SOI CSV not found: {fpath}")
        df = pd.read_csv(fpath, thousands=",")
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"state": "stabbr", "agi_stub": "agistub"})
        year_col = pd.Series(yr, index=df.index, name="year")
        df = pd.concat([df, year_col], axis=1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- Pivot to long format -------------------------------------------


def pivot_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot wide SOI data to long format.

    Returns DataFrame with columns:
      stabbr, year, agistub, soivname, value
    """
    id_cols = ["stabbr", "year", "agistub"]
    value_cols = [c for c in wide_df.columns if c not in id_cols]
    long = wide_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="soivname",
        value_name="value",
    )
    # Drop rows where value is NaN (matches R: filter(!is.na(value)))
    long = long.dropna(subset=["value"]).copy()
    return long


# ---- Derived variables ----------------------------------------------


def create_derived_variables(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived SOI variables and append to long DataFrame.

    Currently creates:
      - a18400 = a18425 + a18450 (SALT income/sales, estimated)
      - n18400 = n18425 + n18450
    """
    # Filter rows for 18425 and 18450 component variables
    mask = long_df["soivname"].str[1:].isin(["18425", "18450"])
    components = long_df.loc[mask].copy()
    if components.empty:
        return long_df
    # Replace the last 5 chars with 18400, keeping prefix (a or n)
    components["soivname"] = components["soivname"].str[0] + "18400"
    # Sum values for same (stabbr, agistub, year, soivname)
    derived = components.groupby(
        ["stabbr", "year", "agistub", "soivname"], as_index=False
    )["value"].sum()
    return pd.concat([long_df, derived], ignore_index=True)


# ---- Classify and annotate ------------------------------------------


def annotate_variables(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basesoivname and vtype columns by classifying soivname.
    """
    classifications = long_df["soivname"].apply(classify_soi_variable)
    long_df = long_df.copy()
    long_df["basesoivname"] = classifications.str[0]
    long_df["vtype"] = classifications.str[1]
    return long_df


def scale_amounts(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiply amount values by 1000 (SOI CSV stores amounts in thousands).
    """
    df = long_df.copy()
    mask = df["vtype"] == "amount"
    df.loc[mask, "value"] = df.loc[mask, "value"] * 1000
    return df


# ---- Full pipeline: raw CSVs to soilong DataFrame ------


def create_soilong(
    raw_data_dir: Path,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Full pipeline from raw SOI CSVs to annotated long DataFrame.

    Steps: read → pivot → derive → classify → scale amounts → add labels.

    Returns DataFrame with columns:
      stabbr, soivname, basesoivname, vtype, agistub, year, value
    """
    wide = read_soi_state_csvs(raw_data_dir, years)
    long = pivot_to_long(wide)
    long = create_derived_variables(long)
    long = annotate_variables(long)
    long = scale_amounts(long)
    # Add AGI labels
    agi_labels = build_agi_labels(AreaType.STATE)
    long = long.merge(
        agi_labels[["agistub", "agilo", "agihi", "agilabel"]],
        on="agistub",
        how="left",
    )
    long = long.sort_values(
        ["stabbr", "soivname", "basesoivname", "vtype", "agistub", "year"]
    ).reset_index(drop=True)
    return long


# ---- Base targets construction --------------------------------------


def create_state_base_targets(
    soilong: pd.DataFrame,
    pop_df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Create base_targets from SOI long data and population.

    Steps:
      1. Filter soilong to specified year, exclude OA.
      2. Add scope, count, fstatus metadata.
      3. Create XTOT population records.
      4. Stack and sort.

    Parameters
    ----------
    soilong : pd.DataFrame
        Output of ``create_soilong()``.
    pop_df : pd.DataFrame
        Population data with columns (stabbr, population).
    year : int
        Year to filter SOI data to.

    Returns
    -------
    pd.DataFrame
        Base targets with columns: stabbr, area, count, scope,
        agilo, agihi, fstatus, target, basesoivname, soivname,
        agistub, agilabel.
    """
    # 1. Filter to year, exclude OA (Other Areas)
    soi = soilong.loc[
        (soilong["year"] == year) & (soilong["stabbr"] != "OA")
    ].copy()

    # 2. Add scope (always 1 for SOI = PUF-derived)
    soi["scope"] = 1

    # 3. Assign count type
    #    amount → 0, allcount vars → 1, other counts → 2
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
    #    mars* variables → last char as int; others → 0
    mars_mask = soi["soivname"].str.startswith("mars")
    soi["fstatus"] = 0
    soi.loc[mars_mask, "fstatus"] = (
        soi.loc[mars_mask, "soivname"].str[-1].astype(int)
    )

    # 5. Rename value → target and select columns
    soi = soi.rename(columns={"value": "target"})
    soi = soi[
        [
            "stabbr",
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

    # 6. Create XTOT (population) records
    agi_labels = build_agi_labels(AreaType.STATE)
    agi0 = agi_labels.loc[agi_labels["agistub"] == 0].iloc[0]
    pop_recs = pop_df.copy()
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
    base_targets["area"] = base_targets["stabbr"]
    base_targets = base_targets.sort_values(
        ["stabbr", "scope", "fstatus", "basesoivname", "count", "agistub"]
    ).reset_index(drop=True)

    return base_targets
