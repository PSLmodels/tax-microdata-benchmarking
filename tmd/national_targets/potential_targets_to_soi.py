"""
Convert irs_aggregate_values.csv → soi.csv format.

This replaces the old pipeline:
  agi_targets.csv → soi_targets.py → soi.csv

New pipeline:
  irs_aggregate_values.csv → this script → soi.csv

Key differences from old pipeline:
- potential_targets has ptarget in dollars (not thousands)
- potential_targets has var_type/value_filter columns instead of embedded names
- potential_targets has marstat column instead of filing status in vname
- potential_targets does NOT have "taxable" subgroup data, so Taxable only
  rows are not produced (reweight.py skips these anyway)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


# ── AGI bound mapping ──────────────────────────────────────────────

AGI_BOUND_MAP = {
    "All returns": (-np.inf, np.inf),
    "All returns, total": (-np.inf, np.inf),
    "No adjusted gross income": (-np.inf, 0),
    "No adjusted gross income (includes deficits)": (-np.inf, 0),
    "$1 under $5,000": (1, 5_000),
    "Under $5,000": (0, 5_000),
    "$5,000 under $10,000": (5_000, 10_000),
    "$10,000 under $15,000": (10_000, 15_000),
    "$15,000 under $20,000": (15_000, 20_000),
    "$20,000 under $25,000": (20_000, 25_000),
    "$25,000 under $30,000": (25_000, 30_000),
    "$30,000 under $35,000": (30_000, 35_000),
    "$30,000 under $40,000": (30_000, 40_000),
    "$35,000 under $40,000": (35_000, 40_000),
    "$40,000 under $45,000": (40_000, 45_000),
    "$40,000 under $50,000": (40_000, 50_000),
    "$45,000 under $50,000": (45_000, 50_000),
    "$50,000 under $55,000": (50_000, 55_000),
    "$50,000 under $75,000": (50_000, 75_000),
    "$55,000 under $60,000": (55_000, 60_000),
    "$60,000 under $75,000": (60_000, 75_000),
    "$75,000 under $100,000": (75_000, 100_000),
    "$100,000 under $200,000": (100_000, 200_000),
    "$200,000 under $500,000": (200_000, 500_000),
    "$500,000 under $1,000,000": (500_000, 1_000_000),
    "$1,000,000 under $1,500,000": (1_000_000, 1_500_000),
    "$1,000,000 or more": (1_000_000, np.inf),
    "$1,500,000 under $2,000,000": (1_500_000, 2_000_000),
    "$2,000,000 under $5,000,000": (2_000_000, 5_000_000),
    "$5,000,000 under $10,000,000": (5_000_000, 10_000_000),
    "$10,000,000 or more": (10_000_000, np.inf),
}

MARSTAT_TO_FILING_STATUS = {
    "all": "All",
    "single": "Single",
    "mfjss": "Married Filing Jointly/Surviving Spouse",
    "mfs": "Married Filing Separately",
    "hoh": "Head of Household",
}

TABLE_NAME_MAP = {
    "tab11": "Table 1.1",
    "tab12": "Table 1.2",
    "tab14": "Table 1.4",
    "tab21": "Table 2.1",
}

SOI_COLUMNS = [
    "Year",
    "SOI table",
    "XLSX column",
    "XLSX row",
    "Variable",
    "Filing status",
    "AGI lower bound",
    "AGI upper bound",
    "Count",
    "Taxable only",
    "Full population",
    "Value",
]


def load_mapping():
    """Load irs_to_puf_map.json."""
    with open(DATA_DIR / "irs_to_puf_map.json", encoding="utf-8") as f:
        raw = json.load(f)
    # Remove the _comment entry
    return {k: v for k, v in raw.items() if k != "_comment"}


def get_tmd_variable_name(mapping, var_name, var_type, value_filter):
    """Map IRS var_name + var_type + value_filter → TMD variable name.

    Returns None if the variable should be skipped.
    """
    entry = mapping.get(var_name)
    if entry is None:
        return None

    # Special case: agi count rows → "count" variable
    if var_name == "agi" and var_type == "count":
        return entry.get("tmd_name_count", "count")

    # For count/number var_type on non-agi variables, use the same
    # TMD name as the amount rows (the Count column distinguishes them)
    combo = entry.get("combination_rule")

    if combo in ("separate_targeting", "separate_codes"):
        if value_filter in ("gt0",):
            return entry.get("tmd_name_gt0")
        if value_filter in ("lt0",):
            return entry.get("tmd_name_lt0")
        # 'all' or 'nz' filter not applicable for gain/loss vars
        return None
    tmd_name = entry.get("tmd_name")
    return tmd_name  # may be None (e.g., "id" variable)


def convert_potential_targets_to_soi(years=None, year_exclude_vars=None):
    """Convert irs_aggregate_values.csv → soi.csv DataFrame.

    Args:
        years: List of years to include, or None for all available.
        year_exclude_vars: Dict mapping year to list of IRS
            var_names to exclude for that year
            (e.g., {2022: ["rentroyalty", "estateincome"]}).

    Returns:
        DataFrame in soi.csv format.
    """
    pt = pd.read_csv(DATA_DIR / "irs_aggregate_values.csv")
    mapping = load_mapping()

    if years:
        pt = pt[pt.year.isin(years)]

    # Apply year-specific variable exclusions
    if year_exclude_vars:
        for excl_year, exclude_list in year_exclude_vars.items():
            pt = pt[
                ~((pt.year == excl_year) & (pt.var_name.isin(exclude_list)))
            ]

    rows = []
    skipped_vars = set()

    for _, row in pt.iterrows():
        var_name = row["var_name"]
        var_type = row["var_type"]
        value_filter = row["value_filter"]

        # Get TMD variable name
        tmd_name = get_tmd_variable_name(
            mapping, var_name, var_type, value_filter
        )
        if tmd_name is None:
            skipped_vars.add(f"{var_name}/{var_type}/{value_filter}")
            continue

        # Count flag: var_type 'count' or 'number' → Count=True in soi.csv
        # Exception: exemptions_n (var_type='number') maps to
        # count_of_exemptions which has Count=False in existing soi.csv
        if tmd_name == "count_of_exemptions":
            is_count = False
        else:
            is_count = var_type in ("count", "number")

        # AGI bounds
        lower, upper = AGI_BOUND_MAP[row["incrange"]]

        # Filing status
        filing_status = MARSTAT_TO_FILING_STATUS[row["marstat"]]

        # SOI table
        soi_table = TABLE_NAME_MAP[row["table"]]

        # Taxable only: potential_targets doesn't have taxable subgroup
        is_taxable = False

        # Full population flag
        is_full_pop = (
            filing_status == "All"
            and lower == -np.inf
            and upper == np.inf
            and not is_taxable
        )

        # Value: ptarget is already in dollars in potential_targets
        value = row["ptarget"]

        rows.append(
            {
                "Year": int(row["year"]),
                "SOI table": soi_table,
                "XLSX column": row["xlcolumn"],
                "XLSX row": int(row["xlrownum"]),
                "Variable": tmd_name,
                "Filing status": filing_status,
                "AGI lower bound": lower,
                "AGI upper bound": upper,
                "Count": is_count,
                "Taxable only": is_taxable,
                "Full population": is_full_pop,
                "Value": value,
            }
        )

    df = pd.DataFrame(rows)

    if skipped_vars:
        print(f"Skipped {len(skipped_vars)} var combos (no TMD mapping):")
        for sv in sorted(skipped_vars):
            print(f"  {sv}")

    # De-duplicate: irs_aggregate_values.csv intentionally preserves
    # all cross-table data including redundancy and minor integer differences
    # between tables (that file is the full audit trail).  soi.csv must have
    # exactly one row per (Year, Variable, Filing status, AGI bounds, Count)
    # key for the optimizer.  Resolution rule: lowest table number wins
    # (tab11 > tab12 > tab14 > tab21), which is the most comprehensive source.
    # Value is excluded from the dedup key so that minor cross-table integer
    # differences (e.g. tab11 reads 10134704, tab12 reads 10134703 for the
    # same cell) don't bypass deduplication.
    dedup_cols = [
        "Year",
        "Variable",
        "Filing status",
        "AGI lower bound",
        "AGI upper bound",
        "Count",
        "Taxable only",
    ]
    before = len(df)
    df = (
        df.sort_values(["Year", "SOI table", "XLSX row"])
        .groupby(dedup_cols, sort=False)
        .first()
        .reset_index()
    )
    after = len(df)
    if before != after:
        print(
            f"De-duplicated: {before} -> {after} rows "
            f"({before - after} removed)"
        )

    # Partner + S-corp aggregation:
    # For years with separate partnerincome/scorpincome (2021, 2022),
    # add S-corp values into partnership_and_s_corp totals
    _aggregate_partner_scorp(df)

    # Deterministic sort for reproducible diffs across pipeline runs
    df = df.sort_values(
        [
            "Year",
            "Variable",
            "Filing status",
            "AGI lower bound",
            "AGI upper bound",
            "Count",
            "Taxable only",
        ]
    ).reset_index(drop=True)

    return df[SOI_COLUMNS]


def _aggregate_partner_scorp(df):
    """Add S-corp values into partnership_and_s_corp for each year.

    Modifies df in place. Matches old soi_targets.py behavior:
      partnership_and_s_corp_income += s_corporation_net_income
      partnership_and_s_corp_losses += s_corporation_net_losses
    """
    for year in df.Year.unique():
        for partner_var, scorp_var in [
            ("partnership_and_s_corp_income", "s_corporation_net_income"),
            ("partnership_and_s_corp_losses", "s_corporation_net_losses"),
        ]:
            scorp_mask = (df.Year == year) & (df.Variable == scorp_var)
            partner_mask = (df.Year == year) & (df.Variable == partner_var)

            if not scorp_mask.any() or not partner_mask.any():
                continue

            scorp_rows = df.loc[scorp_mask].copy()
            partner_rows = df.loc[partner_mask].copy()

            merge_keys = [
                "AGI lower bound",
                "AGI upper bound",
                "Filing status",
                "Count",
                "Taxable only",
            ]

            merged = partner_rows.merge(
                scorp_rows[merge_keys + ["Value"]],
                on=merge_keys,
                suffixes=("", "_scorp"),
            )

            for _, mrow in merged.iterrows():
                mask = (
                    partner_mask
                    & (df["AGI lower bound"] == mrow["AGI lower bound"])
                    & (df["AGI upper bound"] == mrow["AGI upper bound"])
                    & (df["Filing status"] == mrow["Filing status"])
                    & (df["Count"] == mrow["Count"])
                    & (df["Taxable only"] == mrow["Taxable only"])
                )
                df.loc[mask, "Value"] += mrow["Value_scorp"]


def compare_with_existing_soi(new_df, existing_soi_path, year=2021):
    """Compare converter output against existing soi.csv for targeted rows.

    Checks only the rows that reweight.py actually targets.
    """
    existing = pd.read_csv(existing_soi_path)
    existing = existing[existing.Year == year]
    new = new_df[new_df.Year == year]

    # Reweight.py targeting logic
    agi_level_vars = [
        "adjusted_gross_income",
        "count",
        "employment_income",
        "business_net_profits",
        "capital_gains_gross",
        "ordinary_dividends",
        "partnership_and_s_corp_income",
        "qualified_dividends",
        "taxable_interest_income",
        "total_pension_income",
        "total_social_security",
    ]
    agg_level_vars = [
        "business_net_losses",
        "capital_gains_distributions",
        "capital_gains_losses",
        "estate_income",
        "estate_losses",
        "exempt_interest",
        "ira_distributions",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_income",
        "rent_and_royalty_net_losses",
        "taxable_pension_income",
        "taxable_social_security",
        "unemployment_compensation",
    ]

    # Filter existing to targeted rows (same logic as reweight.py)
    targeted = existing[
        (
            existing.Variable.isin(agi_level_vars)
            & (
                (existing["AGI lower bound"] != -np.inf)
                | (existing["AGI upper bound"] != np.inf)
            )
        )
        | (
            existing.Variable.isin(agg_level_vars)
            & (existing["AGI lower bound"] == -np.inf)
            & (existing["AGI upper bound"] == np.inf)
        )
    ]
    targeted = targeted[~targeted["Taxable only"]]

    match_count = 0
    mismatch_count = 0
    missing_count = 0

    for _, erow in targeted.iterrows():
        match = new[
            (new.Variable == erow["Variable"])
            & (new["Filing status"] == erow["Filing status"])
            & (new["AGI lower bound"] == erow["AGI lower bound"])
            & (new["AGI upper bound"] == erow["AGI upper bound"])
            & (new["Count"] == erow["Count"])
            & (new["Taxable only"] == erow["Taxable only"])
        ]

        if len(match) == 0:
            missing_count += 1
            var = erow["Variable"]
            agi_lo = erow["AGI lower bound"]
            agi_hi = erow["AGI upper bound"]
            fs = erow["Filing status"]
            cnt = erow["Count"]
            print(
                f"  MISSING: {var} "
                f"AGI=[{agi_lo},{agi_hi}) "
                f"FS={fs} Count={cnt}"
            )
        elif abs(match.iloc[0]["Value"] - erow["Value"]) > 0.5:
            mismatch_count += 1
            var = erow["Variable"]
            agi_lo = erow["AGI lower bound"]
            agi_hi = erow["AGI upper bound"]
            new_val = match.iloc[0]["Value"]
            old_val = erow["Value"]
            print(
                f"  MISMATCH: {var} "
                f"AGI=[{agi_lo},{agi_hi}) "
                f"existing={old_val} new={new_val} "
                f"diff={new_val - old_val}"
            )
        else:
            match_count += 1

    total = match_count + mismatch_count + missing_count
    print(f"\nTargeted row comparison for {year}:")
    print(f"  Matched: {match_count}/{total}")
    print(f"  Mismatched: {mismatch_count}/{total}")
    print(f"  Missing: {missing_count}/{total}")

    return mismatch_count == 0 and missing_count == 0


if __name__ == "__main__":
    from tmd.storage import STORAGE_FOLDER

    print("Converting irs_aggregate_values.csv → soi.csv format...")
    print()

    # Exclude rentroyalty and estateincome for 2022 (PUF variables
    # don't align with IRS definitions — passive activity limitation
    # differences cause systematic PUF > IRS on both income and loss
    # sides). See session notes for detailed analysis.
    result = convert_potential_targets_to_soi(
        years=[2015, 2021, 2022],
        year_exclude_vars={2022: ["rentroyalty", "estateincome"]},
    )

    print(f"\nTotal rows: {len(result)}")
    for yr in sorted(result.Year.unique()):
        print(f"  Year {yr}: {len(result[result.Year == yr])} rows")
    print(f"Variables: {len(result.Variable.unique())}")
    print()

    # Write to soi.csv
    output_path = STORAGE_FOLDER / "input" / "soi.csv"
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
    print()

    # Compare 2021 with backup soi.csv (if available)
    backup_path = STORAGE_FOLDER / "input" / "soi.csv.bak"
    if backup_path.exists():
        print("Comparing 2021 targeted rows against soi.csv.bak...")
        ok = compare_with_existing_soi(result, backup_path, year=2021)
        if ok:
            print("\nAll 2021 targeted rows match backup. Safe to use.")
        else:
            print("\nWARNING: Some 2021 targeted rows do not match backup!")
