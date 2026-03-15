"""Build irs_aggregate_values.csv from extracted IRS CSV files.

Reads the per-table CSV files produced by extract_irs_to_csv.py and
assembles them into irs_aggregate_values.csv — the input to
potential_targets_to_soi.py.

The output schema matches the existing file so that potential_targets_to_soi.py
works unchanged.

Output columns:
    rownum, idbase, year, table, var_name, var_type, var_description,
    value_filter, subgroup, marstat, incsort, incrange, ptarget,
    fname, xlcell, xl_colnumber, xlcolumn, xlrownum
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmd.national_targets.config.table_layouts import YEARS

DATA_DIR = Path(__file__).parent / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
OUTPUT_PATH = DATA_DIR / "irs_aggregate_values.csv"

TABLES = ("tab11", "tab12", "tab14", "tab21")

# tab21 covers itemized-deduction filers only, but we label all rows
# "filers" for compatibility with potential_targets_to_soi.py, which
# does not filter on the subgroup column.
_TABLE_SUBGROUP = {
    "tab11": "filers",
    "tab12": "filers",
    "tab14": "filers",
    "tab21": "filers",
}

OUTPUT_COLUMNS = [
    "rownum",
    "idbase",
    "year",
    "table",
    "var_name",
    "var_type",
    "var_description",
    "value_filter",
    "subgroup",
    "marstat",
    "incsort",
    "incrange",
    "ptarget",
    "fname",
    "xlcell",
    "xl_colnumber",
    "xlcolumn",
    "xlrownum",
]


def _xlcell(col_letter: str, row: int) -> str:
    """Format an Excel cell reference, e.g. ('D', 9) → 'D9'."""
    return f"{col_letter}{row}"


def build_potential_targets(
    years=YEARS,
    tables=TABLES,
    extracted_dir=EXTRACTED_DIR,
    output_path=OUTPUT_PATH,
) -> pd.DataFrame:
    """Read extracted CSVs and assemble irs_aggregate_values.csv.

    Args:
        years:         Iterable of years to include.
        tables:        Iterable of table keys to include.
        extracted_dir: Directory containing extracted CSVs.
        output_path:   Where to write the assembled CSV.

    Returns:
        The assembled DataFrame (also written to output_path).
    """
    extracted_dir = Path(extracted_dir)
    dfs = []

    for year in years:
        for table in tables:
            csv_path = extracted_dir / str(year) / f"{table}.csv"
            if not csv_path.exists():
                print(
                    f"  WARNING: {csv_path} not found"
                    f" — run extract_irs_to_csv.py first"
                )
                continue
            dfs.append(pd.read_csv(csv_path))

    if not dfs:
        raise FileNotFoundError(
            f"No extracted CSVs found in {extracted_dir}. "
            "Run extract_irs_to_csv.py first."
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Drop rows where the IRS cell held a footnote marker (missing data).
    n_missing = combined["raw_value"].isna().sum()
    if n_missing:
        print(
            f"  Dropping {n_missing} rows with missing IRS values"
            " (footnote markers or suppressed data)"
        )
    combined = combined.dropna(subset=["raw_value"])

    # raw_value is already in dollars (conversion done in extract step).
    combined = combined.rename(
        columns={
            "raw_value": "ptarget",
            "description": "var_description",
        }
    )

    combined["subgroup"] = combined["table"].map(_TABLE_SUBGROUP)
    combined["xlcell"] = combined.apply(
        lambda r: _xlcell(r["xlcolumn"], r["xlrownum"]), axis=1
    )

    # Sort for reproducibility before assigning row numbers.
    combined = combined.sort_values(
        [
            "year",
            "table",
            "var_name",
            "var_type",
            "value_filter",
            "marstat",
            "incsort",
        ]
    ).reset_index(drop=True)

    combined.insert(0, "rownum", range(1, len(combined) + 1))
    combined["idbase"] = (
        combined["year"].astype(str)
        + "_"
        + combined["table"]
        + "_"
        + combined["var_name"]
        + "_"
        + combined["var_type"]
        + "_"
        + combined["value_filter"]
        + "_"
        + combined["marstat"]
    )

    result = combined[OUTPUT_COLUMNS].copy()

    out_path = Path(output_path)
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path.relative_to(Path.cwd())}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build irs_aggregate_values.csv from extracted CSVs."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(YEARS),
        help="Years to include (default: all configured years)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(TABLES),
        help="Tables to include (default: tab11 tab12 tab14 tab21)",
    )
    args = parser.parse_args()

    print("Building irs_aggregate_values.csv...")
    print(f"  Years:  {args.years}")
    print(f"  Tables: {args.tables}")
    print()
    result = build_potential_targets(years=args.years, tables=args.tables)
    print()
    print("Summary:")
    print(f"  Total rows:      {len(result)}")
    for yr in sorted(result.year.unique()):
        n = len(result[result.year == yr])
        print(f"  {yr}: {n} rows")
    print(f"  Unique var_names: {result.var_name.nunique()}")
