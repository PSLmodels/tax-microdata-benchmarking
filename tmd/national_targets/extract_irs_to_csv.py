"""Extract IRS SOI Excel files to long-format CSV files.

This is a one-time (or per-update) step that reads the raw IRS .xls files
and writes clean CSV files to data/extracted/{year}/{table}.csv.

Why extract to CSV first?
  - Separates xlrd (old .xls parser) from the main pipeline
  - CSV files are human-readable, git-diffable, and fast to read
  - If an IRS file is ever corrupted, the CSV is an independent backup
  - Downstream build_targets.py needs no Excel library dependency

When to re-run this script:
  - First-time setup
  - A new year of IRS data is downloaded
  - An extraction error is discovered and corrected

The IRS Excel files remain the ground truth; extracted CSVs are derived.

Units: IRS amount columns are stored in thousands of dollars.
       This script converts amounts to dollars (multiplies by 1000).
       Count and number columns are written as-is (already full units).
"""

import sys
from pathlib import Path

import pandas as pd
import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tmd.national_targets.config.table_layouts import (
    COLUMNS,
    DATA_ROWS,
    FILE_NAMES,
    YEARS,
)

DATA_DIR = Path(__file__).parent / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
TABLES = ("tab11", "tab12", "tab14", "tab21")


def col_letter_to_idx(col: str) -> int:
    """Convert an Excel column letter to a 0-based column index.

    Examples: 'A' → 0, 'B' → 1, 'Z' → 25, 'AA' → 26, 'EI' → 138.
    """
    idx = 0
    for char in col.upper():
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def read_irs_table(table: str, year: int) -> pd.DataFrame:
    """Read one IRS Excel file and return a long-format DataFrame.

    Output columns:
        table, year, fname,
        var_name, var_type, value_filter, marstat, description,
        irs_col_header,   ← raw text IRS printed as the column heading
        xlcolumn, xl_colnumber, incsort, incrange, xlrownum,
        raw_value

    irs_col_header is the concatenation of all non-empty header cell values
    above the data (rows 1 through first_row-1) for that column, joined by
    " | ".  This allows independent verification that a column letter maps
    to the intended IRS variable.

    raw_value is in dollars (amounts already multiplied by 1000).
    Returns an empty DataFrame if the file is not found.
    """
    key = (table, year)
    if key not in FILE_NAMES:
        return pd.DataFrame()

    fname = FILE_NAMES[key]
    fpath = DATA_DIR / str(year) / fname

    if not fpath.exists():
        print(f"  WARNING: {fpath} not found — skipping")
        return pd.DataFrame()

    first_row, last_row = DATA_ROWS[key]
    wb = xlrd.open_workbook(str(fpath))
    ws = wb.sheet_by_index(0)

    # Income range labels are always in column A (index 0).
    # xlrd uses 0-based row indexing; spreadsheet rows are 1-based.
    incrange_labels = [
        str(ws.cell_value(r - 1, 0)).strip()
        for r in range(first_row, last_row + 1)
    ]

    def get_col_header(col_idx: int) -> str:
        """Collect IRS header text for a column from all rows above the data."""
        parts = []
        for r in range(first_row - 1):  # rows 0..(first_row-2) in 0-based
            v = str(ws.cell_value(r, col_idx)).strip()
            if v:
                # Normalise multi-line cells (xlrd preserves \n)
                parts.append(v.replace("\n", " ").replace("  ", " "))
        return " | ".join(parts) if parts else ""

    rows = []
    for spec in COLUMNS[table]:
        col_letter = spec["cols"].get(year)
        if col_letter is None:
            continue  # This variable is not in this year's file

        col_idx = col_letter_to_idx(col_letter)
        xl_colnumber = col_idx + 1  # 1-based, matches IRS column numbering
        var_type = spec["var_type"]
        irs_header = get_col_header(col_idx)

        for i, excel_row in enumerate(range(first_row, last_row + 1)):
            cell = ws.cell(excel_row - 1, col_idx)
            raw = cell.value

            # IRS uses "[1]", "[2]", etc. as footnote markers for suppressed
            # or unavailable data.  Treat these as missing.
            if not isinstance(raw, (int, float)):
                raw = None
            elif var_type == "amount":
                raw = raw * 1000  # thousands → dollars

            rows.append(
                {
                    "table": table,
                    "year": year,
                    "fname": fname,
                    "var_name": spec["var_name"],
                    "var_type": var_type,
                    "value_filter": spec["value_filter"],
                    "marstat": spec["marstat"],
                    "description": spec.get("description", ""),
                    "irs_col_header": irs_header,
                    "xlcolumn": col_letter,
                    "xl_colnumber": xl_colnumber,
                    "incsort": i + 1,
                    "incrange": incrange_labels[i],
                    "xlrownum": excel_row,
                    "raw_value": raw,
                }
            )

    return pd.DataFrame(rows)


def extract_all(
    years=YEARS,
    tables=TABLES,
    output_dir=EXTRACTED_DIR,
    overwrite=False,
):
    """Extract all IRS tables to CSV files.

    Writes: data/extracted/{year}/{table}.csv

    Args:
        years:      Iterable of years to extract.
        tables:     Iterable of table keys to extract.
        output_dir: Root directory for extracted CSVs.
        overwrite:  If False (default), skip files that already exist.
    """
    output_dir = Path(output_dir)

    for year in years:
        year_dir = output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        for table in tables:
            if (table, year) not in FILE_NAMES:
                continue

            out_path = year_dir / f"{table}.csv"
            if out_path.exists() and not overwrite:
                print(f"  Skipping {year}/{table}.csv (already exists)")
                continue

            print(f"  Extracting {year} {table} ...", end=" ", flush=True)
            df = read_irs_table(table, year)

            if df.empty:
                print("no data")
                continue

            df.to_csv(out_path, index=False)
            rel = out_path.relative_to(Path.cwd())
            print(f"{len(df)} rows → {rel}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract IRS SOI Excel files to CSV."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(YEARS),
        help="Years to extract (default: all configured years)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(TABLES),
        help="Tables to extract (default: tab11 tab12 tab14 tab21)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files",
    )
    args = parser.parse_args()

    print("Extracting IRS Excel files to CSV...")
    print(f"  Years:  {args.years}")
    print(f"  Tables: {args.tables}")
    print()
    extract_all(
        years=args.years,
        tables=args.tables,
        overwrite=args.overwrite,
    )
    print()
    print(f"Done. CSVs written to {EXTRACTED_DIR.relative_to(Path.cwd())}/")
