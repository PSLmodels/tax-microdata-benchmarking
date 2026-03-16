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

from pathlib import Path

import pandas as pd
import xlrd

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
    wb = xlrd.open_workbook(str(fpath), formatting_info=True)
    ws = wb.sheet_by_index(0)

    # Income range labels are always in column A (index 0).
    # xlrd uses 0-based row indexing; spreadsheet rows are 1-based.
    incrange_labels = [
        str(ws.cell_value(r - 1, 0)).strip()
        for r in range(first_row, last_row + 1)
    ]

    # Build merge map: expand merged-cell values to every covered cell.
    # Without this, only the top-left cell of a spanner carries the value;
    # all other cells in the span read as empty.
    merge_map: dict = {}
    for row_lo, row_hi, col_lo, col_hi in ws.merged_cells:
        val = ws.cell_value(row_lo, col_lo)
        for r in range(row_lo, row_hi):
            for c in range(col_lo, col_hi):
                merge_map[(r, c)] = val

    def cell_text(row_0: int, col_0: int) -> str:
        """Return cell value, falling back to merged-region value if blank."""
        v = ws.cell_value(row_0, col_0)
        if v == "" or v is None:
            v = merge_map.get((row_0, col_0), "")
        return str(v).strip().replace("\n", " ").replace("  ", " ")

    def get_col_header(col_idx: int) -> str:
        """Build a clean hierarchical header string for one column.

        Skips:
          row 1  — table title (boilerplate, same for every column)
          row 2  — "All figures are estimates..." (boilerplate)
          row first_row-1  — IRS column-number row (numeric labels 1.0, 2.0…)

        Expands merged/spanner cells so that every column gets the full
        parent header text, not just the first column in the span.

        Deduplicates consecutive identical tokens (artefact of multi-row
        merges where the same text is stored in both covered rows).
        """
        skip_rows_0based = {
            0,  # row 1: table title
            1,  # row 2: "All figures..." note
            first_row - 2,  # last header row: IRS column numbers
        }
        raw_parts = []
        for r in range(first_row - 1):  # 0-based rows before first data row
            if r in skip_rows_0based:
                continue
            v = cell_text(r, col_idx)
            if v:
                raw_parts.append(v)

        # Deduplicate consecutive identical values produced by multi-row merges
        deduped = []
        for part in raw_parts:
            if not deduped or part != deduped[-1]:
                deduped.append(part)

        return " | ".join(deduped)

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
    count = 0

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
            count += 1

    return count


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
    n_written = extract_all(
        years=args.years,
        tables=args.tables,
        overwrite=args.overwrite,
    )
    print()
    if n_written:
        print(
            f"Done. {n_written} CSVs written to "
            f"{EXTRACTED_DIR.relative_to(Path.cwd())}/"
        )
    else:
        print(
            "Done. All CSVs already up to date "
            "(use --overwrite to re-extract)."
        )
