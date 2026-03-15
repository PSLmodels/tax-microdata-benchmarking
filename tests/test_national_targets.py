"""Tests for the national targets pipeline.

Tests are organized by pipeline stage:
  1. Config sanity      — table_layouts.py structural integrity
  2. Extracted CSVs     — data/extracted/{year}/{table}.csv
  3. IRS aggregate values — data/irs_aggregate_values.csv
  4. soi.csv            — storage/input/soi.csv

Stages 2-4 read committed CSV files and require no Excel libraries or
pipeline execution.  Stage 2 also includes optional spot-check tests
that read the raw IRS Excel files directly; these are skipped if the
Excel files are absent (e.g., in CI environments that don't include them).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmd.national_targets.config.table_layouts import (
    COLUMNS,
    DATA_ROWS,
    FILE_NAMES,
    YEARS,
)
from tmd.storage import STORAGE_FOLDER

TARGETS_DIR = Path(__file__).resolve().parents[1] / "tmd" / "national_targets"
DATA_DIR = TARGETS_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
TABLES = ("tab11", "tab12", "tab14", "tab21")

VALID_VAR_TYPES = {"amount", "count", "number"}
VALID_VALUE_FILTERS = {"all", "nz", "gt0", "lt0"}
VALID_MARSTATS = {"all", "single", "mfjss", "mfs", "hoh"}


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def potential_targets():
    return pd.read_csv(DATA_DIR / "irs_aggregate_values.csv")


@pytest.fixture(scope="module")
def soi():
    return pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")


# ── 1. Config sanity ───────────────────────────────────────────────────────────


class TestConfig:
    def test_file_names_and_data_rows_keys_match(self):
        """Every FILE_NAMES key must have a DATA_ROWS entry and vice versa."""
        assert set(FILE_NAMES.keys()) == set(DATA_ROWS.keys())

    def test_columns_keys_are_valid_tables(self):
        assert set(COLUMNS.keys()) == set(TABLES)

    def test_years_tuple_nonempty(self):
        assert len(YEARS) > 0

    def test_each_spec_has_required_fields(self):
        required = {"var_name", "var_type", "value_filter", "marstat", "cols"}
        for table, specs in COLUMNS.items():
            for spec in specs:
                missing = required - set(spec.keys())
                assert not missing, (
                    f"{table}: spec for {spec.get('var_name')} missing {missing}"
                )

    def test_var_type_values_are_valid(self):
        for table, specs in COLUMNS.items():
            for spec in specs:
                assert spec["var_type"] in VALID_VAR_TYPES, (
                    f"{table}/{spec['var_name']}: invalid var_type {spec['var_type']!r}"
                )

    def test_value_filter_values_are_valid(self):
        for table, specs in COLUMNS.items():
            for spec in specs:
                assert spec["value_filter"] in VALID_VALUE_FILTERS, (
                    f"{table}/{spec['var_name']}: invalid value_filter {spec['value_filter']!r}"
                )

    def test_marstat_values_are_valid(self):
        for table, specs in COLUMNS.items():
            for spec in specs:
                assert spec["marstat"] in VALID_MARSTATS, (
                    f"{table}/{spec['var_name']}: invalid marstat {spec['marstat']!r}"
                )

    def test_no_duplicate_column_letters_per_table_year(self):
        """Two specs in the same table must not map to the same column in the
        same year — that would mean reading the wrong IRS column silently."""
        for table, specs in COLUMNS.items():
            for year in YEARS:
                if (table, year) not in FILE_NAMES:
                    continue
                letters = [
                    spec["cols"][year]
                    for spec in specs
                    if year in spec["cols"]
                ]
                seen = set()
                for letter in letters:
                    assert letter not in seen, (
                        f"{table}/{year}: column letter {letter!r} assigned twice"
                    )
                    seen.add(letter)

    def test_no_duplicate_var_key_per_table_year(self):
        """Each (var_name, var_type, value_filter, marstat) combo must be
        unique within a table×year — duplicates would silently overwrite rows
        in the extracted CSV."""
        for table, specs in COLUMNS.items():
            for year in YEARS:
                keys = [
                    (s["var_name"], s["var_type"], s["value_filter"], s["marstat"])
                    for s in specs
                    if year in s["cols"]
                ]
                assert len(keys) == len(set(keys)), (
                    f"{table}/{year}: duplicate var key(s): "
                    + str([k for k in keys if keys.count(k) > 1])
                )

    def test_data_rows_first_before_last(self):
        for key, (first, last) in DATA_ROWS.items():
            assert first < last, f"{key}: DATA_ROWS first row >= last row"

    def test_cols_years_are_known_years(self):
        for table, specs in COLUMNS.items():
            for spec in specs:
                for year in spec["cols"]:
                    assert year in YEARS, (
                        f"{table}/{spec['var_name']}: cols has unknown year {year}"
                    )


# ── 2. Extracted CSVs ──────────────────────────────────────────────────────────


EXTRACTED_COLUMNS = {
    "table", "year", "fname", "var_name", "var_type", "value_filter",
    "marstat", "description", "irs_col_header", "xlcolumn",
    "xl_colnumber", "incsort", "incrange", "xlrownum", "raw_value",
}


class TestExtractedCSVs:
    def test_all_csvs_exist(self):
        missing = []
        for year in YEARS:
            for table in TABLES:
                if (table, year) not in FILE_NAMES:
                    continue
                p = EXTRACTED_DIR / str(year) / f"{table}.csv"
                if not p.exists():
                    missing.append(str(p))
        assert not missing, f"Missing extracted CSVs:\n" + "\n".join(missing)

    @pytest.mark.parametrize("year,table", [
        (year, table)
        for year in YEARS
        for table in TABLES
        if (table, year) in FILE_NAMES
    ])
    def test_csv_has_required_columns(self, year, table):
        df = pd.read_csv(EXTRACTED_DIR / str(year) / f"{table}.csv")
        missing = EXTRACTED_COLUMNS - set(df.columns)
        assert not missing, f"{year}/{table} missing columns: {missing}"

    @pytest.mark.parametrize("year,table", [
        (year, table)
        for year in YEARS
        for table in TABLES
        if (table, year) in FILE_NAMES
    ])
    def test_row_count_matches_data_rows(self, year, table):
        """Row count = (last_row - first_row + 1) × number of specs for that year."""
        first, last = DATA_ROWS[(table, year)]
        n_income_rows = last - first + 1
        n_specs = sum(1 for s in COLUMNS[table] if year in s["cols"])
        expected = n_income_rows * n_specs
        df = pd.read_csv(EXTRACTED_DIR / str(year) / f"{table}.csv")
        assert len(df) == expected, (
            f"{year}/{table}: expected {expected} rows, got {len(df)}"
        )

    @pytest.mark.parametrize("year,table", [
        (year, table)
        for year in YEARS
        for table in TABLES
        if (table, year) in FILE_NAMES
    ])
    def test_irs_col_header_nonempty(self, year, table):
        """Every row should have a non-empty IRS column header (validates
        that spanner/merged cell extraction is working)."""
        df = pd.read_csv(EXTRACTED_DIR / str(year) / f"{table}.csv")
        blank = df["irs_col_header"].isna() | (df["irs_col_header"].str.strip() == "")
        assert not blank.any(), (
            f"{year}/{table}: {blank.sum()} rows have empty irs_col_header"
        )

    def test_wages_header_contains_salaries(self):
        """The wages column header should mention 'Salaries' — a regression
        test for merged-cell spanner extraction."""
        df = pd.read_csv(EXTRACTED_DIR / "2021" / "tab14.csv")
        wage_headers = df.loc[df.var_name == "wages", "irs_col_header"]
        assert wage_headers.str.contains("Salaries", case=False).all(), (
            "Wages column header should contain 'Salaries'"
        )

    # ── Spot-check known IRS totals (require Excel files; skip if absent) ──────

    @pytest.mark.parametrize("year,var_name,var_type,table,expected", [
        (2021, "wages",  "amount", "tab14", 9_022_352_941_000),
        (2022, "wages",  "amount", "tab14", 9_738_950_972_000),
        (2021, "agi",    "amount", "tab11", 14_795_614_070_000),
        (2022, "agi",    "amount", "tab11", 14_833_956_956_000),
        (2021, "agi",    "count",  "tab11", 160_824_340),
    ])
    def test_all_returns_spot_check(self, year, var_name, var_type, table, expected):
        """Verify 'All returns' row against known IRS published totals."""
        df = pd.read_csv(EXTRACTED_DIR / str(year) / f"{table}.csv")
        row = df[
            (df.var_name == var_name) &
            (df.var_type == var_type) &
            (df.marstat == "all") &
            (df.value_filter.isin(["all", "nz"])) &
            (df.incsort == 1)
        ]
        assert len(row) == 1, (
            f"{year}/{table} {var_name}/{var_type}: expected 1 All-returns row, got {len(row)}"
        )
        assert row.iloc[0]["raw_value"] == pytest.approx(expected, rel=1e-9), (
            f"{year}/{table} {var_name}/{var_type} all-returns: "
            f"got {row.iloc[0]['raw_value']}, expected {expected}"
        )

    def test_incsort_is_contiguous_per_var(self):
        """incsort should run 1..N with no gaps for each var within a table/year."""
        for year in YEARS:
            for table in TABLES:
                if (table, year) not in FILE_NAMES:
                    continue
                df = pd.read_csv(EXTRACTED_DIR / str(year) / f"{table}.csv")
                first, last = DATA_ROWS[(table, year)]
                n_expected = last - first + 1
                for (vn, vt, vf, ms), grp in df.groupby(
                    ["var_name", "var_type", "value_filter", "marstat"]
                ):
                    sorts = sorted(grp["incsort"].tolist())
                    assert sorts == list(range(1, n_expected + 1)), (
                        f"{year}/{table} {vn}/{vt}/{vf}/{ms}: "
                        f"incsort not contiguous 1..{n_expected}"
                    )


# ── 3. IRS aggregate values ───────────────────────────────────────────────────


class TestPotentialTargets:
    def test_row_count(self, potential_targets):
        assert len(potential_targets) == 6281

    def test_years_present(self, potential_targets):
        assert set(potential_targets["year"].unique()) == {2015, 2021, 2022}

    def test_year_row_counts(self, potential_targets):
        assert len(potential_targets[potential_targets.year == 2015]) == 2023
        assert len(potential_targets[potential_targets.year == 2021]) == 2129
        assert len(potential_targets[potential_targets.year == 2022]) == 2129

    def test_unique_var_name_count(self, potential_targets):
        assert potential_targets["var_name"].nunique() == 42

    def test_no_missing_ptarget(self, potential_targets):
        assert potential_targets["ptarget"].notna().all()

    def test_required_columns_present(self, potential_targets):
        required = {
            "rownum", "idbase", "year", "table", "var_name", "var_type",
            "var_description", "value_filter", "subgroup", "marstat",
            "incsort", "incrange", "ptarget", "fname", "xlcell",
            "xl_colnumber", "xlcolumn", "xlrownum",
        }
        missing = required - set(potential_targets.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_idbase_row_unique(self, potential_targets):
        """(idbase, incsort) must be unique within each source table.
        idbase is a per-variable-group key; incsort distinguishes income
        brackets.  Note: the same (idbase, incsort) CAN appear more than once
        across different tables — potential_targets preserves cross-table
        redundancy by design.  Duplicates within a single table would be a bug."""
        pt = potential_targets
        assert not pt.duplicated(subset=["table", "idbase", "incsort"]).any()

    def test_rownum_is_sequential(self, potential_targets):
        assert list(potential_targets["rownum"]) == list(
            range(1, len(potential_targets) + 1)
        )

    def test_wages_2021_total(self, potential_targets):
        val = potential_targets[
            (potential_targets.year == 2021) &
            (potential_targets.var_name == "wages") &
            (potential_targets.var_type == "amount") &
            (potential_targets.value_filter == "nz") &
            (potential_targets.marstat == "all") &
            (potential_targets.incsort == 1)
        ]["ptarget"].values[0]
        assert val == pytest.approx(9_022_352_941_000, rel=1e-9)

    def test_wages_2022_total(self, potential_targets):
        val = potential_targets[
            (potential_targets.year == 2022) &
            (potential_targets.var_name == "wages") &
            (potential_targets.var_type == "amount") &
            (potential_targets.value_filter == "nz") &
            (potential_targets.marstat == "all") &
            (potential_targets.incsort == 1)
        ]["ptarget"].values[0]
        assert val == pytest.approx(9_738_950_972_000, rel=1e-9)

    def test_agi_2021_total(self, potential_targets):
        val = potential_targets[
            (potential_targets.year == 2021) &
            (potential_targets.var_name == "agi") &
            (potential_targets.var_type == "amount") &
            (potential_targets.table == "tab11") &
            (potential_targets.incsort == 1)
        ]["ptarget"].values[0]
        assert val == pytest.approx(14_795_614_070_000, rel=1e-9)

    def test_agi_2022_total(self, potential_targets):
        val = potential_targets[
            (potential_targets.year == 2022) &
            (potential_targets.var_name == "agi") &
            (potential_targets.var_type == "amount") &
            (potential_targets.table == "tab11") &
            (potential_targets.incsort == 1)
        ]["ptarget"].values[0]
        assert val == pytest.approx(14_833_956_956_000, rel=1e-9)

    def test_marstat_filing_status_sum_check(self, potential_targets):
        """For tab12: sum of single+mfjss+mfs+hoh return counts should equal
        the 'all' total for the All-returns row (incsort=1)."""
        for year in [2021, 2022]:
            total_all = potential_targets[
                (potential_targets.year == year) &
                (potential_targets.table == "tab12") &
                (potential_targets.var_name == "agi") &
                (potential_targets.var_type == "count") &
                (potential_targets.marstat == "all") &
                (potential_targets.incsort == 1)
            ]["ptarget"].values[0]

            total_parts = potential_targets[
                (potential_targets.year == year) &
                (potential_targets.table == "tab12") &
                (potential_targets.var_name == "agi") &
                (potential_targets.var_type == "count") &
                (potential_targets.marstat.isin(["single", "mfjss", "mfs", "hoh"])) &
                (potential_targets.incsort == 1)
            ]["ptarget"].sum()

            assert total_parts == pytest.approx(total_all, rel=1e-6), (
                f"{year} tab12 marstat sum mismatch: "
                f"parts={total_parts}, all={total_all}"
            )

    def test_amounts_in_dollars_not_thousands(self, potential_targets):
        """2021 wages total should be ~9 trillion dollars, not ~9 billion
        (which would indicate a missing ×1000 conversion)."""
        wages = potential_targets[
            (potential_targets.year == 2021) &
            (potential_targets.var_name == "wages") &
            (potential_targets.var_type == "amount") &
            (potential_targets.incsort == 1)
        ]["ptarget"].values[0]
        assert wages > 1e12, (
            f"Wages appear to be in thousands, not dollars: {wages:.0f}"
        )


# ── 4. soi.csv ─────────────────────────────────────────────────────────────────


SOI_COLUMNS = [
    "Year", "SOI table", "XLSX column", "XLSX row", "Variable",
    "Filing status", "AGI lower bound", "AGI upper bound",
    "Count", "Taxable only", "Full population", "Value",
]


class TestSoi:
    def test_required_columns(self, soi):
        missing = set(SOI_COLUMNS) - set(soi.columns)
        assert not missing, f"soi.csv missing columns: {missing}"

    def test_years_present(self, soi):
        assert set(soi["Year"].unique()) == {2015, 2021, 2022}

    def test_row_counts_per_year(self, soi):
        assert len(soi[soi.Year == 2015]) == 1860
        assert len(soi[soi.Year == 2021]) == 1986
        assert len(soi[soi.Year == 2022]) == 1826

    def test_no_missing_values(self, soi):
        for col in SOI_COLUMNS:
            assert soi[col].notna().all(), f"soi.csv has NaN in column {col}"

    def test_employment_income_2021_total(self, soi):
        """employment_income (=wages) for 2021 All-returns should match IRS."""
        val = soi[
            (soi.Year == 2021) &
            (soi.Variable == "employment_income") &
            (soi["Filing status"] == "All") &
            (soi["AGI lower bound"] == -np.inf) &
            (soi["AGI upper bound"] == np.inf) &
            (~soi["Count"])
        ]["Value"].values[0]
        assert val == pytest.approx(9_022_352_941_000, rel=1e-9)

    def test_employment_income_2022_total(self, soi):
        val = soi[
            (soi.Year == 2022) &
            (soi.Variable == "employment_income") &
            (soi["Filing status"] == "All") &
            (soi["AGI lower bound"] == -np.inf) &
            (soi["AGI upper bound"] == np.inf) &
            (~soi["Count"])
        ]["Value"].values[0]
        assert val == pytest.approx(9_738_950_972_000, rel=1e-9)

    def test_full_population_flag(self, soi):
        """Full population = True iff Filing status=All, AGI=(-inf,inf),
        Taxable only=False."""
        expected_full_pop = (
            (soi["Filing status"] == "All") &
            (soi["AGI lower bound"] == -np.inf) &
            (soi["AGI upper bound"] == np.inf) &
            (~soi["Taxable only"])
        )
        assert (soi["Full population"] == expected_full_pop).all()

    def test_taxable_only_is_always_false(self, soi):
        """irs_aggregate_values.csv has no taxable-only subgroup;
        Taxable only should be False for every row."""
        assert (~soi["Taxable only"]).all()

    def test_partner_scorp_aggregation_2021(self, soi):
        """For 2021, partnership_and_s_corp_income should exceed
        s_corporation_net_income alone (because S-corp was added to partnership)."""
        partner = soi[
            (soi.Year == 2021) &
            (soi.Variable == "partnership_and_s_corp_income") &
            (soi["Filing status"] == "All") &
            (soi["AGI lower bound"] == -np.inf) &
            (soi["AGI upper bound"] == np.inf) &
            (~soi["Count"])
        ]["Value"].values[0]

        scorp = soi[
            (soi.Year == 2021) &
            (soi.Variable == "s_corporation_net_income") &
            (soi["Filing status"] == "All") &
            (soi["AGI lower bound"] == -np.inf) &
            (soi["AGI upper bound"] == np.inf) &
            (~soi["Count"])
        ]["Value"].values[0]

        assert partner > scorp, (
            "partnership_and_s_corp_income should be > s_corp alone "
            f"(partner={partner}, scorp={scorp})"
        )

    def test_no_duplicate_targeted_rows(self, soi):
        """No two rows should share the same (Year, Variable, Filing status,
        AGI lower, AGI upper, Count, Taxable only) — that would mean the
        optimizer sees ambiguous targets."""
        key_cols = [
            "Year", "Variable", "Filing status",
            "AGI lower bound", "AGI upper bound", "Count", "Taxable only",
        ]
        dupes = soi.duplicated(subset=key_cols, keep=False)
        assert not dupes.any(), (
            f"{dupes.sum()} duplicate targeted rows in soi.csv:\n"
            + soi[dupes][key_cols].to_string()
        )
