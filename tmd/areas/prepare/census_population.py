"""
Census state population data for area target preparation.

Provides state population estimates from the Census Bureau Population
Estimates Program (PEP), stored in a JSON file under ``data/``.

A user-supplied CSV can override the defaults.

Note: CD (Congressional District) population support will be added
in a future PR.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def _load_population_json(filename: str) -> Dict[str, Dict[str, int]]:
    """Load a population JSON file and return {year_str: {area: pop}}."""
    path = _DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filter out metadata keys (those starting with _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_state_population(
    year: int,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return state population as DataFrame with columns (stabbr, population).

    Parameters
    ----------
    year : int
        Calendar year for the population estimate.
    csv_path : Path, optional
        Path to a CSV with columns ``stabbr`` and a population column
        (named ``pop{year}`` or ``population``).  If provided, this
        overrides the default data.

    Returns
    -------
    pd.DataFrame
        Columns: stabbr (str), population (int).
        Includes 50 states, DC, PR, and US.
    """
    if csv_path is not None:
        return _read_population_csv(csv_path, year)
    all_years = _load_population_json("state_populations.json")
    year_str = str(year)
    if year_str not in all_years:
        raise ValueError(
            f"No state population data for {year}. "
            f"Available years: {sorted(all_years.keys())}. "
            f"Supply a csv_path to use custom data."
        )
    pop = all_years[year_str]
    df = pd.DataFrame(list(pop.items()), columns=["stabbr", "population"])
    df["population"] = df["population"].astype(int)
    return df.sort_values("stabbr").reset_index(drop=True)


def _read_population_csv(csv_path: Path, year: int) -> pd.DataFrame:
    """Read state population CSV, normalise column names."""
    df = pd.read_csv(csv_path)
    pop_col = f"pop{year}"
    if pop_col in df.columns:
        df = df.rename(columns={pop_col: "population"})
    elif "population" not in df.columns:
        pop_cols = [c for c in df.columns if c.startswith("pop")]
        if len(pop_cols) == 1:
            df = df.rename(columns={pop_cols[0]: "population"})
        else:
            raise ValueError(
                f"Cannot find population column in {csv_path}. "
                f"Expected 'pop{year}' or 'population'."
            )
    df = df[["stabbr", "population"]].copy()
    df["population"] = df["population"].astype(int)
    return df.sort_values("stabbr").reset_index(drop=True)
