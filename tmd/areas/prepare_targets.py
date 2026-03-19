# pylint: disable=import-outside-toplevel
"""
Prepare state target files from SOI data and TMD national totals.

Builds per-state target CSV files in two layers:
  1. **Base targets** (recipe-driven): AGI, wages, returns by filing
     status, pensions, Social Security, SALT, partnership income —
     expanded across AGI bins.
  2. **Extended targets**: additional variables using SOI geographic
     shares, Census tax shares (SALT), and aggregate credit targets
     (EITC, CTC).

Both layers are written in a single pass per area.

Usage:
    # All states, default 2022 SOI shares:
    python -m tmd.areas.prepare_targets --scope states

    # Specific states:
    python -m tmd.areas.prepare_targets --scope MN,CA,TX

    # Use 2021 SOI shares:
    python -m tmd.areas.prepare_targets --scope states --year 2021
"""

import argparse
import time
from pathlib import Path

from tmd.areas import AREAS_FOLDER

# --- Default paths ---

_REPO_ROOT = Path(__file__).parent.parent.parent
_RECIPES = _REPO_ROOT / "tmd" / "areas" / "prepare" / "recipes"
_STATE_RECIPE = _RECIPES / "states.json"
_STATE_VARMAP = _RECIPES / "state_variable_mapping.csv"
_STATE_TARGET_DIR = AREAS_FOLDER / "targets" / "states"
_CACHED_ALLVARS = (
    _REPO_ROOT / "tmd" / "storage" / "output" / "cached_allvars.csv"
)

# Areas to exclude from target files
_EXCLUDE = {"US", "PR", "OA"}


def prepare_state_targets(
    scope="states",
    area_data_year=2022,
    national_data_year=0,
    pop_year=0,
):
    """
    Build enhanced targets and write per-state target CSV files.

    Parameters
    ----------
    scope : str
        'states' or comma-separated state codes.
    area_data_year : int
        SOI data year for geographic distribution.
    national_data_year : int
        TMD data year for national levels (0 = same as area_data_year).
    pop_year : int
        Population year (0 = same as area_data_year).

    Returns
    -------
    dict
        Mapping of area code → target count for each area processed.
    """
    from tmd.areas.prepare.constants import AreaType
    from tmd.areas.prepare.extended_targets import build_extended_targets
    from tmd.areas.prepare.target_file_writer import write_area_target_files
    from tmd.areas.prepare.target_sharing import prepare_area_targets

    specific = _parse_scope(scope)

    print(f"Preparing state targets (SOI year {area_data_year})...")
    t0 = time.time()

    # Step 1: Build base targets (recipe-driven, TMD × SOI shares)
    enhanced = prepare_area_targets(
        area_type=AreaType.STATE,
        area_data_year=area_data_year,
        national_data_year=national_data_year,
        pop_year=pop_year,
    )
    enhanced = enhanced[~enhanced["area"].isin(_EXCLUDE)]
    if specific:
        enhanced = enhanced[enhanced["area"].isin(specific)]

    # Step 2: Build extended targets (SOI-shared, Census-shared, credits)
    areas_to_process = (
        specific if specific else sorted(enhanced["area"].unique())
    )
    print(f"  Building extended targets for {len(areas_to_process)} areas...")
    extra_targets = build_extended_targets(
        cached_allvars_path=_CACHED_ALLVARS,
        soi_year=area_data_year,
        areas=areas_to_process,
    )

    # Step 3: Write target files (single pass: base + extended)
    result = write_area_target_files(
        recipe_path=_STATE_RECIPE,
        enhanced_targets=enhanced,
        variable_mapping_path=_STATE_VARMAP,
        output_dir=_STATE_TARGET_DIR,
        extra_targets=extra_targets,
    )

    elapsed = time.time() - t0
    n_areas = len(result)
    n_targets = next(iter(result.values()), 0)
    print(
        f"  Wrote {n_areas} state target files "
        f"({n_targets} targets each, {elapsed:.1f}s)"
    )
    return result


def _parse_scope(scope):
    """Parse scope string into a list of state codes or None for all."""
    scope_lower = scope.lower().strip()
    if scope_lower in ("states", "all"):
        return None
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    return [c for c in codes if len(c) == 2 and c not in _EXCLUDE]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare state target files from SOI + TMD data",
    )
    parser.add_argument(
        "--scope",
        default="states",
        help="'states' or comma-separated state codes (e.g., 'MN,CA,TX')",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="SOI area data year (default: 2022)",
    )
    parser.add_argument(
        "--national-year",
        type=int,
        default=0,
        help="TMD national data year (default: same as --year)",
    )
    parser.add_argument(
        "--pop-year",
        type=int,
        default=0,
        help="Population year (default: same as --year)",
    )
    args = parser.parse_args()

    prepare_state_targets(
        scope=args.scope,
        area_data_year=args.year,
        national_data_year=args.national_year,
        pop_year=args.pop_year,
    )


if __name__ == "__main__":
    main()
