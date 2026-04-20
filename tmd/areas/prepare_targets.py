# pylint: disable=import-outside-toplevel,inconsistent-quotes
"""
Prepare area target files from SOI data and TMD national totals.

Builds per-area target CSV files for states or congressional districts.

Usage:
    # All states:
    python -m tmd.areas.prepare_targets --scope states

    # All congressional districts:
    python -m tmd.areas.prepare_targets --scope cds

    # Specific states or CDs:
    python -m tmd.areas.prepare_targets --scope MN,CA,TX
    python -m tmd.areas.prepare_targets --scope MN01,MN02,CA52
"""

import argparse
import time
from pathlib import Path

from tmd.areas import AREAS_FOLDER
from tmd.areas.create_area_weights import cd_target_dir

# --- Default paths ---

_REPO_ROOT = Path(__file__).parent.parent.parent
_RECIPES = _REPO_ROOT / "tmd" / "areas" / "prepare" / "recipes"
_STATE_RECIPE = _RECIPES / "states.json"
_STATE_VARMAP = _RECIPES / "state_variable_mapping.csv"
_STATE_TARGET_DIR = AREAS_FOLDER / "targets" / "states"
# CD recipe and variable mapping are SHARED across Congress sessions —
# only the set of area codes and the underlying geographic shares
# differ between 118 and 119.  The output directory depends on the
# target Congress session (see ``cd_target_dir``).
_CD_RECIPE = _RECIPES / "cds.json"
_CD_VARMAP = _RECIPES / "cd_variable_mapping.csv"
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


def prepare_cd_targets(
    scope="cds",
    area_data_year=2022,
    national_data_year=0,
    congress=None,
):
    """
    Build enhanced targets and write per-CD target CSV files.

    Parameters
    ----------
    scope : str
        'cds' for all CDs, or comma-separated CD codes (e.g., 'MN01,CA52').
    area_data_year : int
        SOI data year for geographic distribution.
    national_data_year : int
        TMD data year for national levels (0 = same as area_data_year).
    congress : int
        Target Congressional session (118 or 119).  Required — no default.
        Outputs are written to ``targets/cds_{congress}/``.

    Returns
    -------
    dict
        Mapping of area code → target count for each area processed.
    """
    if congress is None:
        raise ValueError(
            "prepare_cd_targets requires an explicit congress argument "
            "(118 or 119)."
        )
    from tmd.areas.prepare.constants import AreaType
    from tmd.areas.prepare.target_file_writer import write_area_target_files
    from tmd.areas.prepare.target_sharing import prepare_area_targets

    specific = _parse_cd_scope(scope)

    print(
        f"Preparing CD targets for Congress {congress} "
        f"(SOI year {area_data_year})..."
    )
    t0 = time.time()

    # Step 1: Build base targets (recipe-driven, TMD × SOI shares)
    enhanced = prepare_area_targets(
        area_type=AreaType.CD,
        area_data_year=area_data_year,
        national_data_year=national_data_year,
        congress=congress,
    )
    if specific:
        enhanced = enhanced[enhanced["area"].isin(specific)]

    # Step 2: Write target files (no extended targets for CDs yet)
    result = write_area_target_files(
        recipe_path=_CD_RECIPE,
        enhanced_targets=enhanced,
        variable_mapping_path=_CD_VARMAP,
        output_dir=cd_target_dir(congress),
    )

    elapsed = time.time() - t0
    n_areas = len(result)
    n_targets = next(iter(result.values()), 0) if result else 0
    print(
        f"  Wrote {n_areas} CD target files "
        f"({n_targets} targets each, {elapsed:.1f}s)"
    )
    return result


# --- New spec-based target preparation ---

_CD_SPEC = _RECIPES / "cd_target_spec.csv"
_STATE_SPEC = _RECIPES / "state_target_spec.csv"
_STATE_SHARES = (
    _REPO_ROOT / "tmd" / "areas" / "prepare" / "data" / "states_shares.csv"
)


def cd_shares_path(congress: int) -> Path:
    """Return the CD shares CSV path for the given Congress session."""
    if congress not in (118, 119):
        raise ValueError(
            f"Unsupported Congress session: {congress}. Supported: (118, 119)"
        )
    return (
        _REPO_ROOT
        / "tmd"
        / "areas"
        / "prepare"
        / "data"
        / f"cds_{congress}_shares.csv"
    )


def prepare_targets_from_spec(
    scope="cds",
    area_data_year=2022,  # pylint: disable=unused-argument
    congress=None,
):
    """
    Build per-area target CSV files from spec + shares + TMD sums.

    This is the new spec-based pipeline that separates:
      - Shares (stable, from SOI data)
      - Spec (the recipe — which targets to include)
      - TMD national sums (volatile, recomputed each run)

    Parameters
    ----------
    scope : str
        'cds', 'states', or comma-separated area codes.
    area_data_year : int
        SOI data year (for selecting the right shares file).
    congress : int, optional
        Target Congressional session for CD scope (118 or 119).
        Required when scope is a CD scope; ignored for states.
    """
    import numpy as np
    import pandas as pd

    from tmd.areas.prepare.constants import (
        ALL_SHARING_MAPPINGS,
        CD_AGI_CUTS,
        STATE_AGI_CUTS,
    )
    from tmd.areas.prepare_shares import (
        EXTENDED_SHARING_MAPPINGS,
    )
    from tmd.areas.prepare.target_sharing import (
        compute_tmd_national_sums,
    )

    scope_lower = scope.lower().strip()
    first_code = scope.split(",")[0].strip()
    is_cd = scope_lower == "cds" or (
        scope_lower not in ("states", "all") and len(first_code) > 2
    )

    if is_cd:
        if congress is None:
            raise ValueError(
                "CD scope requires an explicit congress argument "
                "(118 or 119)."
            )
        spec_path = _CD_SPEC
        shares_path = cd_shares_path(congress)
        output_dir = cd_target_dir(congress)
        agi_cuts = CD_AGI_CUTS
    else:
        spec_path = _STATE_SPEC
        shares_path = _STATE_SHARES
        output_dir = _STATE_TARGET_DIR
        agi_cuts = STATE_AGI_CUTS

    t0 = time.time()

    # 1. Read spec
    spec = pd.read_csv(spec_path)
    # Drop description column for matching
    spec_keys = spec[
        ["varname", "count", "scope", "fstatus", "agilo", "agihi"]
    ].copy()
    print(f"Spec: {len(spec_keys)} targets from {spec_path.name}")

    # 2. Read shares
    shares = pd.read_csv(shares_path)
    print(
        f"Shares: {len(shares):,} rows,"
        f" {shares['area'].nunique()} areas"
        f" from {shares_path.name}"
    )

    # Filter areas if specific scope given
    if scope_lower not in ("cds", "states", "all"):
        codes = [c.strip().upper() for c in scope.split(",")]
        shares = shares[shares["area"].isin(codes)]
        print(f"  Filtered to {shares['area'].nunique()} areas")

    # 3. Compute TMD national sums (base + extended)
    all_mappings = ALL_SHARING_MAPPINGS + EXTENDED_SHARING_MAPPINGS
    print("Computing TMD national sums...")
    tmd_sums = compute_tmd_national_sums(
        _CACHED_ALLVARS, all_mappings, agi_cuts
    )

    # Build lookup: (varname, count, fstatus, agistub) → tmdsum
    # Need to map tmdvar to varname (they're the same in our system)
    tmd_lookup = {}
    for _, r in tmd_sums.iterrows():
        key = (
            r["tmdvar"],
            int(r["count"]),
            int(r["fstatus"]),
            int(r["agistub"]),
        )
        tmd_lookup[key] = r["tmdsum"]

    # 4a. For states, precompute Census SALT shares for e18400/e18500.
    # Census state/local finance data provides the geographic
    # distribution of available (uncapped) SALT. SOI SALT is capped
    # at $10K post-TCJA and understates high-SALT states.
    # These variables are targeted as all-bins totals only — we don't
    # have a reliable per-bin source (SOI bins are cap-distorted).
    _CENSUS_SALT = {}
    if not is_cd:
        from tmd.areas.prepare.extended_targets import load_census_shares

        comb, prop = load_census_shares()
        _CENSUS_SALT = {
            "e18400": comb,  # income/sales → Census combined
            "e18500": prop,  # real estate → Census property
        }

    # 4b. For each area, compute targets by joining spec + shares
    output_dir.mkdir(parents=True, exist_ok=True)
    areas = sorted(shares["area"].unique())
    n_written = 0

    for area in areas:
        area_shares = shares[shares["area"] == area]

        # First pass: compute per-bin targets
        bin_targets = []  # (varname, count, scope, fstatus, target)
        total_specs = []  # specs for total rows (computed later)

        for _, s in spec_keys.iterrows():
            vn = s["varname"]
            cnt = int(s["count"])
            sc = int(s["scope"])
            fs = int(s["fstatus"])
            lo = float(s["agilo"])
            hi = float(s["agihi"])
            is_total = lo < -1e10 and hi > 1e10 and vn != "XTOT"

            if is_total:
                total_specs.append((vn, cnt, sc, fs, lo, hi))
                continue

            if vn == "XTOT":
                xtot_row = area_shares[area_shares["varname"] == "XTOT"]
                if xtot_row.empty:
                    continue
                target_val = xtot_row.iloc[0]["fixed_target"]
                if pd.isna(target_val):
                    continue
            else:
                match = area_shares[
                    (area_shares["varname"] == vn)
                    & (area_shares["count"] == cnt)
                    & (area_shares["fstatus"] == fs)
                    & (np.isclose(area_shares["agilo"], lo))
                    & (np.isclose(area_shares["agihi"], hi))
                ]
                if match.empty:
                    continue
                soi_share = match.iloc[0]["soi_share"]
                if pd.isna(soi_share):
                    continue

                stub = int(match.iloc[0]["agistub"])
                tmd_key = (vn, cnt, fs, stub)
                tmdsum = tmd_lookup.get(tmd_key)
                if tmdsum is None:
                    continue

                # Census SALT: use Census share for state
                # total-only e18400/e18500 targets
                if vn in _CENSUS_SALT and cnt == 0 and fs == 0:
                    c_share = _CENSUS_SALT[vn].get(area, 0)
                    target_val = tmdsum * c_share
                else:
                    target_val = tmdsum * soi_share

            bin_targets.append(
                {
                    "varname": vn,
                    "count": cnt,
                    "scope": sc,
                    "agilo": lo,
                    "agihi": hi,
                    "fstatus": fs,
                    "target": target_val,
                }
            )

        # Second pass: compute totals as sum of per-bin
        # share × tmdsum (using ALL bins from shares file,
        # not just bins in the spec — matches old pipeline)
        for vn, cnt, sc, fs, lo, hi in total_specs:
            bin_shares = area_shares[
                (area_shares["varname"] == vn)
                & (area_shares["count"] == cnt)
                & (area_shares["fstatus"] == fs)
                & (area_shares["agistub"] != 0)
            ]
            total_val = 0.0
            for _, br in bin_shares.iterrows():
                sh = br["soi_share"]
                if pd.isna(sh):
                    continue
                stub = int(br["agistub"])
                tmd_key = (vn, cnt, fs, stub)
                ts = tmd_lookup.get(tmd_key)
                if ts is not None:
                    total_val += ts * sh
            bin_targets.append(
                {
                    "varname": vn,
                    "count": cnt,
                    "scope": sc,
                    "agilo": lo,
                    "agihi": hi,
                    "fstatus": fs,
                    "target": total_val,
                }
            )

        if not bin_targets:
            continue

        tdf = pd.DataFrame(bin_targets)
        outpath = output_dir / f"{area.lower()}_targets.csv"
        tdf.to_csv(outpath, index=False)
        n_written += 1

    elapsed = time.time() - t0
    n_tgts = len(spec_keys)
    print(
        f"Wrote {n_written} target files"
        f" ({n_tgts} targets each, {elapsed:.1f}s)"
    )
    return n_written


def _parse_scope(scope):
    """Parse scope string into a list of state codes or None for all."""
    scope_lower = scope.lower().strip()
    if scope_lower in ("states", "all"):
        return None
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    return [c for c in codes if len(c) == 2 and c not in _EXCLUDE]


def _parse_cd_scope(scope):
    """Parse scope string into a list of CD codes or None for all."""
    scope_lower = scope.lower().strip()
    if scope_lower in ("cds", "all"):
        return None
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    # CD codes are >2 chars (e.g., MN01, CA52)
    return [c for c in codes if len(c) > 2]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare area target files from SOI + TMD data",
    )
    parser.add_argument(
        "--scope",
        default="states",
        help=(
            "'states', 'cds', or comma-separated area codes "
            "(e.g., 'MN,CA' or 'MN01,CA52')"
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="SOI area data year (default: 2022)",
    )
    parser.add_argument(
        "--congress",
        type=int,
        choices=(118, 119),
        default=None,
        help=(
            "Target Congressional session (118 or 119). "
            "REQUIRED for CD scope; ignored for state scope."
        ),
    )
    args = parser.parse_args()

    scope_lower = args.scope.lower().strip()
    first_code = args.scope.split(",")[0].strip()
    is_cd_scope = scope_lower == "cds" or (
        scope_lower not in ("states", "all") and len(first_code) > 2
    )
    if is_cd_scope and args.congress is None:
        parser.error("--congress is required for CD scope (choose 118 or 119)")

    prepare_targets_from_spec(
        scope=args.scope,
        area_data_year=args.year,
        congress=args.congress,
    )


if __name__ == "__main__":
    main()
