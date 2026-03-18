"""
Update expected values in test files after an intentional data change.

Usage:
    python tests/update_expected_values.py [--all] [--weights]
                                         [--taxexp] [--imputed]

Run this after `make data` when you have intentionally changed inputs
(growth rates, targets, etc.) and need to update the regression test
expected values to match the new output.

The --all flag updates all three tests. Individual flags update only
the specified test(s). With no flags, prints computed values
(from the latest `make data` output) without updating anything.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import taxcalc as tc
from tmd.imputation_assumptions import TAXYEAR, CREDIT_CLAIMING
from tmd.storage import STORAGE_FOLDER
from tmd.utils.taxcalc_utils import get_tax_expenditure_results
from tests.conftest import create_tmd_records

TESTS_FOLDER = Path(__file__).resolve().parent


def load_tmd_data():
    """Load TMD output data."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    path = STORAGE_FOLDER / "output" / "tmd.csv.gz"
    return pd.read_csv(path)


def compute_weight_fingerprint(tmd):
    """Compute weight distribution statistics."""
    wght = tmd["s006"].to_numpy()
    return {
        "n": len(wght),
        "total": wght.sum(),
        "mean": wght.mean(),
        "sdev": wght.std(),
        "min": wght.min(),
        "p25": np.percentile(wght, 25),
        "p50": np.median(wght),
        "p75": np.percentile(wght, 75),
        "max": wght.max(),
        "sum_sq": np.sum(wght**2),
    }


def compute_imputed_var_results():
    """Run Tax-Calculator simulations for OBBBA deduction benefits."""
    simyear = 2026
    output_variables = ["s006", "iitax"]
    recs = create_tmd_records(
        data_path=STORAGE_FOLDER / "output" / "tmd.csv.gz",
        weights_path=STORAGE_FOLDER / "output" / "tmd_weights.csv.gz",
        growfactors_path=(STORAGE_FOLDER / "output" / "tmd_growfactors.csv"),
    )
    pol = tc.Policy()
    pol.implement_reform(CREDIT_CLAIMING)
    baseline_sim = tc.Calculator(policy=pol, records=recs)
    baseline_sim.advance_to_year(simyear)
    baseline_sim.calc_all()
    bdf = baseline_sim.dataframe(output_variables)

    deductions = {
        "OTM": {"OvertimeIncomeDed_c": {simyear: [0, 0, 0, 0, 0]}},
        "TIP": {"TipIncomeDed_c": {simyear: 0}},
        "ALI": {"AutoLoanInterestDed_c": {simyear: 0}},
        "ALL": {
            "OvertimeIncomeDed_c": {simyear: [0, 0, 0, 0, 0]},
            "TipIncomeDed_c": {simyear: 0},
            "AutoLoanInterestDed_c": {simyear: 0},
            "SeniorDed_c": {simyear: 0},
        },
    }

    results = {}
    for ded, reform_dict in deductions.items():
        reform_policy = tc.Policy()
        reform_policy.implement_reform(CREDIT_CLAIMING)
        reform_policy.implement_reform(reform_dict)
        reform_sim = tc.Calculator(policy=reform_policy, records=recs)
        reform_sim.advance_to_year(simyear)
        reform_sim.calc_all()
        rdf = reform_sim.dataframe(output_variables)
        weight = bdf["s006"].to_numpy()
        dedben = rdf["iitax"].to_numpy() - bdf["iitax"].to_numpy()
        totben = round(((weight * dedben).sum() * 1e-9), 2)
        affected = dedben > 0
        aff_weight = (affected * weight).sum()
        affpct = round((100 * aff_weight / weight.sum()), 2)
        affben = round((affected * dedben * weight).sum() / aff_weight, 0)
        results[ded] = {
            "totben": totben,
            "affpct": affpct,
            "affben": affben,
        }
        del reform_policy, reform_sim
    return results


def update_weights(dry_run=False):
    """Update expected weight fingerprint in test_weights.py."""
    print("Computing weight fingerprint...")
    tmd = load_tmd_data()
    actual = compute_weight_fingerprint(tmd)
    key = f"tmd{TAXYEAR}"
    print(f"  Computed (from current data) {key} fingerprint:")
    for stat, val in actual.items():
        print(f"    {stat}: {val}")

    if dry_run:
        return

    # Read and update test_weights.py
    path = TESTS_FOLDER / "test_weights.py"
    text = path.read_text()

    # Build replacement dict block with formatted values
    fmt = {
        "n": str(actual["n"]),
        "total": format(actual["total"], ".1f"),
        "mean": format(actual["mean"], ".2f"),
        "sdev": format(actual["sdev"], ".2f"),
        "min": format(actual["min"], ".5f"),
        "p25": format(actual["p25"], ".4f"),
        "p50": format(actual["p50"], ".3f"),
        "p75": format(actual["p75"], ".3f"),
        "max": format(actual["max"], ".2f"),
        "sum_sq": format(actual["sum_sq"], ".1f"),
    }

    # Replace the tmdYYYY block using regex
    pattern = (
        rf'("{key}": \{{)\s*'
        + r"\s*".join(rf'"{s}": [^,]+,' for s in list(fmt.keys())[:-1])
        + r'\s*"sum_sq": [^,}]+,?\s*\}'
    )
    replacement = (
        f'"{key}": {{\n'
        + "".join(f'            "{s}": {v},\n' for s, v in fmt.items())
        + "        }"
    )
    new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    if new_text == text:
        print("  WARNING: regex did not match; no changes made")
        return
    path.write_text(new_text)
    print(f"  Updated {path}")


def update_tax_expenditures(dry_run=False):
    """Update expected tax expenditure file."""
    print("Computing tax expenditures...")
    tmd = load_tmd_data()
    weights_path = STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"
    gf_path = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"
    _ = get_tax_expenditure_results(tmd, TAXYEAR, 2023, weights_path, gf_path)
    act_path = STORAGE_FOLDER / "output" / "tax_expenditures"
    act_text = act_path.read_text()
    print(f"  Computed (from current data) tax expenditures:\n{act_text}")

    if dry_run:
        return

    exp_path = TESTS_FOLDER / f"expected_tax_exp_{TAXYEAR}_data"
    exp_path.write_text(act_text)
    print(f"  Updated {exp_path}")


def update_imputed_variables(dry_run=False):
    """Update expected OBBBA deduction benefit values."""
    print(
        "Computing (from current data) "
        "OBBBA deduction benefits (this takes a minute)..."
    )
    results = compute_imputed_var_results()
    for ded, stats in results.items():
        print(f"  {ded}: {stats}")

    if dry_run:
        return

    path = TESTS_FOLDER / "test_imputed_variables.py"
    text = path.read_text()
    for ded, stats in results.items():
        for stat, val in stats.items():
            # Find the occurrence within the correct deduction block
            ded_pattern = rf'("{ded}": \{{.*?"exp_{stat}_{TAXYEAR}": )[^,]+(,)'
            new_val = str(int(val)) if stat == "affben" else str(val)
            text = re.sub(
                ded_pattern,
                rf"\g<1>{new_val}\2",
                text,
                count=1,
                flags=re.DOTALL,
            )
    path.write_text(text)
    print(f"  Updated {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Update expected test values after data changes."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Update all expected values",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Update weight fingerprint",
    )
    parser.add_argument(
        "--taxexp",
        action="store_true",
        help="Update tax expenditure expected values",
    )
    parser.add_argument(
        "--imputed",
        action="store_true",
        help="Update OBBBA deduction benefit expected values",
    )
    args = parser.parse_args()

    do_all = args.all
    dry_run = not (args.all or args.weights or args.taxexp or args.imputed)

    if dry_run:
        print(
            "No update flags specified. "
            "Showing computed values only (dry run).\n"
            "Use --all or --weights/--taxexp/--imputed to update.\n"
        )

    if do_all or args.weights or dry_run:
        update_weights(dry_run=dry_run)
    if do_all or args.taxexp or dry_run:
        update_tax_expenditures(dry_run=dry_run)
    if do_all or args.imputed or dry_run:
        update_imputed_variables(dry_run=dry_run)

    if not dry_run:
        print("\nDone. Run `make test` to verify tests pass.")


if __name__ == "__main__":
    main()
