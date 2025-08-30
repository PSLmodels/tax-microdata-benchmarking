"""
This module provides utilities for reweighting a flat file to match AGI targets.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tmd.storage import STORAGE_FOLDER
from tmd.utils.soi_replication import tc_to_soi
from tmd.imputation_assumptions import (
    REWEIGHT_MULTIPLIER_MIN,
    REWEIGHT_MULTIPLIER_MAX,
    REWEIGHT_DEVIATION_PENALTY,
)


INCOME_RANGES = [
    -np.inf,
    1,
    5e3,
    1e4,
    1.5e4,
    2e4,
    2.5e4,
    3e4,
    4e4,
    5e4,
    7.5e4,
    1e5,
    2e5,
    5e5,
    1e6,
    1.5e6,
    2e6,
    5e6,
    1e7,
    np.inf,
]


def fmt(x):
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    if x < 1e3:
        return f"{x:.0f}"
    if x < 1e6:
        return f"{x/1e3:.0f}k"
    if x < 1e9:
        return f"{x/1e6:.0f}m"
    return f"{x/1e9:.1f}bn"


def reweight(
    flat_file: pd.DataFrame,
    time_period: int = 2021,
    weight_multiplier_min: float = REWEIGHT_MULTIPLIER_MIN,
    weight_multiplier_max: float = REWEIGHT_MULTIPLIER_MAX,
    weight_deviation_penalty: float = REWEIGHT_DEVIATION_PENALTY,
    use_gpu: bool = True,
):
    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")

    if time_period not in targets.Year.unique():
        raise ValueError(f"Year {time_period} not in targets.")
    print(f"...reweighting for year {time_period}")

    def build_loss_matrix(df):
        loss_matrix = pd.DataFrame()
        df = tc_to_soi(df, time_period)
        agi = df["adjusted_gross_income"].values
        filer = df["is_tax_filer"].values
        targets_array = []
        soi_subset = targets
        soi_subset = soi_subset[soi_subset.Year == time_period]
        agi_level_targeted_variables = [
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
        aggregate_level_targeted_variables = [
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
        aggregate_level_targeted_variables = [
            variable
            for variable in aggregate_level_targeted_variables
            if variable in df.columns
        ]
        soi_subset = soi_subset[
            soi_subset.Variable.isin(agi_level_targeted_variables)
            & (
                (soi_subset["AGI lower bound"] != -np.inf)
                | (soi_subset["AGI upper bound"] != np.inf)
            )
            | (
                soi_subset.Variable.isin(aggregate_level_targeted_variables)
                & (soi_subset["AGI lower bound"] == -np.inf)
                & (soi_subset["AGI upper bound"] == np.inf)
            )
        ]
        for _, row in soi_subset.iterrows():
            if row["Taxable only"]:
                continue  # exclude "taxable returns" statistics

            mask = (
                (agi >= row["AGI lower bound"])
                * (agi < row["AGI upper bound"])
                * filer
            ) > 0

            if row["Filing status"] == "Single":
                mask *= df["filing_status"].values == "SINGLE"
            elif (
                row["Filing status"]
                == "Married Filing Jointly/Surviving Spouse"
            ):
                mask *= df["filing_status"].values == "JOINT"
            elif row["Filing status"] == "Head of Household":
                mask *= df["filing_status"].values == "HEAD_OF_HOUSEHOLD"
            elif row["Filing status"] == "Married Filing Separately":
                mask *= df["filing_status"].values == "SEPARATE"

            values = df[row["Variable"]].values

            if row["Count"]:
                values = (values > 0).astype(float)

            agi_range_label = (
                f"{fmt(row['AGI lower bound'])}-{fmt(row['AGI upper bound'])}"
            )
            taxable_label = (
                "taxable" if row["Taxable only"] else "all" + " returns"
            )
            filing_status_label = row["Filing status"]

            variable_label = row["Variable"].replace("_", " ")

            if row["Count"] and not row["Variable"] == "count":
                label = (
                    f"{variable_label}/count/AGI in "
                    f"{agi_range_label}/{taxable_label}/{filing_status_label}"
                )
            elif row["Variable"] == "count":
                label = (
                    f"{variable_label}/count/AGI in "
                    f"{agi_range_label}/{taxable_label}/{filing_status_label}"
                )
            else:
                label = (
                    f"{variable_label}/total/AGI in "
                    f"{agi_range_label}/{taxable_label}/{filing_status_label}"
                )

            if label not in loss_matrix.columns:
                loss_matrix[label] = mask * values
                targets_array.append(row["Value"])

        return loss_matrix.copy(), np.array(targets_array)

    # GPU Detection and Device Selection
    gpu_available = torch.cuda.is_available()
    use_gpu_actual = use_gpu and gpu_available

    device = torch.device("cuda" if use_gpu_actual else "cpu")

    if use_gpu_actual:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"...GPU acceleration enabled: {gpu_name} ({gpu_memory:.1f} GB)")
    elif use_gpu and not gpu_available:
        print("...GPU requested but not available, using CPU")
    elif not use_gpu and gpu_available:
        print("...GPU available but disabled by user, using CPU")
    else:
        print("...GPU not available, using CPU")

    rng_seed = 65748392
    torch.manual_seed(rng_seed)  # set the random number seed for CPU
    torch.cuda.manual_seed_all(rng_seed)  # set the seed for all GPUs

    # Create tensors directly on the selected device to avoid non-leaf tensor issues
    weights = torch.tensor(
        flat_file.s006.values, dtype=torch.float32, device=device
    )
    weight_multiplier = torch.tensor(
        np.ones_like(flat_file.s006.values),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    original_weights = weights.clone()
    output_matrix, target_array = build_loss_matrix(flat_file)

    print(f"Targeting {len(target_array)} SOI statistics")
    # print out non-numeric columns
    for col in output_matrix.columns:
        try:
            torch.tensor(output_matrix[col].values, dtype=torch.float32)
        except ValueError:
            print(f"Column {col} is not numeric")
    output_matrix_tensor = torch.tensor(
        output_matrix.values, dtype=torch.float32, device=device
    )
    target_array = torch.tensor(
        target_array, dtype=torch.float32, device=device
    )

    outputs = (weights * output_matrix_tensor.T).sum(axis=1)
    original_loss_value = (((outputs + 1) / (target_array + 1) - 1) ** 2).sum()

    # First, check for NaN columns and print out the labels

    for i in range(len(target_array)):
        if torch.isnan(outputs[i]).any():
            print(f"Column {output_matrix.columns[i]} has NaN values")
        if target_array[i] == 0:
            pass  # print(f"Column {output_matrix.columns[i]} has target 0")

    optimizer = torch.optim.Adam([weight_multiplier], lr=1e-1)

    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from datetime import datetime

    writer = SummaryWriter(
        log_dir=STORAGE_FOLDER
        / "output"
        / "reweighting"
        / f"{time_period}_{datetime.now().isoformat()}"
    )

    for i in tqdm(range(2_000), desc="Optimising weights"):
        optimizer.zero_grad()
        new_weights = weights * (
            torch.clamp(
                weight_multiplier,
                min=weight_multiplier_min,
                max=weight_multiplier_max,
            )
        )
        outputs = (new_weights * output_matrix_tensor.T).sum(axis=1)
        weight_deviation = (
            (new_weights - original_weights).abs().sum()
            / original_weights.sum()
            * weight_deviation_penalty
            * original_loss_value
        )
        loss_value = (
            ((outputs + 1) / (target_array + 1) - 1) ** 2
        ).sum() + weight_deviation
        loss_value.backward()
        optimizer.step()
        if i % 100 == 0:
            writer.add_scalar("Summary/Loss", loss_value, i)
            for j in range(len(target_array)):
                metric_name = output_matrix.columns[j]
                total_projection = outputs[j]
                rel_error = (
                    total_projection - target_array[j]
                ) / target_array[j]
                writer.add_scalar(
                    f"Estimate/{metric_name}", total_projection, i
                )
                writer.add_scalar(f"Target/{metric_name}", target_array[j], i)
                writer.add_scalar(
                    f"Absolute relative error/{metric_name}", abs(rel_error), i
                )

            writer.add_scalar(
                "Summary/Max relative error",
                ((outputs + 1) / (target_array + 1) - 1).abs().max(),
                i,
            )
            writer.add_scalar(
                "Summary/Mean relative error",
                ((outputs + 1) / (target_array + 1) - 1).abs().mean(),
                i,
            )

    print("...reweighting finished")

    # Move final weights back to CPU for numpy conversion
    final_weights = new_weights.detach().cpu().numpy()
    flat_file["s006"] = final_weights
    return flat_file
