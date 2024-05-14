import torch
from torch.optim import Adam
import numpy as np
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tax_microdata_benchmarking.storage import STORAGE_FOLDER

warnings.filterwarnings("ignore")

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
    weight_deviation_penalty: float = 0,
):
    targets = pd.read_csv(STORAGE_FOLDER / "input" / "agi_targets.csv")

    if time_period not in targets.year.unique():
        raise ValueError(f"Year {time_period} not in targets.")

    def build_loss_matrix(df):
        loss_matrix = pd.DataFrame()
        agi = df.c00100
        taxable = df.c09200 - df.refund > 0
        targets_array = []
        for i in range(len(INCOME_RANGES) - 1):
            mask = (
                (agi.values >= INCOME_RANGES[i])
                * (agi.values < INCOME_RANGES[i + 1])
                * taxable
            )
            loss_matrix[
                f"Total AGI {fmt(INCOME_RANGES[i])}-{fmt(INCOME_RANGES[i + 1])}"
            ] = (mask * agi)
            agi_target = targets[targets.table == "tab11"][
                targets.year == time_period
            ][targets.vname.isin(["agi"])][targets.datatype == "taxable"][
                targets.incsort - 2 == i
            ].ptarget
            targets_array.append(agi_target.iloc[0] * 1e3)
            nret_target = targets[targets.table == "tab11"][
                targets.year == time_period
            ][targets.vname.isin(["nret_all"])][targets.datatype == "taxable"][
                targets.incsort - 2 == i
            ].ptarget
            loss_matrix[
                f"Returns {fmt(INCOME_RANGES[i])}-{fmt(INCOME_RANGES[i + 1])}"
            ] = mask.astype(np.float32)
            targets_array.append(nret_target.iloc[0])
        return loss_matrix, np.array(targets_array)

    weights = torch.tensor(
        flat_file.s006.values, dtype=torch.float32, requires_grad=True
    )
    original_weights = weights.clone()
    output_matrix, target_array = build_loss_matrix(flat_file)
    output_matrix_tensor = torch.tensor(
        output_matrix.values, dtype=torch.float32
    )
    target_array = torch.tensor(target_array, dtype=torch.float32)

    outputs = (weights * output_matrix_tensor.T).sum(axis=1)
    outputs, target_array

    optimizer = Adam([weights], lr=1e0)

    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from datetime import datetime

    writer = SummaryWriter(
        log_dir=STORAGE_FOLDER
        / "output"
        / "reweighting"
        / f"{time_period}_{datetime.now().isoformat()}"
    )

    for i in list(range(10_000)):
        optimizer.zero_grad()
        outputs = (weights * output_matrix_tensor.T).sum(axis=1)
        weight_deviation = (
            (weights - original_weights).abs().sum()
            / original_weights.sum()
            * weight_deviation_penalty
        )
        loss_value = (
            (outputs / target_array - 1) ** 2
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
                    f"Relative error/{metric_name}", rel_error, i
                )

            writer.add_scalar(
                "Summary/Max relative error",
                (outputs / target_array - 1).abs().max(),
                i,
            )
            writer.add_scalar(
                "Summary/Mean relative error",
                (outputs / target_array - 1).abs().mean(),
                i,
            )

    flat_file["s006"] = weights.detach().numpy()
    return flat_file
