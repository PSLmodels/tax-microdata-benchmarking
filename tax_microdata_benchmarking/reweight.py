import pandas as pd
from pathlib import Path

FOLDER = Path(__file__).parent


def reweight(df, time_period: int):
    if time_period not in [2021]:
        raise ValueError(
            f"Only years 2021 are supported. Received {time_period}."
        )
    targets = pd.read_csv(FOLDER / "agi_targets.csv")
