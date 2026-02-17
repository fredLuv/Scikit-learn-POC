from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "sale_price"
DROP_COLS = ["house_number", "unit_number", "street_name", "zip_code"]


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COL}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    for col in DROP_COLS:
        if col in work.columns:
            del work[col]

    y = work[TARGET_COL]
    x = work.drop(columns=[TARGET_COL])
    return x, y


def make_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> DatasetSplit:
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return DatasetSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
