from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import load_dataset, make_split


def permutation_importance_report(
    model_path: str | Path,
    data_csv: str | Path,
    output_json: str | Path,
    top_k: int = 20,
) -> None:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    columns = artifact["feature_columns"]

    df = load_dataset(data_csv)
    split = make_split(df)
    x_test = split.x_test.copy()
    y_test = split.y_test

    for col in columns:
        if col not in x_test.columns:
            x_test[col] = None
    x_test = x_test[columns]

    imp = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
        scoring="neg_mean_absolute_error",
    )

    rows = []
    for idx, col in enumerate(columns):
        rows.append(
            {
                "feature": col,
                "importance_mean": float(imp.importances_mean[idx]),
                "importance_std": float(imp.importances_std[idx]),
            }
        )

    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    Path(output_json).write_text(json.dumps({"top_features": rows[:top_k]}, indent=2), encoding="utf-8")
