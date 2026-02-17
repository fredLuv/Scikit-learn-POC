from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .data import load_dataset, make_split
from .modeling import (
    ModelMetrics,
    build_preprocessor,
    evaluate,
    make_candidate_pipelines,
    tune_hist_gbr,
)


@dataclass(frozen=True)
class TrainingReport:
    best_model: str
    metrics: list[ModelMetrics]


def _fit_interval_model(
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 0.1,
) -> tuple[object, dict[str, float | int]]:
    x_fit, x_calib, y_fit, y_calib = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=101,
    )
    interval_model = clone(model).fit(x_fit, y_fit)
    calib_preds = interval_model.predict(x_calib)
    abs_errors = np.abs(y_calib.to_numpy() - calib_preds)
    q = float(np.quantile(abs_errors, 1.0 - alpha))
    interval_meta = {
        "alpha": alpha,
        "coverage_target": 1.0 - alpha,
        "quantile_abs_error": q,
        "calibration_size": int(len(x_calib)),
    }
    return interval_model, interval_meta


def _make_error_slices(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    preds = model.predict(x_test)
    abs_err = np.abs(y_test.to_numpy() - preds)

    diagnostics: dict[str, object] = {
        "overall": {
            "test_size": int(len(x_test)),
            "mean_absolute_error": float(abs_err.mean()),
            "p90_absolute_error": float(np.quantile(abs_err, 0.90)),
        }
    }

    if "city" in x_test.columns:
        city_df = pd.DataFrame({"city": x_test["city"].astype(str), "abs_err": abs_err})
        city_stats = (
            city_df.groupby("city", as_index=False)
            .agg(count=("abs_err", "size"), mae=("abs_err", "mean"))
            .sort_values(by="mae", ascending=False)
        )
        diagnostics["city_error_slices"] = city_stats.head(12).to_dict(orient="records")

    price_bins = pd.qcut(y_test, q=5, duplicates="drop")
    band_df = pd.DataFrame({"price_band": price_bins.astype(str), "abs_err": abs_err})
    band_stats = (
        band_df.groupby("price_band", as_index=False)
        .agg(count=("abs_err", "size"), mae=("abs_err", "mean"))
        .sort_values(by="price_band")
    )
    diagnostics["price_band_error_slices"] = band_stats.to_dict(orient="records")

    return diagnostics


def train_and_save(
    data_csv: str | Path,
    model_out: str | Path,
    report_out: str | Path,
    diagnostics_out: str | Path | None = None,
) -> TrainingReport:
    df = load_dataset(data_csv)
    split = make_split(df)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(split.x_train)

    candidates = make_candidate_pipelines(preprocessor)

    trained: dict[str, object] = {}
    train_seconds: dict[str, float] = {}
    for name, pipeline in candidates.items():
        t0 = perf_counter()
        if name == "hist_gbr":
            trained[name] = tune_hist_gbr(pipeline, split.x_train, split.y_train)
        else:
            trained[name] = pipeline.fit(split.x_train, split.y_train)
        train_seconds[name] = perf_counter() - t0

    metrics = [
        evaluate(
            name,
            model,
            split.x_test,
            split.y_test,
            train_seconds=train_seconds[name],
        )
        for name, model in trained.items()
    ]

    metrics.sort(key=lambda m: m.mae)
    best = metrics[0]

    best_model = trained[best.model_name]
    interval_model, interval_meta = _fit_interval_model(best_model, split.x_train, split.y_train)
    diagnostics = _make_error_slices(best_model, split.x_test, split.y_test)
    artifact = {
        "model": best_model,
        "interval_model": interval_model,
        "prediction_interval": interval_meta,
        "feature_columns": list(split.x_train.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": "sale_price",
    }

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    report = TrainingReport(best_model=best.model_name, metrics=metrics)
    report_path = Path(report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "best_model": report.best_model,
                "metrics": [asdict(m) for m in report.metrics],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if diagnostics_out is not None:
        diagnostics_path = Path(diagnostics_out)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return report
