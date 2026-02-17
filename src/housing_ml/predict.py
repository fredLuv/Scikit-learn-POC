from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def _prepare_frame(records: list[dict], cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for col in cols:
        if col not in frame.columns:
            frame[col] = None
    return frame[cols]


def predict_from_records(model_path: str | Path, records: list[dict]) -> list[float]:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    cols = artifact["feature_columns"]

    frame = _prepare_frame(records, cols)

    preds = model.predict(frame)
    return [float(v) for v in preds]


def predict_from_json(
    model_path: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    include_intervals: bool = False,
) -> None:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of house records.")

    artifact = joblib.load(model_path)
    cols = artifact["feature_columns"]
    frame = _prepare_frame(payload, cols)
    preds = [float(v) for v in artifact["model"].predict(frame)]

    output: dict[str, object] = {"predictions": preds}
    if include_intervals:
        interval_meta = artifact.get("prediction_interval")
        interval_model = artifact.get("interval_model")
        if interval_meta is not None and interval_model is not None:
            q = float(interval_meta["quantile_abs_error"])
            coverage = float(interval_meta["coverage_target"])
            interval_preds = [float(v) for v in interval_model.predict(frame)]
            output["prediction_intervals"] = [
                {
                    "prediction": p,
                    "lower": p - q,
                    "upper": p + q,
                    "coverage_target": coverage,
                }
                for p in interval_preds
            ]

    Path(output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
