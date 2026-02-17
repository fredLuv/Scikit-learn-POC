#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from housing_ml.train import train_and_save


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_csv = root / "Codes" / "ml_house_data_set.csv"
    model_out = root / "outputs" / "housing_model_bundle.joblib"
    report_out = root / "outputs" / "training_report.json"
    diagnostics_out = root / "outputs" / "error_diagnostics.json"

    report = train_and_save(data_csv, model_out, report_out, diagnostics_out=diagnostics_out)

    print(f"Best model: {report.best_model}")
    for metric in report.metrics:
        print(
            f"{metric.model_name:<14} MAE={metric.mae:,.2f} RMSE={metric.rmse:,.2f} "
            f"R2={metric.r2:.4f} TrainSec={metric.train_seconds:.2f}"
        )
    print(f"Saved model:  {model_out}")
    print(f"Saved report: {report_out}")
    print(f"Saved diagnostics: {diagnostics_out}")


if __name__ == "__main__":
    main()
