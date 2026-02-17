#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from housing_ml.explain import permutation_importance_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model = root / "outputs" / "housing_model_bundle.joblib"
    data = root / "Codes" / "ml_house_data_set.csv"
    out = root / "outputs" / "feature_importance.json"

    permutation_importance_report(model, data, out, top_k=25)
    print(f"Wrote feature importance report to {out}")


if __name__ == "__main__":
    main()
