#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from housing_ml.predict import predict_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house prices from JSON records")
    parser.add_argument("--model", default="outputs/housing_model_bundle.joblib")
    parser.add_argument("--input", default="configs/sample_houses.json")
    parser.add_argument("--output", default="outputs/predictions.json")
    parser.add_argument(
        "--with-intervals",
        action="store_true",
        help="Include calibrated prediction intervals in output JSON",
    )
    args = parser.parse_args()

    predict_from_json(
        Path(args.model),
        Path(args.input),
        Path(args.output),
        include_intervals=args.with_intervals,
    )
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
