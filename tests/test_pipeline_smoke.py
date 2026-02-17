from __future__ import annotations

import json
import unittest
from pathlib import Path

from housing_ml.predict import predict_from_json
from housing_ml.train import train_and_save
from housing_ml.predict import predict_from_records


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.data_csv = cls.root / "Codes" / "ml_house_data_set.csv"
        cls.model_out = cls.root / "outputs" / "_test_model.joblib"
        cls.report_out = cls.root / "outputs" / "_test_report.json"
        cls.diagnostics_out = cls.root / "outputs" / "_test_diagnostics.json"

        train_and_save(
            cls.data_csv,
            cls.model_out,
            cls.report_out,
            diagnostics_out=cls.diagnostics_out,
        )

    def test_train_and_predict_smoke(self) -> None:
        report = train_and_save(
            self.data_csv,
            self.model_out,
            self.report_out,
            diagnostics_out=self.diagnostics_out,
        )
        self.assertTrue(self.model_out.exists())
        self.assertTrue(self.report_out.exists())
        self.assertTrue(self.diagnostics_out.exists())
        self.assertTrue(len(report.metrics) >= 1)
        self.assertTrue(all(metric.train_seconds >= 0.0 for metric in report.metrics))

        preds = predict_from_records(
            self.model_out,
            [
                {
                    "year_built": 2000,
                    "stories": 2,
                    "num_bedrooms": 4,
                    "full_bathrooms": 3,
                    "half_bathrooms": 1,
                    "livable_sqft": 2500,
                    "total_sqft": 3000,
                    "garage_type": "attached",
                    "garage_sqft": 420,
                    "carport_sqft": 0,
                    "has_fireplace": True,
                    "has_pool": False,
                    "has_central_heating": True,
                    "has_central_cooling": True,
                    "city": "Brownport",
                }
            ],
        )
        self.assertEqual(len(preds), 1)
        self.assertGreater(preds[0], 0.0)

    def test_prediction_intervals_smoke(self) -> None:
        input_json = self.root / "configs" / "sample_houses.json"
        output_json = self.root / "outputs" / "_test_predictions_with_intervals.json"

        predict_from_json(self.model_out, input_json, output_json, include_intervals=True)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertIn("predictions", payload)
        self.assertIn("prediction_intervals", payload)
        self.assertEqual(len(payload["predictions"]), len(payload["prediction_intervals"]))


if __name__ == "__main__":
    unittest.main()
