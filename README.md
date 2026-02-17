# Scikit-learn POC (Advanced Refresh)

This repo started as a basic house-price regression demo. It is now upgraded to a modern sklearn workflow with:

- robust preprocessing pipeline (`ColumnTransformer` + imputers + one-hot)
- multi-model comparison (`Ridge`, `RandomForest`, `HistGradientBoosting`)
- hyperparameter tuning (`RandomizedSearchCV`)
- model training-time leaderboard for speed/quality trade-offs
- calibrated prediction intervals (conformal-style residual quantile)
- error-slice diagnostics by city and price band
- reproducible model bundle saving (`joblib`)
- batch prediction from JSON inputs
- explainability via permutation importance

## Project structure

- `src/housing_ml/` core package
- `scripts/` runnable entry points
- `configs/` sample prediction inputs
- `outputs/` generated artifacts
- `Codes/` original dataset and legacy scripts

## Setup

```bash
cd Scikit-learn-POC
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Train advanced model

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/train_advanced.py
```

Outputs:

- `outputs/housing_model_bundle.joblib`
- `outputs/training_report.json`
- `outputs/error_diagnostics.json`

## Predict from JSON

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/predict_advanced.py \
  --model outputs/housing_model_bundle.joblib \
  --input configs/sample_houses.json \
  --output outputs/predictions.json \
  --with-intervals
```

## Explain model (permutation importance)

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/explain_advanced.py
```

Output:

- `outputs/feature_importance.json`
- `outputs/predictions.json` can now include:
  - point prediction (`predictions`)
  - interval bounds (`prediction_intervals`)

## Why this is stronger than the original POC

- safer handling of missing values and unknown categories
- less leakage-prone preprocessing (all inside model pipeline)
- benchmark-style model comparison before selecting best estimator
- uncertainty-aware outputs for production-style decision support
- segment-level diagnostics for model risk visibility
- reproducible artifacts for downstream serving/inference

## Model stack and findings (latest run)

### Models used

All models share the same preprocessing pipeline:

- numeric features: median imputation + standard scaling
- categorical features: most-frequent imputation + one-hot encoding

Candidate regressors:

| Model | Why included |
| --- | --- |
| Ridge | fast linear baseline, good for sanity checks |
| RandomForestRegressor | strong non-linear baseline, robust on mixed tabular data |
| HistGradientBoostingRegressor (+ random search) | high-capacity boosting model with tuning |

Best model selected by lowest test MAE.

### Performance leaderboard

From `outputs/training_report.json`:

| Model | MAE | RMSE | R² | Train time (s) |
| --- | ---:| ---:| ---:| ---:|
| random_forest | 60,036.08 | 171,548.49 | 0.7038 | 7.87 |
| hist_gbr | 64,466.26 | 177,116.10 | 0.6842 | 33.74 |
| ridge | 93,924.43 | 209,997.10 | 0.5561 | 0.05 |

Interpretation:
- `random_forest` currently gives the best error/fit tradeoff on this dataset.
- `hist_gbr` is competitive but slower in this configuration.
- `ridge` is much faster but less accurate; useful as a control baseline.

### Data insights from diagnostics

From `outputs/error_diagnostics.json`:

| Metric | Value |
| --- | ---:|
| Test size | 8,541 |
| Overall MAE | 60,036.08 |
| P90 absolute error | 122,647.66 |

Price-band insight:

| Price band | Count | MAE |
| --- | ---:| ---:|
| (243,180, 337,679] | 1,709 | 35,975.14 |
| (337,679, 422,097] | 1,709 | 36,905.57 |
| (422,097, 529,199] | 1,708 | 43,529.30 |
| (529,199, 10,836,004] | 1,706 | 130,740.88 |
| (627, 243,180] | 1,709 | 53,143.97 |

Key takeaway:
- error is highest in the top price band, which is common in housing data because luxury properties are rarer and more heterogeneous.
- city-level slices also show uneven error concentration, helping prioritize where to collect more data or train segment-specific models.

## Business insights for investors

This model is best used as an investment decision-support layer, not a replacement for underwriting.

### What the housing data suggests

| Business question | Data signal | Investor implication |
| --- | --- | --- |
| Where is pricing most reliable? | Mid-price bands have materially lower MAE than the highest price band. | Prioritize capital in segments where prediction error is lower and expected spread capture is cleaner. |
| Where is geographic risk concentrated? | City-level error slices are uneven (some cities have much higher MAE). | Use location-based risk premiums and tighter filters in high-error cities. |
| Can this be used for autonomous pricing? | Overall fit is solid (R² ~0.70), but tail errors are still large (high P90 absolute error). | Use model for deal ranking/screening first, then human review for final bid decisions. |

### Practical strategy from this POC

1. Deploy confidence-weighted decision rules:
mid-band inventory gets faster approval paths; high-end assets require larger margin-of-safety.
2. Add market segmentation:
train separate models for luxury vs non-luxury and for major city clusters.
3. Use uncertainty in risk controls:
prediction intervals should drive position sizing and max bid caps.
