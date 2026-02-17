# Scikit-learn POC (Advanced Refresh)

A housing-price ML project upgraded from a basic regression demo into a practical investor-facing decision support workflow.

## Business insights for investors

This model is best used for ranking and risk-filtering deals, not for fully autonomous pricing.

| Business question | Data signal | Investor implication |
| --- | --- | --- |
| Where is pricing most reliable? | Mid-price bands show much lower MAE than the top price band. | Prioritize capital in segments with lower model error and cleaner spread capture. |
| Where is geographic risk concentrated? | Error slices by city are uneven. | Apply city-level risk premiums and stricter filters in high-error locations. |
| Is model-only pricing safe? | Fit is solid (R² ~0.70), but tail errors are still meaningful (high P90 absolute error). | Use model-first screening, then human underwriting on final bids. |

Practical strategy:
1. Use confidence-weighted rules: faster approvals in mid-band, larger margin-of-safety in high-end properties.
2. Segment models by market: luxury vs non-luxury and major city clusters.
3. Tie uncertainty to risk: use prediction intervals for bid caps and position sizing.

## Model stack and findings (latest run)

### Models used

All models share one preprocessing pipeline:
- numeric: median imputation + standard scaling
- categorical: most-frequent imputation + one-hot encoding

Candidate regressors:

| Model | Why included |
| --- | --- |
| Ridge | fast linear baseline for sanity checking |
| RandomForestRegressor | robust non-linear tabular baseline |
| HistGradientBoostingRegressor (+ random search) | high-capacity boosted model with tuning |

Best model is selected by lowest test MAE.

### Performance leaderboard

From `outputs/training_report.json`:

| Model | MAE | RMSE | R² | Train time (s) |
| --- | ---:| ---:| ---:| ---:|
| random_forest | 60,036.08 | 171,548.49 | 0.7038 | 7.87 |
| hist_gbr | 64,466.26 | 177,116.10 | 0.6842 | 33.74 |
| ridge | 93,924.43 | 209,997.10 | 0.5561 | 0.05 |

### Diagnostics snapshot

From `outputs/error_diagnostics.json`:

| Metric | Value |
| --- | ---:|
| Test size | 8,541 |
| Overall MAE | 60,036.08 |
| P90 absolute error | 122,647.66 |

Price-band error profile:

| Price band | Count | MAE |
| --- | ---:| ---:|
| (243,180, 337,679] | 1,709 | 35,975.14 |
| (337,679, 422,097] | 1,709 | 36,905.57 |
| (422,097, 529,199] | 1,708 | 43,529.30 |
| (529,199, 10,836,004] | 1,706 | 130,740.88 |
| (627, 243,180] | 1,709 | 53,143.97 |

## Why this is better (short version)

- Cleaner ML engineering: leakage-safe pipeline with robust missing/unknown handling.
- Better decision quality: model comparison + calibrated prediction intervals.
- Better risk visibility: error diagnostics by city and price segment.

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
