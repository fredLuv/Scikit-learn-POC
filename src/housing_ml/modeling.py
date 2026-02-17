from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelMetrics:
    model_name: str
    mae: float
    rmse: float
    r2: float
    train_seconds: float


def build_preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def make_candidate_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gbr": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=42,
                        max_iter=500,
                        learning_rate=0.05,
                    ),
                ),
            ]
        ),
    }


def tune_hist_gbr(base_pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    param_dist = {
        "model__max_depth": [4, 6, 8, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 30],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__max_bins": [255],
    }

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=0,
        n_jobs=1,
        random_state=42,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def evaluate(
    model_name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    train_seconds: float,
) -> ModelMetrics:
    preds = model.predict(x_test)
    return ModelMetrics(
        model_name=model_name,
        mae=float(mean_absolute_error(y_test, preds)),
        rmse=float(root_mean_squared_error(y_test, preds)),
        r2=float(r2_score(y_test, preds)),
        train_seconds=train_seconds,
    )
