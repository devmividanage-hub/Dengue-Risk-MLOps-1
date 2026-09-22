"""CPU-friendly estimators with preprocessing fitted solely during model.fit."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.utils.helpers import RISK_CLASSES, DataValidationError

def encode_labels(labels: pd.Series) -> np.ndarray:
    """Use a fixed, persisted LOW/MEDIUM/HIGH integer order for all estimators."""
    encoded = labels.map({label: index for index, label in enumerate(RISK_CLASSES)})
    if encoded.isna().any():
        raise DataValidationError("Training labels must be LOW, MEDIUM, or HIGH.")
    return encoded.to_numpy(dtype=int)

def decode_labels(encoded: np.ndarray) -> np.ndarray:
    """Decode the documented estimator class order."""
    return np.asarray(RISK_CLASSES)[np.asarray(encoded, dtype=int)]


def build_model(name: str, columns: list[str], config: dict[str, Any]) -> Pipeline:
    """Construct an unfitted sklearn Pipeline, scaling only logistic regression."""
    settings = dict(config["models"][name])
    seed = config["random_state"]
    if name == "logistic_regression":
        estimator = LogisticRegression(class_weight="balanced", random_state=seed, **settings)
    elif name == "random_forest":
        estimator = RandomForestClassifier(class_weight="balanced", random_state=seed, **settings)
    elif name == "xgboost":
        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            **settings,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(numeric_steps),
                [column for column in columns if column != "district"],
            ),
            ("categorical", categorical, ["district"]),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def fit_model(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    """Fit with training-only class balancing and require all three risk classes."""
    encoded = encode_labels(labels)
    if len(np.unique(encoded)) != 3:
        raise DataValidationError("Training period must contain all three risk classes")
    fit_parameters = {}
    if isinstance(model.named_steps["classifier"], XGBClassifier):
        fit_parameters["classifier__sample_weight"] = compute_sample_weight("balanced", encoded)
    model.fit(features, encoded, **fit_parameters)
    return model