"""Small randomized searches with calendar CV and fold-specific target thresholds."""

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.features.create_target import classify_cases, fit_thresholds
from src.models.train import encode_labels, fit_model
from src.utils.helpers import DataValidationError


class QuantileClassifier(ClassifierMixin, BaseEstimator):
    """Fit both threshold definition and model inside each chronological CV fold."""

    def __init__(self, pipeline: Pipeline, low_quantile: float = 0.5, medium_quantile: float = 0.8):
        self.pipeline = pipeline
        self.low_quantile = low_quantile
        self.medium_quantile = medium_quantile

    def fit(self, features: pd.DataFrame, cases: pd.Series) -> "QuantileClassifier":
        """Use only fold training counts to establish classes and fit preprocessing."""
        values = pd.Series(cases, index=features.index)
        self.thresholds_ = fit_thresholds(values, self.low_quantile, self.medium_quantile)
        labels = classify_cases(values, self.thresholds_)
        self.pipeline_ = fit_model(clone(self.pipeline), features, labels)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, features: pd.DataFrame):
        """Predict encoded risk classes."""
        return self.pipeline_.predict(features)

def fold_macro_f1(estimator: QuantileClassifier, features: pd.DataFrame, cases: pd.Series) -> float:
    """Evaluate validation counts with that fold's frozen training thresholds."""
    truth = encode_labels(classify_cases(pd.Series(cases), estimator.thresholds_))
    return float(
        f1_score(
            truth, estimator.predict(features), labels=[0, 1, 2], average="macro", zero_division=0
        )
    )

def tune_model(
    pipeline: Pipeline,
    name: str,
    training: pd.DataFrame,
    columns: list[str],
    config: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune exclusively inside training; districts from the same week stay together."""
    spaces = {
        "logistic_regression": {"classifier__C": [0.1, 1, 5]},
        "random_forest": {
            "classifier__max_depth": [6, 10, 14],
            "classifier__min_samples_leaf": [2, 4, 8],
        },
        "xgboost": {
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.04, 0.08, 0.12],
        },
    }
    ordered = training.sort_values(["start_date", "district"]).reset_index(drop=True)
    dates = sorted(ordered.start_date.unique())
    cv = []
    for past, future in TimeSeriesSplit(n_splits=config["tuning"]["folds"]).split(dates):
        boundary = pd.Timestamp(dates[future[0]])
        train_indices = ordered.index[
            ordered.start_date.isin([dates[index] for index in past])
            & ordered.target_date.lt(boundary)
        ].to_numpy()
        validation_indices = ordered.index[
            ordered.start_date.isin([dates[index] for index in future])
        ].to_numpy()
        if len(train_indices) == 0:
            raise DataValidationError("Insufficient training weeks for chronological tuning")
        cv.append((train_indices, validation_indices))
    quantiles = config["target_quantiles"]
    wrapped = QuantileClassifier(pipeline, quantiles["low"], quantiles["medium"])
    search = RandomizedSearchCV(
        wrapped,
        {f"pipeline__{key}": value for key, value in spaces[name].items()},
        n_iter=config["tuning"]["iterations"],
        cv=cv,
        scoring=fold_macro_f1,
        random_state=config["random_state"],
        n_jobs=1,
        error_score="raise",
    )
    search.fit(ordered[columns], ordered.next_week_cases)
    return search.best_estimator_.pipeline_, {
        "best_parameters": search.best_params_,
        "cv_macro_f1": float(search.best_score_),
        "folds": len(cv),
        "thresholds": "fitted separately inside each fold",
    }
