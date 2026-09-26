"""Small model fixtures are synthetic and exist strictly inside unit tests."""

import joblib
import pandas as pd
import pytest

from pipelines.training_pipeline import create_feature_schema
from src.models.train import build_model, fit_model
from src.utils.helpers import RISK_CLASSES


@pytest.fixture
def tiny_model(tmp_path):
    """Fit a tiny real pipeline and persist a test-only coherent artifact bundle."""
    features = pd.DataFrame(
        {"district": ["Colombo", "Kandy"] * 15, "cases": list(range(30)), "month": [1] * 30}
    )
    labels = pd.Series(["LOW"] * 10 + ["MEDIUM"] * 10 + ["HIGH"] * 10)
    config = {
        "random_state": 42,
        "models": {
            "random_forest": {
                "n_estimators": 8,
                "max_depth": 3,
                "n_jobs": 1,
            }
        },
    }
    pipeline = fit_model(build_model("random_forest", list(features), config), features, labels)
    schema = create_feature_schema(features)
    metadata = {
        "model_version": "unit-test",
        "model_name": "random_forest",
        "class_order": list(RISK_CLASSES),
        "requires_complete_weather": False,
    }
    path = tmp_path / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "schema": schema,
            "metadata": metadata,
            "thresholds": {"low": 9, "medium": 19},
        },
        path,
    )
    return path