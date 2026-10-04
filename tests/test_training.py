"""Exercise local training with test-only data and isolated output directories."""

import sys

import pandas as pd
import pytest
import yaml

from src.utils.helpers import RISK_CLASSES, load_config, read_json


@pytest.mark.parametrize("tuning", [False, True])
def test_local_training_and_saved_prediction(tmp_path, monkeypatch, tuning):
    # Removed services must not be needed even when unavailable in the environment.
    for module in ("mlflow", "dotenv", "fastapi", "evidently", "dvc"):
        monkeypatch.setitem(sys.modules, module, None)
    from pipelines.training_pipeline import run_training_pipeline
    from src.models.predict import RiskPredictor

    dates = pd.date_range("2020-01-04", periods=60, freq="7D")
    frame = pd.DataFrame(
        {
            "district": "Colombo",
            "start_date": dates,
            "end_date": dates + pd.Timedelta(days=6),
            "year": dates.isocalendar().year.to_numpy(),
            "week": dates.isocalendar().week.to_numpy(),
            "cases": [(index * 7) % 31 for index in range(len(dates))],
            "weather_complete": True,
        }
    )
    processed = tmp_path / "weekly.csv"
    frame.to_csv(processed, index=False)
    data_settings = load_config("configs/data.yaml")
    data_settings["paths"]["processed"] = str(processed)
    settings = load_config("configs/model.yaml")
    settings["artifacts_dir"] = str(tmp_path / "models")
    settings["reports_dir"] = str(tmp_path / "reports")
    settings["tuning"] = {"enabled": tuning, "iterations": 1, "folds": 2}
    for name in ("random_forest", "xgboost"):
        settings["models"][name].update(n_estimators=3, n_jobs=1)
    data_config = tmp_path / "data.yaml"
    model_config = tmp_path / "model.yaml"
    data_config.write_text(yaml.safe_dump(data_settings), encoding="utf-8")
    model_config.write_text(yaml.safe_dump(settings), encoding="utf-8")

    metadata = run_training_pipeline(data_config, model_config)

    comparison = read_json(tmp_path / "reports/metrics/validation_comparison.json")
    assert set(comparison) == set(settings["models"])
    assert metadata["validation_score"] == max(
        result["validation"][settings["selection_metric"]] for result in comparison.values()
    )
    assert read_json(tmp_path / "models/model_metadata.json") == metadata
    assert read_json(tmp_path / "reports/metrics/best_model_test.json") == metadata["test_metrics"]
    if tuning:
        for name in settings["models"]:
            assert read_json(tmp_path / f"reports/metrics/{name}_tuning.json")["folds"] == 2
    example = read_json(tmp_path / "reports/prediction_example.json")
    predictor = RiskPredictor(tmp_path / "models/best_model.joblib")
    result = predictor.predict(example["district"], example["features"])
    assert result["risk_class"] in RISK_CLASSES
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert result["model_version"] == metadata["model_version"]
