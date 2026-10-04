"""Validation-selected local model training and one test evaluation."""

import hashlib
import importlib.metadata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.split_data import chronological_split
from src.data.validate_data import validate_data
from src.features.build_features import build_features, feature_columns
from src.features.create_target import add_next_week_cases, classify_cases, fit_thresholds
from src.models.evaluate import calculate_metrics, save_evaluation
from src.models.train import build_model, decode_labels, fit_model
from src.models.tune import tune_model
from src.utils.helpers import (
    RISK_CLASSES,
    DataValidationError,
    load_config,
    resolve_path,
    save_json,
)
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def create_feature_schema(features: pd.DataFrame) -> dict[str, Any]:
    """Record the exact engineered input contract consumed by the saved pipeline."""
    return {
        "columns": list(features.columns),
        "categorical": ["district"],
        "numeric": [column for column in features if column != "district"],
        "nullable": [
            column
            for column in features
            if column
            not in {"district", "cases", "month", "quarter", "week_of_year", "calendar_year"}
        ],
        "class_order": list(RISK_CLASSES),
        "availability": "Inputs observed through week t; prediction concerns week t+1",
    }


def run_training_pipeline(
    data_config: str | Path = "configs/data.yaml",
    model_config: str | Path = "configs/model.yaml",
) -> dict[str, Any]:
    """Train three comparable pipelines; select by validation before opening test labels."""
    data_settings = load_config(data_config)
    settings = load_config(model_config)
    processed = resolve_path(data_settings["paths"]["processed"])
    if not processed.exists():
        raise FileNotFoundError("Processed weekly.csv missing; run scripts/run_pipeline.py first")
    frame = pd.read_csv(processed, parse_dates=["start_date", "end_date"])
    frame = frame.sort_values(["district", "start_date"])
    validate_data(frame)
    labeled = add_next_week_cases(build_features(frame))
    supervised = labeled.dropna(subset=["next_week_cases", "target_date"]).copy()
    excluded_weather = 0
    if data_settings["training"].get("require_complete_weather", True):
        if "weather_complete" not in supervised:
            raise DataValidationError("Missing weather_complete flag in processed dataset")
        eligible = supervised.weather_complete.eq(True)  # noqa: E712
        excluded_weather = int((~eligible).sum())
        supervised = supervised.loc[eligible].copy()
        LOGGER.info("Training scope: excluded %d rows without complete weather", excluded_weather)
    splits, split_report = chronological_split(supervised, data_settings["split"])
    quantiles = settings["target_quantiles"]
    thresholds = fit_thresholds(
        splits["train"].next_week_cases, quantiles["low"], quantiles["medium"]
    )
    thresholds["training_target_start"] = split_report["train"]["target_start"]
    thresholds["training_target_end"] = split_report["train"]["target_end"]
    for name in ["train", "validation"]:
        splits[name]["risk_class"] = classify_cases(splits[name].next_week_cases, thresholds)
    columns = feature_columns(supervised)
    schema = create_feature_schema(splits["train"][columns])
    artifacts_dir = resolve_path(settings["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = resolve_path(settings["reports_dir"])
    selection = settings["selection_metric"]
    if selection not in {"f1_macro", "weighted_f1", "accuracy", "high_risk_recall"}:
        raise ValueError(f"Unsupported validation selection metric: {selection}")
    comparison: dict[str, Any] = {}
    best_model = None
    best_name = ""
    best_score = -1.0
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model = build_model(name, columns, settings)
        if settings["tuning"]["enabled"]:
            model, tuning_report = tune_model(model, name, splits["train"], columns, settings)
            save_json(reports_dir / "metrics" / f"{name}_tuning.json", tuning_report)
        else:
            model = fit_model(model, splits["train"][columns], splits["train"].risk_class)
        validation = splits["validation"]
        metrics = calculate_metrics(
            validation.risk_class,
            decode_labels(model.predict(validation[columns])),
            model.predict_proba(validation[columns]),
        )
        save_evaluation(metrics, reports_dir, f"{name}_validation")
        comparison[name] = {"validation": metrics}
        if metrics[selection] > best_score:
            best_model, best_name, best_score = model, name, metrics[selection]
    if best_model is None:
        raise RuntimeError("No model was trained")
    # Selection is complete. Only now classify and evaluate the untouched test labels.
    test = splits["test"]
    test_labels = classify_cases(test.next_week_cases, thresholds)
    test_metrics = calculate_metrics(
        test_labels,
        decode_labels(best_model.predict(test[columns])),
        best_model.predict_proba(test[columns]),
    )
    save_evaluation(test_metrics, reports_dir, "best_model_test")
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    metadata = {
        "model_version": version,
        "model_name": best_name,
        "selection_metric": selection,
        "validation_score": best_score,
        "test_metrics": test_metrics,
        "split_periods": split_report,
        "class_order": list(RISK_CLASSES),
        "training_districts": sorted(splits["train"].district.unique().tolist()),
        "training_rows_excluded_missing_weather": excluded_weather,
        "unlabeled_rows": len(labeled) - len(labeled.dropna(subset=["next_week_cases"])),
        "requires_complete_weather": data_settings["training"]["require_complete_weather"],
        "processed_data_sha256": hashlib.sha256(processed.read_bytes()).hexdigest(),
        "data_configuration": data_settings,
        "model_configuration": settings,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]
        },
        "disclaimer": "Academic risk classification system; not a medical diagnostic system.",
    }
    bundle = {
        "pipeline": best_model,
        "schema": schema,
        "thresholds": thresholds,
        "metadata": metadata,
    }
    temporary = artifacts_dir / "best_model.joblib.tmp"
    joblib.dump(bundle, temporary)
    temporary.replace(artifacts_dir / "best_model.joblib")
    save_json(artifacts_dir / "target_thresholds.json", thresholds)
    save_json(artifacts_dir / "feature_schema.json", schema)
    save_json(artifacts_dir / "model_metadata.json", metadata)
    splits["train"][columns].to_csv(artifacts_dir / "reference_data.csv", index=False)
    save_json(reports_dir / "metrics" / "validation_comparison.json", comparison)
    row = test[columns].iloc[-1]
    save_json(
        reports_dir / "prediction_example.json",
        {
            "district": row.district,
            "features": {
                key: None if pd.isna(value) else float(value)
                for key, value in row.items()
                if key != "district"
            },
        },
    )
    report = (
        f"# Calculated model report\n\nSelected model: **{best_name}**. Version: `{version}`.\n\n"
        f"Selection used validation {selection}: {best_score:.4f}. "
        f"Final test macro F1: {test_metrics['f1_macro']:.4f}; "
        f"HIGH recall: {test_metrics['high_risk_recall']:.4f}.\n\n"
        "See metrics/validation_comparison.json and metrics/best_model_test.json for full results. "
        "These results were calculated from the supplied files, not hardcoded.\n\n"
        "The model is fitted on the training period only. Thresholds stay frozen. "
        "Complete-weather filtering defines the evaluated coverage; unsupported regions and "
        "later periods need new evaluation. Source provenance and reporting latency require "
        "verification before any operational use. This is an academic risk classification "
        "system and not a medical diagnostic system.\n"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_report.md").write_text(report, encoding="utf-8")
    LOGGER.info(
        "Selected %s; validation %s=%.4f; test macro F1=%.4f",
        best_name,
        selection,
        best_score,
        test_metrics["f1_macro"],
    )
    return metadata
