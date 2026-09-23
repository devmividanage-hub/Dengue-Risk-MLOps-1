"""Validation-selected model training, experiment tracking, and one test evaluation."""

import hashlib
import importlib.metadata
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature

from src.data.split_data import chronological_split
from src.data.validate_data import validate_data
from src.features.build_features import build_features, feature_columns
from src.features.create_target import add_next_week_cases, classify_cases, fit_thresholds
from src.models.evaluate import calculate_metrics, save_evaluation
from src.models.train import build_model, decode_labels, fit_model
from src.models.tune import tune_model
from src.utils.helpers import (
    RISK_CLASSES,
    ROOT,
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
