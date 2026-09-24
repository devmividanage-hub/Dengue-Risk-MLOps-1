"""Load one coherent model bundle and validate engineered inference features."""

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.clean_data import normalize_districts
from src.utils.helpers import RISK_CLASSES, ModelNotFoundError, resolve_path


class RiskPredictor:
    """An immutable-per-request service contract around a fitted sklearn pipeline."""

    def __init__(self, model_path: str | Path = "models/best_model.joblib"):
        path = resolve_path(model_path)
        if not path.exists():
            raise ModelNotFoundError("Model artifact not found; run training before prediction")
        # joblib is trusted local input only, never supplied through the API.
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.schema = bundle["schema"]
        self.metadata = bundle["metadata"]
        self.thresholds = bundle["thresholds"]
        if list(self.pipeline.feature_names_in_) != self.schema["columns"]:
            raise ValueError("Model feature schema does not match fitted preprocessing")
        if not np.array_equal(self.pipeline.classes_, [0, 1, 2]):
            raise ValueError("Model class order does not match LOW/MEDIUM/HIGH")

    def validate_features(self, district: str, features: dict[str, Any]) -> pd.DataFrame:
        """Require exact feature names; nullable numeric fields use saved imputation."""
        if not isinstance(district, str) or not district.strip() or len(district) > 80:
            raise ValueError("district must be a nonempty name of at most 80 characters")
        expected = set(self.schema["numeric"])
        missing, unknown = expected - set(features), set(features) - expected
        if missing or unknown:
            raise ValueError(
                f"Invalid feature schema: missing={sorted(missing)}, extra={sorted(unknown)}"
            )
        values: dict[str, Any] = {"district": normalize_districts(pd.Series([district])).iloc[0]}
        for name, value in features.items():
            if value is None:
                if name not in self.schema["nullable"]:
                    raise ValueError(f"{name} cannot be null")
                values[name] = np.nan
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric or null")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            nonnegative = name.startswith("cases") and name != "cases_growth_rate"
            nonnegative = nonnegative or any(
                word in name for word in ("rainfall", "wind", "humidity")
            )
            if nonnegative and value < 0:
                raise ValueError(f"{name} cannot be negative")
            bounds = {
                "month": (1, 12),
                "quarter": (1, 4),
                "week_of_year": (1, 53),
                "calendar_year": (1900, 2200),
            }
            if name in bounds and (
                not bounds[name][0] <= value <= bounds[name][1] or value != int(value)
            ):
                raise ValueError(f"{name} is outside valid integer bounds")
            if (name == "cases" or name.startswith("cases_lag_")) and value != int(value):
                raise ValueError(f"{name} must be a whole case count")
            values[name] = float(value)
        if self.metadata.get("requires_complete_weather"):
            observed = [
                key
                for key in expected
                if key
                in {
                    "mean_temperature",
                    "min_temperature",
                    "max_temperature",
                    "total_rainfall",
                    "mean_humidity",
                    "mean_wind_speed",
                    "mean_daily_max_wind_speed",
                }
            ]
            if any(pd.isna(values[key]) for key in observed):
                raise ValueError("This model requires complete observed weather for week t")
        return pd.DataFrame([values], columns=self.schema["columns"])

    def predict(self, district: str, features: dict[str, Any]) -> dict[str, Any]:
        """Return next-week risk, probabilities, and the persisted model version."""
        frame = self.validate_features(district, features)
        prediction = int(self.pipeline.predict(frame)[0])
        probabilities = self.pipeline.predict_proba(frame)[0]
        return {
            "district": frame.district.iloc[0],
            "risk_class": RISK_CLASSES[prediction],
            "probabilities": dict(zip(RISK_CLASSES, map(float, probabilities), strict=True)),
            "model_version": self.metadata["model_version"],
        }
