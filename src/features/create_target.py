"""Next-week labels using thresholds learned exclusively from training targets."""

from typing import Any

import numpy as np
import pandas as pd

from src.data.validate_data import require_columns
from src.utils.helpers import DataValidationError


def add_next_week_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Shift within district, requiring genuinely adjacent calendar weeks.

    Missing final weeks and gaps have no label; no artificial targets are created.
    """
    require_columns(frame, ["district", "start_date", "cases"])
    result = frame.sort_values(["district", "start_date"]).copy()
    groups = result.groupby("district", sort=False)
    dates = groups.start_date.shift(-1)
    adjacent = (dates - result.start_date).dt.days.eq(7)
    result["target_date"] = dates.where(adjacent)
    result["next_week_cases"] = groups.cases.shift(-1).where(adjacent)
    return result


def fit_thresholds(
    training_cases: pd.Series,
    low_quantile: float = 0.5,
    medium_quantile: float = 0.8,
) -> dict[str, Any]:
    """Fit global quantile thresholds only on the supplied training target counts."""
    if not 0 < low_quantile < medium_quantile < 1:
        raise ValueError("Quantiles must satisfy 0 < low < medium < 1")
    values = pd.to_numeric(training_cases, errors="raise").dropna()
    if values.empty or values.lt(0).any() or not np.isfinite(values).all():
        raise DataValidationError("Training targets must contain finite nonnegative counts")
    low, medium = values.quantile([low_quantile, medium_quantile]).tolist()
    if low >= medium:
        raise DataValidationError("Quantile thresholds collapse; cannot define three risk classes")
    return {
        "low": float(low),
        "medium": float(medium),
        "low_quantile": low_quantile,
        "medium_quantile": medium_quantile,
        "training_target_rows": len(values),
        "definition": "LOW <= low; MEDIUM > low and <= medium; HIGH > medium",
    }


def classify_cases(cases: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    """Reuse frozen thresholds for any labeled period; missing targets stay missing."""
    result = pd.Series(pd.NA, index=cases.index, dtype="string", name="risk_class")
    result.loc[cases.le(thresholds["low"])] = "LOW"
    result.loc[cases.gt(thresholds["low"]) & cases.le(thresholds["medium"])] = "MEDIUM"
    result.loc[cases.gt(thresholds["medium"])] = "HIGH"
    return result
