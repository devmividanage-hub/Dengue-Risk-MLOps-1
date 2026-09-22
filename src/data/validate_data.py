"""Strict dataset contracts and inspectable quality statistics."""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.utils.helpers import DataValidationError


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Fail on empty data or missing critical columns."""
    if frame.empty:
        raise DataValidationError("Dataset is empty")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")


def validate_data(frame: pd.DataFrame, kind: str = "dengue") -> dict[str, Any]:
    """Validate cleaned data and report duplicate/missingness rates."""
    if kind not in {"dengue", "weather"}:
        raise ValueError("kind must be dengue or weather")
    date_column = "start_date" if kind == "dengue" else "date"
    required = ["district", date_column]
    if kind == "dengue":
        required += ["end_date", "year", "week", "cases"]
    require_columns(frame, required)
    for column in [date_column] + (["end_date"] if kind == "dengue" else []):
        if not pd.api.types.is_datetime64_any_dtype(frame[column]) or frame[column].isna().any():
            raise DataValidationError(f"Invalid or missing dates in {column}")
    if frame.district.isna().any() or frame.district.astype(str).str.strip().eq("").any():
        raise DataValidationError("Missing district values")
    keys = ["district", date_column]
    if frame.duplicated(keys).any():
        raise DataValidationError(f"Conflicting duplicate observations for {keys}")
    if kind == "dengue":
        if not pd.api.types.is_numeric_dtype(frame.cases):
            raise DataValidationError("Cases must be numeric")
        if frame.cases.dropna().lt(0).any():
            raise DataValidationError("Negative dengue case counts are impossible")
        if frame.cases.dropna().mod(1).ne(0).any():
            raise DataValidationError("Dengue cases must be whole counts")
        if not frame.week.between(1, 53).all():
            raise DataValidationError("Reporting week must be between 1 and 53")
        durations = (frame.end_date - frame.start_date).dt.days
        if not durations.between(5, 7).all():
            raise DataValidationError("Invalid dengue reporting interval")
        if frame.duplicated(["district", "year", "week"]).any():
            raise DataValidationError("Duplicate district/year/week keys")
    irregular = 0
    for district, group in frame.groupby("district", sort=False):
        if not group[date_column].is_monotonic_increasing:
            raise DataValidationError(f"Dates are not chronological for {district}")
        if kind == "dengue":
            if (group.start_date <= group.end_date.shift()).any():
                raise DataValidationError(f"Overlapping dengue intervals for {district}")
            gaps = group.start_date.diff().dt.days.dropna()
            irregular += int(gaps.ne(7).sum())
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rate": float(frame.duplicated().mean()),
        "missing_percentages": (frame.isna().mean() * 100).to_dict(),
        "irregular_week_intervals": irregular,
        "start": frame[date_column].min().isoformat(),
        "end": frame[date_column].max().isoformat(),
    }
