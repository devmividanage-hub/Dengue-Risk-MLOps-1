"""District-specific historical features available after reporting week t ends."""

import numpy as np
import pandas as pd

from src.data.validate_data import require_columns
from src.utils.helpers import DataValidationError


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build t -> t+1 features; current cases/weather and past observations are allowed.

    Calendar gaps invalidate affected lags/windows, rather than silently lagging
    over missing weeks. Rolling windows include t, never t+1. No imputation is
    fitted here. Raw reporting dates remain metadata and are excluded by training.
    """
    require_columns(frame, ["district", "start_date", "end_date", "cases", "year", "week"])
    result = frame.copy()
    for column in ["start_date", "end_date"]:
        result[column] = pd.to_datetime(result[column], errors="raise")
    if result.duplicated(["district", "start_date"]).any():
        raise DataValidationError("Duplicate district/week rows before feature engineering")
    result = result.sort_values(["district", "start_date"]).reset_index(drop=True)
    groups = result.groupby("district", sort=False)
    breaks = groups.start_date.diff().dt.days.ne(7)
    segments = breaks.groupby(result.district).cumsum()
    for lag in range(1, 5):
        elapsed = (result.start_date - groups.start_date.shift(lag)).dt.days
        uninterrupted = segments.eq(segments.groupby(result.district).shift(lag))
        result[f"cases_lag_{lag}"] = groups.cases.shift(lag).where(
            elapsed.eq(lag * 7) & uninterrupted
        )
    # Split windows into consecutive-calendar segments so gaps cannot enter a rolling mean.
    for window in [4, 8]:
        result[f"cases_rolling_mean_{window}"] = result.groupby(
            [result.district, segments], sort=False
        ).cases.transform(lambda values, size=window: values.rolling(size, min_periods=size).mean())
    denominator = result.cases_lag_1.replace(0, np.nan)
    result["cases_growth_rate"] = (result.cases - result.cases_lag_1) / denominator
    weather_lags = {
        "total_rainfall": "rainfall_lag_1",
        "mean_temperature": "temperature_lag_1",
        "mean_humidity": "humidity_lag_1",
    }
    adjacent = (result.start_date - groups.start_date.shift()).dt.days.eq(7)
    for source, output in weather_lags.items():
        if source in result:
            result[output] = groups[source].shift().where(adjacent)
    result["month"] = result.start_date.dt.month
    result["quarter"] = result.start_date.dt.quarter
    result["week_of_year"] = result.start_date.dt.isocalendar().week.astype(int)
    # Source year/week remain unchanged for audit; feature year uses the observed date.
    result["calendar_year"] = result.start_date.dt.year
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Explicit allowlist prevents target and audit metadata entering the model."""
    numerical = [
        "cases",
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
        "cases_rolling_mean_4",
        "cases_rolling_mean_8",
        "cases_growth_rate",
        "mean_temperature",
        "min_temperature",
        "max_temperature",
        "total_rainfall",
        "mean_humidity",
        "mean_wind_speed",
        "mean_daily_max_wind_speed",
        "rainfall_lag_1",
        "temperature_lag_1",
        "humidity_lag_1",
        "month",
        "quarter",
        "week_of_year",
        "calendar_year",
    ]
    return ["district"] + [column for column in numerical if column in frame]
