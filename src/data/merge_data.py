"""Aggregate daily weather into the source dengue reporting calendar."""

from typing import Any

import numpy as np
import pandas as pd

from src.data.validate_data import require_columns
from src.utils.helpers import DataValidationError
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def aggregate_weather(
    dengue: pd.DataFrame,
    weather: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Assign daily dates to actual reporting intervals, never future days.

    Source year/week labels are preserved, including non-ISO year boundaries.
    Incomplete weeks remain visible but their numeric aggregates are missing.
    Rainfall requires every day's observation, so partial sums are never called totals.
    """
    require_columns(dengue, ["district", "year", "week", "start_date", "end_date"])
    require_columns(weather, ["district", "date"])
    available = {key: value for key, value in settings["columns"].items() if key in weather}
    if not available:
        raise DataValidationError("No supported weather measurements available")
    parts = []
    for district, daily in weather.groupby("district"):
        calendar = dengue.loc[
            dengue.district.eq(district), ["start_date", "end_date", "year", "week"]
        ].sort_values("start_date")
        if calendar.empty:
            continue
        assigned = pd.merge_asof(
            daily.sort_values("date"),
            calendar,
            left_on="date",
            right_on="start_date",
            direction="backward",
        )
        assigned = assigned[assigned.date.le(assigned.end_date)].copy()
        if assigned.empty:
            continue
        grouped = assigned.groupby(["district", "year", "week"], observed=True)
        weekly = grouped.date.nunique().to_frame("weather_days")
        weekly["weather_complete"] = weekly.weather_days.ge(settings.get("min_days_per_week", 7))
        for source, (output, statistic) in available.items():
            if statistic not in {"mean", "min", "max", "sum"}:
                raise ValueError(f"Unsupported weather aggregation: {statistic}")
            values = grouped[source].agg(statistic)
            complete = weekly.weather_complete & grouped[source].count().eq(weekly.weather_days)
            weekly[output] = values.where(complete, np.nan)
        weekly["weather_complete"] &= (
            weekly[[output for output, _ in available.values()]].notna().all(axis=1)
        )
        parts.append(weekly.reset_index())
    if not parts:
        raise DataValidationError(
            "Weather has no district/date overlap with dengue reporting weeks"
        )
    return pd.concat(parts, ignore_index=True)


def merge_data(
    dengue: pd.DataFrame,
    weather: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left merge with coverage statistics and an explicit minimum match contract."""
    weekly = aggregate_weather(dengue, weather, settings)
    merged = dengue.merge(
        weekly, on=["district", "year", "week"], how="left", validate="one_to_one", indicator=True
    )
    matched = merged["_merge"].eq("both")
    fraction = float(matched.mean())
    merged["weather_complete"] = merged.weather_complete.fillna(False).astype(bool)
    report = {
        "dengue_rows_before_merge": len(dengue),
        "weather_rows_before_merge": len(weather),
        "weather_weekly_rows": len(weekly),
        "rows_after_merge": len(merged),
        "unmatched_dengue_rows": int((~matched).sum()),
        "percentage_successfully_matched": 100 * fraction,
        "complete_weather_rows": int(merged.weather_complete.sum()),
        "unmatched_dengue_districts": sorted(set(dengue.district) - set(weather.district)),
        "unused_weather_locations": sorted(set(weather.district) - set(dengue.district)),
        "weekly_weather_features": [
            value[0] for key, value in settings["columns"].items() if key in weather
        ],
        "calendar": "source start_date/end_date intervals; source year/week labels",
    }
    LOGGER.info("Merge quality: %s", report)
    if fraction < settings.get("min_match_fraction", 0.65):
        raise DataValidationError(
            f"Weather match rate {fraction:.1%} below configured minimum; quality: {report}"
        )
    if (~matched).any():
        LOGGER.warning("Retaining %d dengue rows without weather", (~matched).sum())
    return merged.drop(columns="_merge").sort_values(["district", "start_date"]), report
