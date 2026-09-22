"""Schema-aware cleaning with no retrospective filling or raw-file changes."""

import re
from typing import Any

import numpy as np
import pandas as pd

from .validate_data import require_columns, validate_data
from src.utils.helpers import DataValidationError
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize punctuation and units, preserving recognizable source names."""
    result = frame.copy()
    result.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") for column in result.columns
    ]
    if result.columns.duplicated().any():
        raise DataValidationError("Column normalization creates duplicate names")
    return result


def normalize_districts(values: pd.Series) -> pd.Series:
    """Resolve observed spelling variants without inventing geographic mappings."""
    aliases = {
        "nuwaraeliya": "Nuwara Eliya",
        "hambanthota": "Hambantota",
        "moneragala": "Monaragala",
        "kalmune": "Kalmunai",
    }
    cleaned = values.astype("string").str.replace(r"\[\d+\]", "", regex=True)
    cleaned = cleaned.str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    return cleaned.map(lambda value: aliases.get(str(value).lower(), value))


def parse_dates(values: pd.Series, settings: dict[str, Any] | None = None) -> pd.Series:
    """Strict date parsing; malformed and missing values are never silently dropped."""
    settings = settings or {}
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = values.copy()
    else:
        parsed = pd.to_datetime(
            values,
            format=settings.get("format", "mixed"),
            dayfirst=settings.get("dayfirst", False),
            errors="coerce",
        )
    if parsed.isna().any():
        examples = values[parsed.isna()].head(3).tolist()
        raise DataValidationError(f"Invalid or missing dates: {examples}")
    return parsed.dt.normalize()


def numeric_column(values: pd.Series, name: str) -> pd.Series:
    """Preserve true missingness; reject malformed numeric observations."""
    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & (converted.isna() | ~np.isfinite(converted))
    if invalid.any():
        raise DataValidationError(f"Invalid numeric values in {name}")
    return converted


def clean_dengue(frame: pd.DataFrame, dates: dict[str, Any] | None = None) -> pd.DataFrame:
    """Clean dengue counts, retain missing cases, and sort independently by district."""
    result = normalize_columns(frame).drop_duplicates()
    require_columns(result, ["year", "week", "start_date", "end_date", "district", "cases"])
    for column in ["start_date", "end_date"]:
        result[column] = parse_dates(result[column], dates)
    result["district"] = normalize_districts(result.district)
    for column in ["cases", "year", "week"]:
        result[column] = numeric_column(result[column], column)
    if result[["year", "week"]].isna().any().any():
        raise DataValidationError("Missing year/week keys")
    for column in ["year", "week"]:
        if result[column].mod(1).ne(0).any():
            raise DataValidationError(f"{column} must contain integers")
        result[column] = result[column].astype(int)
    result = result.drop_duplicates().sort_values(["district", "start_date"]).reset_index(drop=True)
    validate_data(result)
    LOGGER.info(
        "Cleaned dengue: %d rows, %d missing counts", len(result), result.cases.isna().sum()
    )
    return result


def clean_weather(
    frame: pd.DataFrame,
    locations: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
    dates: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Join real location metadata; retain missing weather for training-only imputation.

    Unmapped towns are retained in the quality report, never assigned to districts
    by guesswork. No forward/backward filling is used.
    """
    settings = settings or {}
    result = normalize_columns(frame).drop_duplicates()
    result = result.rename(columns={settings.get("date_column", "date"): "date"})
    district_column = settings.get("district_column", "district")
    if district_column in result:
        result = result.rename(columns={district_column: "district"})
    else:
        key = settings.get("location_column", "location_id")
        name = settings.get("location_name_column", "city_name")
        require_columns(result, [key])
        lookup = normalize_columns(locations if locations is not None else pd.DataFrame())
        require_columns(lookup, [key, name])
        if lookup[key].duplicated().any():
            raise DataValidationError("Weather location IDs must be unique")
        result = result.merge(lookup[[key, name]], on=key, how="left", validate="many_to_one")
        result = result.rename(columns={name: "district"})
    require_columns(result, ["date", "district"])
    result["date"] = parse_dates(result.date, dates)
    result["district"] = normalize_districts(result.district)
    mapping = settings.get("columns", {})
    if not any(column in result for column in mapping):
        raise DataValidationError(
            f"No configured numeric weather columns found. Actual columns: {list(result.columns)}"
        )
    for column in mapping:
        if column in result:
            result[column] = numeric_column(result[column], column)
            if any(word in column for word in ("rain", "precipitation", "wind", "humidity")):
                if result[column].dropna().lt(0).any():
                    raise DataValidationError(f"Negative weather values in {column}")
    result = result.drop_duplicates().sort_values(["district", "date"]).reset_index(drop=True)
    validate_data(result, "weather")
    LOGGER.info("Cleaned weather: %d rows; no temporal filling applied", len(result))
    return result
