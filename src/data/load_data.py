"""Read configured CSV inputs without modifying raw files."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.helpers import DatasetNotFoundError, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def load_csv_files(path: str | Path, exclude: set[str] | None = None) -> pd.DataFrame:
    """Concatenate CSV shards; metadata files may be explicitly excluded."""
    location = resolve_path(path)
    files = [location] if location.is_file() else sorted(location.glob("*.csv"))
    files = [file for file in files if file.name not in (exclude or set())]
    if not files:
        raise DatasetNotFoundError(f"No CSV datasets found in {location}")
    frames = []
    for file in files:
        try:
            frame = pd.read_csv(file)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
            raise ValueError(f"Cannot read CSV {file.name}: {error}") from error
        LOGGER.info("Loaded %s: %d rows, %d columns", file.name, *frame.shape)
        frames.append(frame)
    schemas = [set(frame.columns) for frame in frames]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(f"CSV shards have conflicting schemas in {location}")
    return pd.concat(frames, ignore_index=True)


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dengue, weather observations, and optional weather location lookup.

    Dates are parsed strictly during cleaning after column normalization.
    """
    paths = config["paths"]
    dengue = load_csv_files(paths["dengue"])
    weather = load_csv_files(
        paths["weather"], {config["weather"].get("location_filename", "locationData.csv")}
    )
    lookup_path = paths.get("weather_locations")
    locations = load_csv_files(lookup_path) if lookup_path else pd.DataFrame()
    return dengue, weather, locations
