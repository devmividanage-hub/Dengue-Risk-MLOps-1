"""Load, clean, validate, merge, and persist reproducible weekly observations."""

from pathlib import Path

import pandas as pd

from src.data.clean_data import clean_dengue, clean_weather
from src.data.load_data import load_data
from src.data.merge_data import merge_data
from src.data.validate_data import validate_data
from src.utils.helpers import load_config, resolve_path, save_json


def run_data_pipeline(config_path: str | Path = "configs/data.yaml") -> pd.DataFrame:
    """Persist derived tables and quality reports; raw files are read-only inputs."""
    config = load_config(config_path)
    dengue_raw, weather_raw, locations = load_data(config)
    dengue = clean_dengue(dengue_raw, config["dates"])
    weather = clean_weather(weather_raw, locations, config["weather"], config["dates"])
    merged, report = merge_data(dengue, weather, config["weather"])
    report["dengue_validation"] = validate_data(dengue)
    report["weather_validation"] = validate_data(weather, "weather")
    report["removed_exact_dengue_duplicates"] = len(dengue_raw) - len(dengue)
    report["removed_exact_weather_duplicates"] = len(weather_raw) - len(weather)
    interim = resolve_path(config["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    dengue.to_csv(interim / "dengue_clean.csv", index=False)
    weather.to_csv(interim / "weather_clean.csv", index=False)
    processed = resolve_path(config["paths"]["processed"])
    processed.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(processed, index=False)
    save_json(config["paths"]["quality_report"], report)
    return merged
