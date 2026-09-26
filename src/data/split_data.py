"""Shared calendar splits with a purge of labels crossing the next split boundary."""

from typing import Any

import pandas as pd

from src.utils.helpers import DataValidationError
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def chronological_split(
    frame: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Split entire forecast weeks together and purge future boundary labels.

    No training label may come from the first validation forecast week or later;
    no validation label may come from the test forecast period.
    """
    ratios = [settings[name] for name in ["train", "validation", "test"]]
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Positive train/validation/test ratios must sum to one")
    dates = sorted(frame.start_date.unique())
    if len(dates) < 12:
        raise DataValidationError("At least 12 distinct forecast weeks are required")
    explicit_validation = settings.get("validation_start")
    explicit_test = settings.get("test_start")
    if bool(explicit_validation) != bool(explicit_test):
        raise ValueError("Configure both validation_start and test_start, or neither")
    validation_start = pd.Timestamp(explicit_validation or dates[int(len(dates) * ratios[0])])
    test_start = pd.Timestamp(explicit_test or dates[int(len(dates) * sum(ratios[:2]))])
    if validation_start >= test_start:
        raise ValueError("validation_start must precede test_start")
    masks = {
        "train": frame.start_date.lt(validation_start) & frame.target_date.lt(validation_start),
        "validation": frame.start_date.ge(validation_start)
        & frame.start_date.lt(test_start)
        & frame.target_date.lt(test_start),
        "test": frame.start_date.ge(test_start),
    }
    splits = {
        name: frame.loc[mask].sort_values(["start_date", "district"]).copy()
        for name, mask in masks.items()
    }
    report: dict[str, Any] = {"purged_rows": len(frame) - sum(map(len, splits.values()))}
    for name, split in splits.items():
        if split.empty:
            raise DataValidationError(f"Chronological {name} split is empty")
        report[name] = {
            "rows": len(split),
            "forecast_start": split.start_date.min().isoformat(),
            "forecast_end": split.start_date.max().isoformat(),
            "target_start": split.target_date.min().isoformat(),
            "target_end": split.target_date.max().isoformat(),
        }
        LOGGER.info("%s split: %s", name, report[name])
    return splits, report


