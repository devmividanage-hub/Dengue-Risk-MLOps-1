"""Dataset contracts using tiny unit fixtures, never training data."""

import pandas as pd
import pytest

from src.data.clean_data import clean_dengue, clean_weather, normalize_districts, parse_dates
from src.data.load_data import load_csv_files
from src.data.merge_data import merge_data
from src.utils.helpers import DatasetNotFoundError, DataValidationError


@pytest.fixture
def dengue() -> pd.DataFrame:
    """One real-calendar-shaped reporting week for contract tests."""
    return pd.DataFrame(
        {
            "year": [2020],
            "week": [2],
            "start.date": ["01/04/2020"],
            "end.date": ["01/10/2020"],
            "district": [" Colombo "],
            "cases": [10],
        }
    )


def test_missing_required_columns(dengue):
    with pytest.raises(DataValidationError, match="cases"):
        clean_dengue(dengue.drop(columns="cases"))


def test_negative_cases_rejected(dengue):
    dengue["cases"] = -1
    with pytest.raises(DataValidationError, match="Negative"):
        clean_dengue(dengue)


def test_district_normalization():
    result = normalize_districts(
        pd.Series([" NuwaraEliya ", "Hambanthota", "Kilinochchi[1]", "Moneragala", "Kalmune"])
    )
    assert result.tolist() == [
        "Nuwara Eliya",
        "Hambantota",
        "Kilinochchi",
        "Monaragala",
        "Kalmunai",
    ]


def test_date_parsing():
    assert parse_dates(pd.Series(["01/04/2020"])).iloc[0] == pd.Timestamp("2020-01-04")
    with pytest.raises(DataValidationError, match="dates"):
        parse_dates(pd.Series(["not-a-date"]))


def test_missing_counts_preserved(dengue):
    dengue["cases"] = None
    assert clean_dengue(dengue).cases.isna().all()


def test_conflicting_duplicates_rejected(dengue):
    conflict = pd.concat([dengue, dengue.assign(cases=20)])
    with pytest.raises(DataValidationError, match="duplicate"):
        clean_dengue(conflict)


def test_missing_dataset(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        load_csv_files(tmp_path)


def test_daily_weather_calendar_and_future_exclusion(dengue):
    cleaned = clean_dengue(dengue)
    weather = pd.DataFrame(
        {
            "district": ["Colombo"] * 8,
            "date": pd.date_range("2020-01-04", periods=8),
            "rain": [1] * 7 + [9999],
        }
    )
    settings = {"columns": {"rain": ["total_rainfall", "sum"]}, "min_match_fraction": 1}
    weather = clean_weather(weather, settings=settings)
    result, report = merge_data(cleaned, weather, settings)
    assert result.total_rainfall.iloc[0] == 7
    assert report["rows_after_merge"] == 1
    assert report["percentage_successfully_matched"] == 100


def test_partial_weather_not_reported_as_full_total(dengue):
    weather = pd.DataFrame(
        {
            "district": ["Colombo"] * 6,
            "date": pd.date_range("2020-01-04", periods=6),
            "rain": [1] * 6,
        }
    )
    settings = {"columns": {"rain": ["total_rainfall", "sum"]}}
    result, _ = merge_data(clean_dengue(dengue), weather, settings)
    assert pd.isna(result.total_rainfall.iloc[0])
    assert not result.weather_complete.iloc[0]