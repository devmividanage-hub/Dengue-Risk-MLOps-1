"""Additional critical date, numeric, lookup, and merge contracts."""

import pandas as pd
import pytest

from src.data.clean_data import clean_dengue, clean_weather
from src.data.validate_data import validate_data
from src.utils.helpers import DataValidationError


def valid_dengue():
    """Two consecutive unit-test reporting periods."""
    return pd.DataFrame(
        {
            "year": [2020, 2020],
            "week": [1, 2],
            "start.date": ["01/04/2020", "01/11/2020"],
            "end.date": ["01/10/2020", "01/17/2020"],
            "district": ["Colombo", "Colombo"],
            "cases": [1, 2],
        }
    )


@pytest.mark.parametrize("value", ["broken", float("inf"), 1.5])
def test_invalid_counts_fail(value):
    frame = valid_dengue()
    frame["cases"] = [value, value]
    with pytest.raises(DataValidationError):
        clean_dengue(frame)


def test_dates_and_chronology_contract():
    cleaned = clean_dengue(valid_dengue())
    with pytest.raises(DataValidationError, match="chronological"):
        validate_data(cleaned.iloc[::-1])
    cleaned.loc[0, "end_date"] = pd.Timestamp("2020-01-11")
    with pytest.raises(DataValidationError, match="Overlapping"):
        validate_data(cleaned)


def test_real_weather_lookup_schema():
    observations = pd.DataFrame(
        {"location_id": [1], "date": ["01/04/2020"], "temperature_2m_mean (°C)": [25]}
    )
    lookup = pd.DataFrame({"location_id": [1], "city_name": ["NuwaraEliya"]})
    settings = {"columns": {"temperature_2m_mean_c": ["mean_temperature", "mean"]}}
    cleaned = clean_weather(observations, lookup, settings)
    assert cleaned.district.iloc[0] == "Nuwara Eliya"
    assert cleaned.date.iloc[0] == pd.Timestamp("2020-01-04")
    with pytest.raises(DataValidationError, match="Missing district"):
        clean_weather(observations.assign(location_id=99), lookup, settings)
    with pytest.raises(DataValidationError, match="No configured numeric"):
        clean_weather(observations, lookup, {"columns": {"humidity": ["mean_humidity", "mean"]}})