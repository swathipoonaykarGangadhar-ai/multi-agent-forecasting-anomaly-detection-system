"""
Unit tests for the deterministic core (no API key required).

Run with:  pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.forecasting_tools import _forecast_series
from tools.anomaly_tools import _detect_anomalies
from data.generate_sample_data import generate_sample_data


@pytest.fixture(scope="module")
def sample_df():
    return generate_sample_data(n_days=365, seed=7, write_csv=False)


def test_generated_data_has_expected_columns(sample_df):
    expected = {"date", "revenue", "units_sold", "active_customers", "churn_rate"}
    assert expected.issubset(set(sample_df.columns))
    assert len(sample_df) == 365


def test_generated_data_has_no_nulls(sample_df):
    assert not sample_df.isnull().values.any()


def test_forecast_returns_expected_shape(sample_df):
    result = _forecast_series(sample_df["revenue"], periods=10, seasonal_periods=7)
    assert len(result["forecast_values"]) == 10
    assert len(result["lower_bound_95"]) == 10
    assert len(result["upper_bound_95"]) == 10
    assert result["trend_direction"] in {"up", "down", "flat"}


def test_forecast_confidence_bounds_are_ordered(sample_df):
    result = _forecast_series(sample_df["revenue"], periods=10)
    for lo, val, hi in zip(result["lower_bound_95"], result["forecast_values"], result["upper_bound_95"]):
        assert lo <= val <= hi


def test_forecast_falls_back_gracefully_on_short_series():
    short_series = pd.Series(np.linspace(100, 120, 5))
    result = _forecast_series(short_series, periods=3, seasonal_periods=7)
    assert result["method"] == "linear_trend_fallback"
    assert len(result["forecast_values"]) == 3


def test_forecast_detects_upward_trend():
    trending = pd.Series(np.arange(50) * 10 + 100.0)
    result = _forecast_series(trending, periods=5, seasonal_periods=7)
    assert result["trend_direction"] == "up"


def test_anomaly_detection_flags_injected_spike():
    """A single 5-sigma spike in an otherwise flat series must be caught."""
    rng = np.random.default_rng(0)
    series = pd.Series(100 + rng.normal(0, 2, 60))
    series.iloc[30] = 100 + 5 * 20  # obvious spike
    result = _detect_anomalies(series, window=14)
    assert result.loc[30, "is_anomaly"] == True  # noqa: E712


def test_anomaly_detection_no_false_positive_on_flat_series():
    rng = np.random.default_rng(1)
    series = pd.Series(100 + rng.normal(0, 1, 60))
    result = _detect_anomalies(series, window=14)
    # A gentle-noise flat series should have very few (ideally zero) flags
    assert result["is_anomaly"].sum() <= 2


def test_anomaly_detection_catches_known_injected_anomalies(sample_df):
    """Cross-check against the ground-truth anomalies the generator injects."""
    df = generate_sample_data(n_days=365, seed=42, write_csv=False)  # matches default demo dataset
    result = _detect_anomalies(df["revenue"], window=14)
    flagged_dates = set(df.loc[result["is_anomaly"], "date"].dt.date)
    # These are the two revenue anomalies injected by generate_sample_data(seed=42)
    assert pd.Timestamp("2025-04-11").date() in flagged_dates
    assert pd.Timestamp("2025-07-30").date() in flagged_dates
