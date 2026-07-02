"""
Forecasting tool used by the Forecasting Agent.

Deliberately NOT LLM-based: the numbers come from real statistical models
(Holt-Winters exponential smoothing, with a linear-regression fallback for
short/non-seasonal series). The LLM agent's job is to *interpret* these
numbers, not invent them.
"""
import json
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from crewai.tools import tool

warnings.filterwarnings("ignore")


def _forecast_series(
    series: pd.Series,
    periods: int = 14,
    seasonal_periods: int = 7,
) -> dict:
    """Core forecasting logic, returns a plain dict (JSON-serializable)."""
    series = series.astype(float)
    n = len(series)

    method_used = "holt_winters"
    try:
        if n < 2 * seasonal_periods + 1:
            raise ValueError("series too short for seasonal model")

        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)

        forecast = model.forecast(periods)
        resid_std = float(np.std(model.resid))
        lower = forecast - 1.96 * resid_std
        upper = forecast + 1.96 * resid_std

    except Exception:
        # Fallback: simple linear trend extrapolation
        method_used = "linear_trend_fallback"
        x = np.arange(n)
        coeffs = np.polyfit(x, series.values, 1)
        slope, intercept = coeffs
        future_x = np.arange(n, n + periods)
        forecast = pd.Series(slope * future_x + intercept)
        resid = series.values - (slope * x + intercept)
        resid_std = float(np.std(resid))
        lower = forecast - 1.96 * resid_std
        upper = forecast + 1.96 * resid_std

    recent_avg = float(series.tail(seasonal_periods).mean())
    forecast_avg = float(np.mean(forecast))
    pct_change = ((forecast_avg - recent_avg) / recent_avg * 100) if recent_avg else 0.0

    return {
        "method": method_used,
        "history_points": n,
        "forecast_horizon": periods,
        "forecast_values": [round(float(v), 2) for v in forecast],
        "lower_bound_95": [round(float(v), 2) for v in lower],
        "upper_bound_95": [round(float(v), 2) for v in upper],
        "recent_avg": round(recent_avg, 2),
        "forecast_avg": round(forecast_avg, 2),
        "pct_change_vs_recent": round(pct_change, 2),
        "trend_direction": "up" if pct_change > 1 else ("down" if pct_change < -1 else "flat"),
    }


@tool("forecast_metric")
def forecast_metric(csv_path: str, column: str, periods: int = 14, seasonal_periods: int = 7) -> str:
    """
    Forecast future values of a business metric column from a CSV time series.

    Args:
        csv_path: Path to a CSV file with a 'date' column and the target metric column.
        column: Name of the numeric column to forecast (e.g. 'revenue', 'units_sold').
        periods: Number of future periods (days) to forecast. Default 14.
        seasonal_periods: Seasonality length in periods. Default 7 (weekly).

    Returns:
        A JSON string with forecast values, 95% confidence bounds, trend direction,
        and percent change vs the recent average. Use this to ground any forecasting
        claims you make -- do not invent numbers.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found. Available: {list(df.columns)}"})

    result = _forecast_series(df[column], periods=periods, seasonal_periods=seasonal_periods)
    result["column"] = column
    result["last_date"] = str(df["date"].max().date())
    return json.dumps(result, indent=2)
