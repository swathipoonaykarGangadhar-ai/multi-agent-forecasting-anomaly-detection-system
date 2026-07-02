"""Simple grounding tool: gives agents real summary statistics of the dataset
so the Data Analyst agent doesn't have to (and can't) hallucinate numbers."""
import json

import pandas as pd
from crewai.tools import tool


@tool("summarize_dataset")
def summarize_dataset(csv_path: str) -> str:
    """
    Load a CSV time series of business metrics and return summary statistics:
    date range, row count, numeric columns available, and per-column
    mean/std/min/max/last-value/7-day-change.

    Args:
        csv_path: Path to the CSV file (must have a 'date' column).

    Returns:
        A JSON string with dataset shape and per-column statistics. Always call
        this first so you know which columns exist before forecasting or
        anomaly-detecting them.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]

    col_stats = {}
    for col in numeric_cols:
        s = df[col]
        last_val = float(s.iloc[-1])
        week_ago_val = float(s.iloc[-8]) if len(s) >= 8 else float(s.iloc[0])
        pct_change_7d = ((last_val - week_ago_val) / week_ago_val * 100) if week_ago_val else 0.0
        col_stats[col] = {
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "last_value": round(last_val, 2),
            "pct_change_7d": round(pct_change_7d, 2),
        }

    return json.dumps({
        "csv_path": csv_path,
        "row_count": len(df),
        "date_range": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date())},
        "numeric_columns": numeric_cols,
        "column_stats": col_stats,
    }, indent=2)
