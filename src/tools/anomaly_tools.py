"""
Anomaly detection tool used by the Anomaly Detection Agent.

Ensembles three classical methods so a single noisy point doesn't get
over/under-flagged:
  1. Rolling z-score on residuals (catches sudden spikes/drops)
  2. IQR (Tukey) rule on residuals (robust to non-normal data)
  3. Isolation Forest (catches multivariate/contextual weirdness)

A point is flagged as an anomaly if at least 2 of the 3 methods agree.
This is deterministic, explainable math -- the LLM agent interprets the
output, it does not decide which points are anomalous.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from crewai.tools import tool

warnings.filterwarnings("ignore")


def _detect_anomalies(
    series: pd.Series,
    window: int = 14,
    z_thresh: float = 3.0,
    contamination: float = 0.03,
) -> pd.DataFrame:
    s = series.astype(float)
    n = len(s)

    # --- Detrend with rolling median to get residuals (robust to trend/seasonality) ---
    rolling_median = s.rolling(window=window, center=True, min_periods=max(3, window // 3)).median()
    rolling_median = rolling_median.bfill().ffill()
    residual = s - rolling_median

    # --- Method 1: rolling z-score on residuals ---
    roll_std = residual.rolling(window=window, min_periods=max(3, window // 3)).std().bfill().ffill()
    roll_std = roll_std.replace(0, residual.std() or 1.0)
    z_scores = residual / roll_std
    flag_z = z_scores.abs() > z_thresh

    # --- Method 2: IQR rule on residuals ---
    q1, q3 = residual.quantile(0.25), residual.quantile(0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag_iqr = (residual < lower_fence) | (residual > upper_fence)

    # --- Method 3: Isolation Forest on [value, residual] ---
    X = np.column_stack([s.values, residual.values])
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso_pred = iso.fit_predict(X)  # -1 = anomaly
    flag_iso = pd.Series(iso_pred == -1, index=s.index)

    votes = flag_z.astype(int) + flag_iqr.astype(int) + flag_iso.astype(int)
    is_anomaly = votes >= 2

    out = pd.DataFrame({
        "value": s,
        "residual": residual.round(2),
        "z_score": z_scores.round(2),
        "votes": votes,
        "is_anomaly": is_anomaly,
    })
    return out


@tool("detect_anomalies")
def detect_anomalies(csv_path: str, column: str, window: int = 14, z_thresh: float = 3.0) -> str:
    """
    Detect anomalies (spikes, drops, level shifts) in a business metric time series
    using an ensemble of rolling z-score, IQR, and Isolation Forest methods.

    Args:
        csv_path: Path to a CSV file with a 'date' column and the target metric column.
        column: Name of the numeric column to inspect (e.g. 'revenue', 'churn_rate').
        window: Rolling window size (in periods) for detrending. Default 14.
        z_thresh: Z-score threshold for the z-score method. Default 3.0.

    Returns:
        A JSON string listing every flagged anomaly with its date, value, residual,
        z-score, and how many of the 3 detection methods voted for it (2 or 3 = flagged).
        Use this to ground any anomaly claims you make -- do not invent anomalies or
        dates that aren't in this output.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found. Available: {list(df.columns)}"})

    result_df = _detect_anomalies(df[column], window=window, z_thresh=z_thresh)
    result_df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    anomalies = result_df[result_df["is_anomaly"]]
    anomaly_list = [
        {
            "date": row["date"],
            "value": round(float(row["value"]), 2),
            "residual": float(row["residual"]),
            "z_score": float(row["z_score"]),
            "votes_out_of_3": int(row["votes"]),
        }
        for _, row in anomalies.iterrows()
    ]

    return json.dumps({
        "column": column,
        "total_points_checked": len(result_df),
        "anomalies_found": len(anomaly_list),
        "anomalies": anomaly_list,
    }, indent=2)
