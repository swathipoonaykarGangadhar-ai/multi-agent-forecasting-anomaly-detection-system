"""
Generates realistic synthetic daily business metrics: revenue, units_sold,
active_customers, and churn_rate — with trend, weekly seasonality, noise,
and a handful of deliberately injected anomalies (spikes/drops/level-shifts).

Run directly to write data/sample_business_metrics.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_data(
    n_days: int = 365,
    start_date: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    # ---- Revenue: upward trend + weekly seasonality + noise ----
    trend = 12000 + t * 18
    weekly = 1500 * np.sin(2 * np.pi * t / 7 - 1.2)  # weekday/weekend pattern
    noise = rng.normal(0, 500, n_days)
    revenue = trend + weekly + noise

    # ---- Units sold: correlated with revenue but its own noise ----
    units_sold = (revenue / 45) + rng.normal(0, 15, n_days)

    # ---- Active customers: slow-growing, smoother ----
    active_customers = 4000 + t * 3.2 + rng.normal(0, 40, n_days)

    # ---- Churn rate: mostly stable around 2%, mean-reverting ----
    churn_rate = 0.02 + 0.004 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.0015, n_days)
    churn_rate = np.clip(churn_rate, 0.001, None)

    df = pd.DataFrame({
        "date": dates,
        "revenue": revenue.round(2),
        "units_sold": units_sold,  # cast to int only after anomaly injection
        "active_customers": active_customers,  # cast to int only after anomaly injection
        "churn_rate": churn_rate.round(4),
    })

    # ---- Inject deliberate anomalies so the anomaly-detection agent has signal ----
    anomaly_log = []

    def inject_spike(col, idx, factor):
        df.loc[idx, col] = df.loc[idx, col] * factor
        anomaly_log.append((dates[idx].date(), col, "spike" if factor > 1 else "drop"))

    def inject_level_shift(col, start_idx, end_idx, delta):
        df.loc[start_idx:end_idx, col] = df.loc[start_idx:end_idx, col] + delta
        anomaly_log.append((dates[start_idx].date(), col, f"level_shift x{end_idx-start_idx+1}d"))

    # A flash-sale style revenue spike
    inject_spike("revenue", min(100, n_days - 1), 2.4)
    # A revenue crash (e.g. outage)
    inject_spike("revenue", min(210, n_days - 1), 0.35)
    # Sustained churn spike (e.g. price increase backlash) for 5 days
    _start = min(260, max(0, n_days - 6))
    inject_level_shift("churn_rate", _start, min(_start + 4, n_days - 1), 0.02)
    # Sudden customer count drop (data glitch or mass cancellation)
    inject_spike("active_customers", min(320, n_days - 1), 0.85)
    # Units sold spike without matching revenue spike (heavy discounting)
    inject_spike("units_sold", min(150, n_days - 1), 1.9)

    df["units_sold"] = df["units_sold"].round(0).astype(int)
    df["active_customers"] = df["active_customers"].round(0).astype(int)

    out_path = Path(__file__).resolve().parents[2] / "data" / "sample_business_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print("Injected anomalies (ground truth, for validation only):")
    for d, col, kind in anomaly_log:
        print(f"  {d}  {col:<18} {kind}")

    return df


if __name__ == "__main__":
    generate_sample_data()
