"""
Multi-Agent Forecasting & Anomaly Detection System - CLI entry point.

Usage:
    python src/main.py --csv data/sample_business_metrics.csv --periods 14
    python src/main.py --csv path/to/your_metrics.csv --periods 30 --no-llm

Requires an ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment (or a .env
file) for the LLM-driven crew. Use --no-llm to run only the deterministic
forecasting/anomaly pipeline and skip the narrative report (no API key needed).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.forecasting_tools import _forecast_series
from tools.anomaly_tools import _detect_anomalies


def run_deterministic_pipeline(csv_path: str, periods: int, outputs_dir: Path):
    """Runs forecasting + anomaly detection directly (no LLM) and saves charts.
    Useful for quick validation and for the --no-llm mode."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]

    outputs_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for col in numeric_cols:
        forecast = _forecast_series(df[col], periods=periods)
        anomalies = _detect_anomalies(df[col])
        flagged = anomalies[anomalies["is_anomaly"]]
        results[col] = {"forecast": forecast, "anomalies": flagged}

        # --- Chart: history + anomalies + forecast ---
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(df["date"], df[col], color="#2563eb", linewidth=1.3, label="History")

        if not flagged.empty:
            ax.scatter(
                df.loc[flagged.index, "date"], flagged["value"],
                color="#dc2626", zorder=5, s=45, label="Anomaly",
            )

        future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=periods)
        ax.plot(future_dates, forecast["forecast_values"], color="#16a34a", linewidth=1.5,
                 linestyle="--", label="Forecast")
        ax.fill_between(future_dates, forecast["lower_bound_95"], forecast["upper_bound_95"],
                          color="#16a34a", alpha=0.15, label="95% CI")

        ax.set_title(f"{col} — history, anomalies & {periods}-day forecast")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("date")
        fig.tight_layout()
        fig.savefig(outputs_dir / f"{col}_chart.png", dpi=130)
        plt.close(fig)

    return results


def print_deterministic_summary(results: dict):
    print("\n" + "=" * 70)
    print("DETERMINISTIC PIPELINE RESULTS (no LLM)")
    print("=" * 70)
    for col, r in results.items():
        f = r["forecast"]
        print(f"\n[{col}]")
        print(f"  Forecast method     : {f['method']}")
        print(f"  Trend direction     : {f['trend_direction']}  ({f['pct_change_vs_recent']:+.2f}% vs recent avg)")
        print(f"  Forecast avg        : {f['forecast_avg']}")
        n_anom = len(r["anomalies"])
        print(f"  Anomalies detected  : {n_anom}")
        for _, row in r["anomalies"].iterrows():
            print(f"    - value={row['value']:.2f}  z={row['z_score']:.2f}  votes={row['votes']}/3")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Forecasting & Anomaly Detection System")
    parser.add_argument("--csv", default="data/sample_business_metrics.csv", help="Path to input CSV (must have a 'date' column)")
    parser.add_argument("--periods", type=int, default=14, help="Forecast horizon in days")
    parser.add_argument("--outputs", default="outputs", help="Directory to write charts/report to")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM crew; only run deterministic forecasting/anomaly detection + charts")
    args = parser.parse_args()

    csv_path = str(Path(args.csv).resolve())
    outputs_dir = Path(args.outputs).resolve()

    print(f"Loading data from: {csv_path}")
    results = run_deterministic_pipeline(csv_path, args.periods, outputs_dir)
    print_deterministic_summary(results)
    print(f"\nCharts saved to: {outputs_dir}")

    if args.no_llm:
        print("\n--no-llm set: skipping LLM crew / narrative report.")
        return

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print(
            "\nNo ANTHROPIC_API_KEY or OPENAI_API_KEY found in environment. "
            "Set one (e.g. in a .env file) to run the LLM agent crew, "
            "or re-run with --no-llm to skip it."
        )
        return

    from agents.crew_setup import build_crew

    print("\nLaunching agent crew (Data Analyst -> Forecaster -> Anomaly Detective -> Reporter)...\n")
    crew = build_crew(csv_path, forecast_periods=args.periods)
    result = crew.kickoff()

    report_path = outputs_dir / "executive_report.md"
    report_path.write_text(str(result))
    print(f"\nExecutive report written to: {report_path}")


if __name__ == "__main__":
    main()
