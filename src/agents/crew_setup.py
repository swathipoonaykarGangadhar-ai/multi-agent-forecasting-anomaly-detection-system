"""
Defines the four-agent crew:

  1. Data Analyst      - narrates pre-computed dataset statistics
  2. Forecasting Agent  - narrates pre-computed forecasts
  3. Anomaly Detective   - narrates pre-computed anomalies
  4. Insights Reporter   - synthesizes everything into a business-readable report

ARCHITECTURE NOTE: Earlier versions had agents call tools (summarize_dataset,
forecast_metric, detect_anomalies) themselves via LLM tool-calling. That hit a
well-documented, unresolved class of CrewAI bugs (GitHub #4238, #2895, #4093)
where tool calls silently don't reach Gemini (and are flaky elsewhere too),
causing agents to fabricate plausible-looking numbers instead.

The fix: never let the LLM decide whether to call a tool. All statistics,
forecasts, and anomalies are computed directly in Python (deterministic,
100% reliable) and embedded as JSON directly into each task's description.
Agents only narrate/format data they're handed -- there is no tool-calling
path for them to skip. This is strictly more reliable for a pipeline where
correctness of the numbers matters more than agent autonomy.
"""
import json
import os

from crewai import Agent, Task, Crew, Process, LLM

from tools.data_tools import _summarize_dataset
from tools.forecasting_tools import _forecast_series
from tools.anomaly_tools import _detect_anomalies

import pandas as pd

# --- Workaround for CrewAI GH issue #5886 ---
# CrewAI 1.14.4+ injects an Anthropic-only prompt-caching field ("cache_breakpoint")
# into every provider's messages, but only the Anthropic adapter knows how to strip
# it back out. Non-Anthropic providers with strict schema validation (Groq, some
# OpenAI-compatible endpoints) reject the request outright. This neutralizes the
# injection until CrewAI ships an official fix. Safe no-op for Anthropic too.
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except ImportError:
    pass  # older/newer crewai versions may not have this module; harmless if so


def get_llm() -> LLM:
    """
    Configure the LLM via LiteLLM model strings so you can swap providers
    with one env var. Examples:
      MODEL=gemini/gemini-flash-latest     (needs GEMINI_API_KEY, free tier, generous)
      MODEL=anthropic/claude-sonnet-5      (needs ANTHROPIC_API_KEY, paid)
      MODEL=openai/gpt-4o                  (needs OPENAI_API_KEY, paid)
      MODEL=groq/openai/gpt-oss-20b        (needs GROQ_API_KEY, free tier, tight rate limits)

    If MODEL isn't set explicitly, auto-picks based on whichever API key is present.
    Since agents no longer need to call tools (see module docstring), any of these
    providers works reliably now -- this is purely a narrative-writing LLM call.
    """
    model = os.getenv("MODEL")
    if not model:
        if os.getenv("GEMINI_API_KEY"):
            model = "gemini/gemini-flash-latest"
        elif os.getenv("GROQ_API_KEY"):
            model = "groq/openai/gpt-oss-20b"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = "anthropic/claude-sonnet-5"
        elif os.getenv("OPENAI_API_KEY"):
            model = "openai/gpt-4o"
        else:
            model = "anthropic/claude-sonnet-5"  # will fail fast with a clear error if no key is set

    force_litellm = model.startswith("gemini/")
    return LLM(model=model, temperature=0.3, num_retries=8, is_litellm=force_litellm)


def _compute_all_data(csv_path: str, forecast_periods: int) -> dict:
    """Runs the full deterministic pipeline in Python: summary stats, a forecast,
    and anomaly detection for every numeric column. No LLM involved anywhere here."""
    summary = _summarize_dataset(csv_path)
    numeric_cols = summary["numeric_columns"]

    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    forecasts = {}
    anomalies = {}
    for col in numeric_cols:
        f = _forecast_series(df[col], periods=forecast_periods)
        f["column"] = col
        f["last_date"] = str(df["date"].max().date())
        forecasts[col] = f

        a = _detect_anomalies(df[col])
        a["date"] = df["date"].dt.strftime("%Y-%m-%d")
        flagged = a[a["is_anomaly"]]
        anomalies[col] = [
            {
                "date": row["date"],
                "value": round(float(row["value"]), 2),
                "z_score": float(row["z_score"]),
                "votes_out_of_3": int(row["votes"]),
            }
            for _, row in flagged.iterrows()
        ]

    return {"summary": summary, "forecasts": forecasts, "anomalies": anomalies}


def build_crew(csv_path: str, forecast_periods: int = 14) -> Crew:
    llm = get_llm()
    data = _compute_all_data(csv_path, forecast_periods)

    data_analyst = Agent(
        role="Business Data Analyst",
        goal="Present accurate ground-truth statistics about the dataset clearly and concisely.",
        backstory=(
            "A meticulous analyst who reports exactly what the numbers say -- never "
            "more, never less, never estimated."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    forecaster = Agent(
        role="Forecasting Specialist",
        goal="Explain statistically grounded forecasts for each key business metric and what's driving the trend.",
        backstory=(
            "A time-series specialist who explains forecasts clearly but never alters "
            "or invents a single number from what the model actually produced."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    anomaly_detective = Agent(
        role="Anomaly Detective",
        goal="Explain unusual movements in the business metrics using the confirmed detection results.",
        backstory=(
            "A former fraud-analytics investigator who only reports anomalies that were "
            "actually confirmed by statistical detection -- never a guess."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Insights & Reporting Lead",
        goal="Synthesize the analyst's, forecaster's, and detective's findings into a clear, prioritized executive report.",
        backstory=(
            "A sharp communicator who turns dense analysis into a report a CEO can read "
            "in two minutes and act on immediately. Never invents findings not given to them."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task_summarize = Task(
        description=(
            "Here is the ground-truth summary of the dataset, already computed -- do not "
            "recompute or alter any figure, just present it clearly:\n\n"
            f"{json.dumps(data['summary'], indent=2)}\n\n"
            "Report the date range, row count, and per-column stats (mean, std, min, max, "
            "last value, 7-day % change) in a clear, readable format (e.g. a table)."
        ),
        expected_output=(
            "A concise ground-truth summary of the dataset: date range, row count, and a "
            "per-column stats table, using only the numbers given above."
        ),
        agent=data_analyst,
    )

    task_forecast = Task(
        description=(
            "Here are the pre-computed 14-day forecasts for every numeric column in the "
            "dataset -- do not recompute or alter any figure, just explain them:\n\n"
            f"{json.dumps(data['forecasts'], indent=2)}\n\n"
            "For each metric report: the forecast method used, trend direction (up/down/flat), "
            "percent change vs the recent average, and the forecast range (min to max across "
            "the horizon, from forecast_values)."
        ),
        expected_output=(
            "A per-metric forecast summary listing trend direction, % change vs recent average, "
            "and the forecast value range, for every numeric column, using only the numbers given above."
        ),
        agent=forecaster,
        context=[task_summarize],
    )

    task_anomalies = Task(
        description=(
            "Here are the pre-computed, statistically confirmed anomalies for every numeric "
            "column in the dataset -- do not invent, alter, or add any anomaly not listed here:\n\n"
            f"{json.dumps(data['anomalies'], indent=2)}\n\n"
            "For each anomaly found, report the exact date, value, and how many of the 3 "
            "detection methods voted for it. If a column's list is empty, state clearly that "
            "no anomalies were found for it."
        ),
        expected_output=(
            "A list of confirmed anomalies grouped by metric, each with date, value, and vote "
            "count, or a clear statement that no anomalies were found for a given metric -- "
            "using only the data given above."
        ),
        agent=anomaly_detective,
        context=[task_summarize],
    )

    task_report = Task(
        description=(
            "Synthesize the dataset summary, forecasts, and anomalies into a single executive report "
            "with these sections: "
            "1) Executive Summary (3-4 sentences), "
            "2) Key Metrics & Trends (per metric: current state + forecast direction), "
            "3) Anomalies & Root-Cause Hypotheses (only for confirmed anomalies -- propose plausible "
            "business explanations, e.g. 'flash sale', 'outage', 'pricing change', clearly labeled as hypotheses), "
            "4) Recommended Actions (concrete, prioritized). "
            "Ground every claim in the analyst/forecaster/detective outputs -- do not invent data."
        ),
        expected_output="A polished, well-formatted markdown executive report following the 4 sections described.",
        agent=reporter,
        context=[task_summarize, task_forecast, task_anomalies],
    )

    crew = Crew(
        agents=[data_analyst, forecaster, anomaly_detective, reporter],
        tasks=[task_summarize, task_forecast, task_anomalies, task_report],
        process=Process.sequential,
        verbose=True,
        max_rpm=20,
    )
    return crew