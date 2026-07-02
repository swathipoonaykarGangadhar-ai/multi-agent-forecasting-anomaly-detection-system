"""
Defines the four-agent crew:

  1. Data Analyst      - grounds the crew in real dataset stats
  2. Forecasting Agent  - projects each metric forward using forecast_metric
  3. Anomaly Detective   - flags historical anomalies using detect_anomalies
  4. Insights Reporter   - synthesizes everything into a business-readable report

All numeric grounding comes from deterministic tools (statsmodels / sklearn),
not from the LLM -- the agents' job is retrieval-orchestration + narrative
synthesis, not arithmetic.
"""
import os

from crewai import Agent, Task, Crew, Process, LLM

from tools.data_tools import summarize_dataset
from tools.forecasting_tools import forecast_metric
from tools.anomaly_tools import detect_anomalies


def get_llm() -> LLM:
    """
    Configure the LLM via LiteLLM model strings so you can swap providers
    with one env var. Examples:
      MODEL=anthropic/claude-sonnet-4-6   (needs ANTHROPIC_API_KEY)
      MODEL=openai/gpt-4o                 (needs OPENAI_API_KEY)
    """
    model = os.getenv("MODEL", "anthropic/claude-sonnet-4-6")
    return LLM(model=model, temperature=0.3)


def build_crew(csv_path: str, forecast_periods: int = 14) -> Crew:
    llm = get_llm()

    data_analyst = Agent(
        role="Business Data Analyst",
        goal="Establish accurate ground-truth statistics about the dataset before any analysis happens.",
        backstory=(
            "A meticulous analyst who never lets a forecast or anomaly claim go out "
            "without first checking the raw numbers. Distrustful of assumptions."
        ),
        tools=[summarize_dataset],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    forecaster = Agent(
        role="Forecasting Specialist",
        goal="Produce statistically grounded forecasts for each key business metric and explain what's driving the trend.",
        backstory=(
            "A time-series specialist who always calls the forecasting tool for numbers "
            "and never estimates a projection from memory or vibes."
        ),
        tools=[forecast_metric],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    anomaly_detective = Agent(
        role="Anomaly Detective",
        goal="Identify and explain unusual movements in the business metrics using rigorous statistical detection, not guesswork.",
        backstory=(
            "A former fraud-analytics investigator who treats every spike and dip as a "
            "lead to run down. Only reports anomalies the detection tool actually confirms."
        ),
        tools=[detect_anomalies],
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

    numeric_hint = (
        "The dataset's numeric columns typically include revenue, units_sold, "
        "active_customers, and churn_rate -- but always confirm via the summarize_dataset tool first, "
        "since the actual CSV may differ."
    )

    task_summarize = Task(
        description=(
            f"Load and summarize the dataset at '{csv_path}' using the summarize_dataset tool. "
            f"{numeric_hint} Report the date range, row count, and per-column stats "
            "(mean, std, min, max, last value, 7-day % change) clearly."
        ),
        expected_output=(
            "A concise ground-truth summary of the dataset: date range, row count, and a "
            "per-column stats table, based only on the tool's actual output."
        ),
        agent=data_analyst,
    )

    task_forecast = Task(
        description=(
            f"Using the forecast_metric tool, forecast the next {forecast_periods} days for EVERY numeric "
            f"column identified by the Data Analyst (call the tool once per column, csv_path='{csv_path}'). "
            "For each metric report: the forecast method used, trend direction (up/down/flat), "
            "percent change vs the recent average, and the forecast range (min to max across the horizon). "
            "Do not fabricate numbers -- only report what the tool returns."
        ),
        expected_output=(
            "A per-metric forecast summary listing trend direction, % change vs recent average, "
            "and the forecast value range, for every numeric column."
        ),
        agent=forecaster,
        context=[task_summarize],
    )

    task_anomalies = Task(
        description=(
            f"Using the detect_anomalies tool, check EVERY numeric column identified by the Data Analyst "
            f"for anomalies (call the tool once per column, csv_path='{csv_path}'). "
            "For each anomaly found, report the exact date, value, and how many of the 3 detection "
            "methods voted for it. Only report anomalies the tool actually returns -- never invent dates."
        ),
        expected_output=(
            "A list of confirmed anomalies grouped by metric, each with date, value, and vote count, "
            "or a clear statement that no anomalies were found for a given metric."
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
    )
    return crew
