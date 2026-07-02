# Multi-Agent Forecasting & Anomaly Detection System

A four-agent [CrewAI](https://docs.crewai.com/) system for business metrics
(revenue, units sold, active customers, churn, etc.) that forecasts trends and
flags anomalies — grounded in real statistics, not LLM guesses.

## How it works

```
             ┌─────────────────────┐
   CSV  ──▶  │  Data Analyst Agent │  summarize_dataset tool
             └──────────┬──────────┘
                         │ (ground-truth stats)
        ┌────────────────┴────────────────┐
        ▼                                  ▼
┌───────────────────┐            ┌────────────────────┐
│ Forecasting Agent  │            │ Anomaly Detective   │
│ forecast_metric     │           │ detect_anomalies     │
│ (Holt-Winters /     │           │ (z-score + IQR +     │
│  linear fallback)   │           │  Isolation Forest    │
└──────────┬──────────┘           │  ensemble vote)      │
           │                      └──────────┬───────────┘
           └───────────────┬──────────────────┘
                            ▼
                 ┌────────────────────┐
                 │ Insights Reporter  │  synthesizes into
                 │                     │  executive markdown
                 └────────────────────┘
```

**Key design choice:** the LLM agents never do arithmetic themselves. Every
number in the final report traces back to a deterministic tool call
(statsmodels for forecasting, an ensemble of z-score/IQR/Isolation Forest for
anomaly detection). Agents orchestrate, interpret, and narrate — they don't
invent numbers.

## Project structure

```
forecast_system/
├── data/
│   └── sample_business_metrics.csv   # synthetic demo data (generated)
├── outputs/                          # charts + executive_report.md land here
├── src/
│   ├── agents/
│   │   └── crew_setup.py             # Agent/Task/Crew definitions
│   ├── data/
│   │   └── generate_sample_data.py   # synthetic data generator
│   ├── tools/
│   │   ├── data_tools.py             # summarize_dataset
│   │   ├── forecasting_tools.py      # forecast_metric (Holt-Winters)
│   │   └── anomaly_tools.py          # detect_anomalies (z-score+IQR+IsoForest)
│   └── main.py                       # CLI entry point
├── .env.example
└── requirements.txt
```

## Setup

```bash
cd forecast_system
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY + MODEL=openai/gpt-4o)
```

## Usage

**Quick check — deterministic pipeline only (no API key needed):**
```bash
python src/main.py --csv data/sample_business_metrics.csv --periods 14 --no-llm
```
This prints forecast/anomaly stats and saves a chart per metric to `outputs/`.

**Full run — with the agent crew and narrative executive report:**
```bash
python src/main.py --csv data/sample_business_metrics.csv --periods 14
```
This additionally writes `outputs/executive_report.md` with an executive
summary, per-metric trends, anomaly root-cause hypotheses, and recommended
actions.

**Bring your own data:** any CSV with a `date` column plus one or more
numeric metric columns works —
```bash
python src/main.py --csv path/to/your_metrics.csv --periods 30
```

**Regenerate the synthetic sample data:**
```bash
python src/data/generate_sample_data.py
```

## Testing

```bash
pytest tests/ -v
```
9 unit tests cover: data generation integrity, forecast output shape/bounds,
fallback behavior on short series, trend detection, and anomaly detection
accuracy against known injected anomalies. No API key required — these test
the deterministic core only.

## CI/CD

- **`.github/workflows/ci.yml`** — runs on every push/PR: installs deps, runs
  `pytest`, and smoke-tests the CLI in `--no-llm` mode. No secrets required.
- **`.github/workflows/run-report.yml`** — optional, manually-triggered (or
  schedule it) workflow that runs the *full* LLM agent crew and commits the
  resulting report/charts back to the repo. Requires an `ANTHROPIC_API_KEY`
  repo secret. Since this spends real API credits, it's manual-trigger-only
  by default.

## Extending it

- **More agents:** add a "Root Cause Investigator" that cross-references
  anomalies across metrics (e.g. did a churn spike coincide with a revenue
  drop?), or a "Budget/Target Agent" that compares forecasts to goals.
- **More tools:** swap in Prophet or a neural forecaster; add a Slack/email
  tool so the Reporter can push alerts directly.
- **Scheduling:** wrap `main.py` in a cron job or Airflow DAG for daily runs.
- **Dashboard:** the `outputs/*.png` charts + `executive_report.md` can be
  fed into a simple Streamlit or web dashboard for a live view.
