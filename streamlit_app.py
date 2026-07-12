"""
Streamlit dashboard for the Multi-Agent Forecasting & Anomaly Detection System.

Run locally:    streamlit run streamlit_app.py
Deploy free:    push to GitHub -> share.streamlit.io -> point at this file ->
                add your API key under the app's Settings > Secrets, e.g.:
                    GEMINI_API_KEY = "AQ...."
                    MODEL = "gemini/gemini-flash-latest"
"""
import sys
import os
import io
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tools.forecasting_tools import _forecast_series
from tools.anomaly_tools import _detect_anomalies
from tools.data_tools import _summarize_dataset

# --- Bridge Streamlit Cloud secrets into env vars so src/agents/crew_setup.py
# (which reads os.getenv) works unchanged whether run locally (.env) or deployed
# (st.secrets). Safe no-op locally if no secrets.toml exists. ---
for _key in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MODEL"):
    try:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = st.secrets[_key]
    except Exception:
        pass  # no secrets.toml present (e.g. local run without cloud secrets) -- fine

st.set_page_config(
    page_title="Forecast & Anomaly Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================== STYLING ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
    }
    .hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 0.3rem 0; letter-spacing: -0.02em; }
    .hero p { font-size: 0.95rem; color: #94a3b8; margin: 0; }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
    div[data-testid="stMetricValue"] { font-weight: 800; color: #0f172a; }

    .anomaly-card {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .anomaly-card b { color: #991b1b; }

    .no-anomaly-card {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #166534;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; }

    div[data-testid="stExpander"] { border-radius: 12px; border: 1px solid #e2e8f0; }

    .footer-note { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

TREND_COLORS = {"up": "#16a34a", "down": "#dc2626", "flat": "#64748b"}
TREND_ARROWS = {"up": "▲", "down": "▼", "flat": "▬"}

# ============================== HEADER ==============================
st.markdown(f"""
<div class="hero">
    <h1>📈 Forecast &amp; Anomaly Intelligence</h1>
    <p>Multi-agent system &middot; statistically grounded forecasting &middot; ensemble anomaly detection &middot; last run {datetime.now().strftime('%b %d, %Y %H:%M')}</p>
</div>
""", unsafe_allow_html=True)

# ============================== SIDEBAR ==============================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    data_source = st.radio("Data source", ["Sample dataset", "Upload your own CSV"], index=0)

    if data_source == "Upload your own CSV":
        uploaded = st.file_uploader("CSV with a 'date' column + numeric metrics", type=["csv"])
        if uploaded is not None:
            tmp_path = Path("uploaded_data.csv")
            tmp_path.write_bytes(uploaded.getvalue())
            csv_path = str(tmp_path)
        else:
            csv_path = None
    else:
        csv_path = str(Path(__file__).resolve().parent / "data" / "sample_business_metrics.csv")

    st.markdown("---")
    periods = st.slider("Forecast horizon (days)", min_value=7, max_value=60, value=14, step=1)

    st.markdown("---")
    st.markdown("### 🤖 Executive Report")
    has_key = any(os.getenv(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    if has_key:
        st.success("LLM configured ✓")
    else:
        st.warning("No LLM key found. Set one in `.env` (local) or app Secrets (cloud) to enable the AI-written executive report.")

# ============================== DATA PIPELINE (cached) ==============================
@st.cache_data(show_spinner=False)
def load_and_analyze(csv_path: str, periods: int, file_hash: str):
    """file_hash busts the cache when the underlying CSV content changes."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    summary = _summarize_dataset(csv_path)
    numeric_cols = summary["numeric_columns"]

    results = {}
    for col in numeric_cols:
        forecast = _forecast_series(df[col], periods=periods)
        anomalies_df = _detect_anomalies(df[col])
        anomalies_df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        flagged = anomalies_df[anomalies_df["is_anomaly"]]
        results[col] = {"forecast": forecast, "anomalies": flagged}
    return df, summary, results


if not csv_path or not Path(csv_path).exists():
    st.info("👈 Upload a CSV to get started, or switch to the sample dataset in the sidebar.")
    st.stop()

file_bytes = Path(csv_path).read_bytes()
file_hash = hashlib.md5(file_bytes).hexdigest()

with st.spinner("Running forecasting & anomaly detection..."):
    df, summary, results = load_and_analyze(csv_path, periods, file_hash)

numeric_cols = summary["numeric_columns"]
total_anomalies = sum(len(r["anomalies"]) for r in results.values())

# ============================== KPI ROW ==============================
st.markdown(f"**{summary['row_count']} rows** &middot; **{summary['date_range']['start']} to {summary['date_range']['end']}** &middot; **{total_anomalies} anomalies flagged** across {len(numeric_cols)} metrics")

kpi_cols = st.columns(len(numeric_cols))
for i, col in enumerate(numeric_cols):
    stats = summary["column_stats"][col]
    f = results[col]["forecast"]
    arrow = TREND_ARROWS[f["trend_direction"]]
    with kpi_cols[i]:
        st.metric(
            label=col.replace("_", " ").title(),
            value=f"{stats['last_value']:,.2f}",
            delta=f"{arrow} {f['pct_change_vs_recent']:+.1f}% forecast",
        )

st.markdown("<br>", unsafe_allow_html=True)

# ============================== TABS ==============================
tab_forecast, tab_anomaly, tab_report, tab_data = st.tabs(
    ["📊 Forecasts", "🚨 Anomalies", "🧠 Executive Report", "📋 Raw Data"]
)

# --- TAB: Forecasts ---
with tab_forecast:
    for col in numeric_cols:
        f = results[col]["forecast"]
        flagged = results[col]["anomalies"]
        color = TREND_COLORS[f["trend_direction"]]

        fig = make_subplots()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines", name="History",
            line=dict(color="#2563eb", width=1.6),
        ))
        if not flagged.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(flagged["date"]), y=flagged["value"], mode="markers",
                name="Anomaly", marker=dict(color="#dc2626", size=9, symbol="circle"),
            ))
        future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=periods)
        fig.add_trace(go.Scatter(
            x=future_dates, y=f["upper_bound_95"], mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=f["lower_bound_95"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(22, 163, 74, 0.15)", name="95% CI", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=f["forecast_values"], mode="lines", name="Forecast",
            line=dict(color=color, width=2, dash="dash"),
        ))

        fig.update_layout(
            title=dict(text=f"{col.replace('_', ' ').title()} — {f['trend_direction'].upper()} trend ({f['pct_change_vs_recent']:+.1f}%)", font=dict(size=15)),
            height=340, margin=dict(l=10, r=10, t=45, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB: Anomalies ---
with tab_anomaly:
    for col in numeric_cols:
        flagged = results[col]["anomalies"]
        st.markdown(f"#### {col.replace('_', ' ').title()}")
        if flagged.empty:
            st.markdown('<div class="no-anomaly-card">✓ No anomalies detected for this metric.</div>', unsafe_allow_html=True)
        else:
            for _, row in flagged.iterrows():
                direction = "spike" if row["z_score"] > 0 else "drop"
                st.markdown(
                    f'<div class="anomaly-card"><b>{row["date"]}</b> — value <b>{row["value"]:,.2f}</b> '
                    f'({direction}, z-score {row["z_score"]:.2f}, {row["votes"]}/3 detection methods agreed)</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

# --- TAB: Executive Report ---
with tab_report:
    if not has_key:
        st.info("Add an API key (GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY) in `.env` or your deployment's Secrets to enable this.")
    else:
        if st.button("🧠 Generate AI Executive Report", type="primary"):
            with st.spinner("Running the 4-agent crew (Data Analyst → Forecaster → Anomaly Detective → Reporter)..."):
                try:
                    from agents.crew_setup import build_crew
                    crew = build_crew(csv_path, forecast_periods=periods)
                    result = crew.kickoff()
                    st.session_state["report"] = str(result)
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if "report" in st.session_state:
            st.markdown(st.session_state["report"])
            st.download_button(
                "⬇️ Download report (.md)",
                data=st.session_state["report"],
                file_name="executive_report.md",
                mime="text/markdown",
            )

# --- TAB: Raw Data ---
with tab_data:
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download data (.csv)",
        data=df.to_csv(index=False),
        file_name="business_metrics.csv",
        mime="text/csv",
    )

st.markdown('<div class="footer-note">Multi-Agent Forecasting &amp; Anomaly Detection System &middot; forecasts via Holt-Winters &middot; anomalies via z-score + IQR + Isolation Forest ensemble</div>', unsafe_allow_html=True)
