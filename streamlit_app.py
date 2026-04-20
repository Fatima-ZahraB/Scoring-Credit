import streamlit as st
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from monitoring.drift_report import (
    load_prediction_logs,
    compute_monitoring_metrics,
    get_score_distribution_data,
)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d1117; }
    .block-container { padding-top: 2rem; }

    .risk-card {
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
    }
    .risk-low    { background: #0d2b1a; border: 1px solid #2ea043; color: #2ea043; }
    .risk-medium { background: #2b200d; border: 1px solid #d29922; color: #d29922; }
    .risk-high   { background: #2b0d0d; border: 1px solid #f85149; color: #f85149; }

    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk")
    st.markdown("---")

    # API health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API Online")
        st.markdown(f"**Clients loaded:** {health.get('total_clients', '—')}")
        st.markdown(f"**Model:** {'✅' if health.get('model_loaded') else '❌'}")
        st.markdown(f"**Data:** {'✅' if health.get('data_loaded') else '❌'}")
    except Exception:
        st.error("API Offline")
        st.info(f"Expected at: `{API_URL}`")

    st.markdown("---")
    st.markdown("**v1.0.0** — Home Credit Default Risk")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Monitoring"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Client Default Risk Prediction")
    st.markdown("Enter a client ID to get their probability of default.")

    col_input, col_spacer = st.columns([1, 2])
    with col_input:
        sk_id = st.number_input(
            "SK_ID_CURR",
            min_value=1,
            step=1,
            value=None,
            placeholder="e.g. 100001",
            help="Unique client identifier from the deployed dataset",
        )
        predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")

    if predict_btn:
        if sk_id is None:
            st.warning("Please enter a valid SK_ID_CURR.")
        else:
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"SK_ID_CURR": int(sk_id)},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        proba      = data["probability_default"]
                        risk       = data["risk_level"]
                        rec        = data["recommendation"]
                        latency    = data["inference_time_ms"]
                        timestamp  = data["timestamp"]

                        # Risk card
                        risk_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[risk]
                        risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}[risk]

                        st.markdown(f"""
                        <div class="risk-card {risk_class}">
                            <div style="font-size:2.5rem">{risk_emoji}</div>
                            <div style="font-size:1.2rem;margin:0.5rem 0">{risk} RISK</div>
                            <div style="font-size:3rem;font-weight:600">{proba:.1%}</div>
                            <div style="font-size:0.9rem;opacity:0.8">{rec}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{sk_id}</div>
                                <div class="metric-label">SK_ID_CURR</div>
                            </div>""", unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{latency:.1f}ms</div>
                                <div class="metric-label">Inference Time</div>
                            </div>""", unsafe_allow_html=True)
                        with m3:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{timestamp[11:19]}</div>
                                <div class="metric-label">Timestamp</div>
                            </div>""", unsafe_allow_html=True)

                    elif resp.status_code == 404:
                        st.error(f"Client not found: SK_ID_CURR {sk_id} is not in the deployed dataset.")
                    else:
                        st.error(f"API error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure it's running.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Monitoring Dashboard")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    logs = load_prediction_logs()

    if logs.empty:
        st.info("No predictions logged yet. Run some predictions first.")
    else:
        metrics = compute_monitoring_metrics(logs)

        # ── KPI row ──────────────────────────────────────────────────────────
        st.markdown("### Key Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)

        kpis = [
            (k1, f"{metrics['total_predictions']}", "Total Predictions"),
            (k2, f"{metrics['mean_proba']:.3f}",    "Mean Default Proba"),
            (k3, f"{metrics['pct_high_risk']:.1%}", "High Risk Rate"),
            (k4, f"{metrics['mean_latency_ms']:.1f}ms", "Avg Latency"),
            (k5, f"{metrics['p95_latency_ms']:.1f}ms", "P95 Latency"),
        ]
        for col, val, label in kpis:
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row ───────────────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Score Distribution")
            dist = get_score_distribution_data(logs)
            if dist["bins"]:
                chart_df = pd.DataFrame({
                    "Probability": dist["bins"],
                    "Count": dist["counts"],
                })
                st.bar_chart(chart_df.set_index("Probability"))

        with col_right:
            st.markdown("### Risk Level Breakdown")
            risk_counts = logs["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            st.bar_chart(risk_counts.set_index("Risk Level"))

        # ── Latency over time ─────────────────────────────────────────────────
        st.markdown("### Latency Over Time")
        latency_df = logs[["timestamp", "inference_time_ms"]].dropna()
        latency_df = latency_df.set_index("timestamp")
        st.line_chart(latency_df)

        # ── Raw logs table ────────────────────────────────────────────────────
        st.markdown("### Recent Predictions")
        display_cols = ["timestamp", "SK_ID_CURR", "probability_default", "risk_level", "inference_time_ms", "status"]
        available = [c for c in display_cols if c in logs.columns]
        st.dataframe(
            logs[available].sort_values("timestamp", ascending=False).head(50),
            use_container_width=True,
        )

        # ── Evidently report ─────────────────────────────────────────────────
        st.markdown("### Data Drift Report")
        report_path = Path(__file__).parent / "logs" / "drift_report.html"
        if report_path.exists():
            with open(report_path, "r") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
        else:
            st.info(
                "No drift report generated yet. "
                "Run `python monitoring/drift_report.py` to generate one."
            )
