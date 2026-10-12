"""Streamlit dashboard over the reports the pipeline writes.

Run the pipeline first, then:
    uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_risk_monitor import db

REPORTS = Path(__file__).resolve().parents[1] / "reports"

st.set_page_config(page_title="Credit Card Portfolio Risk Monitor", layout="wide")
st.title("Credit Card Portfolio Risk Monitor")
st.caption("UCI Default of Credit Card Clients (Taiwan, 2005). Amounts in NT$. LGD and CCF are assumptions.")

if not (REPORTS / "metrics.json").exists():
    st.error("No reports found. Run `uv run credit-risk-monitor` first.")
    st.stop()

metrics = json.loads((REPORTS / "metrics.json").read_text())
facts = json.loads((REPORTS / "facts.json").read_text())
card = metrics["scorecard_holdout"]
gbm = metrics["gradient_boosting_holdout"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scorecard Gini (holdout)", f"{card['gini']:.3f}")
c2.metric("Scorecard KS (holdout)", f"{card['ks']:.3f}")
c3.metric("PSI dev vs holdout", f"{metrics['psi_dev_vs_holdout']:.4f}")
c4.metric("Expected loss, baseline", f"NT${facts['expected_loss_baseline']:,}")

tab_portfolio, tab_model, tab_stress, tab_watch = st.tabs(
    ["Portfolio", "Model", "Stress test", "Watchlist"]
)

with tab_portfolio:
    grades = pd.read_csv(REPORTS / "grade_summary.csv")
    st.subheader("Risk grades")
    st.dataframe(grades, hide_index=True, width="stretch")
    st.bar_chart(grades.set_index("grade")[["expected_loss"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Delinquency buckets")
        st.dataframe(pd.read_csv(REPORTS / "delinquency_buckets.csv"), hide_index=True)
    with right:
        st.subheader("Roll rates (month 2 to month 1)")
        st.dataframe(pd.read_csv(REPORTS / "roll_rates.csv", index_col=0).style.format("{:.1%}"))

    st.subheader("Utilization bands (SQL view)")
    st.dataframe(db.query(REPORTS / "portfolio.db", "SELECT * FROM v_utilization_bands"), hide_index=True)

    st.subheader("Commentary")
    st.markdown((REPORTS / "commentary.md").read_text().split("-->", 1)[-1])

with tab_model:
    st.subheader("Scorecard vs gradient boosting (holdout)")
    st.dataframe(
        pd.DataFrame({"scorecard": card, "gradient_boosting": gbm}).T.round(3),
        width="stretch",
    )
    st.write("Features dropped for a wrong-direction coefficient:", ", ".join(metrics["dropped_wrong_sign"]) or "none")
    st.subheader("Information value")
    st.bar_chart(pd.read_csv(REPORTS / "information_value.csv").set_index("feature"))
    st.subheader("Calibration by decile")
    calib = pd.read_csv(REPORTS / "calibration.csv").set_index("decile")
    st.line_chart(calib[["predicted_pd", "observed_default_rate"]])
    st.subheader("Scorecard points")
    st.dataframe(pd.read_csv(REPORTS / "scorecard_points.csv"), hide_index=True, width="stretch")

with tab_stress:
    stress = pd.read_csv(REPORTS / "stress_test.csv")
    st.dataframe(stress, hide_index=True, width="stretch")
    st.bar_chart(stress.set_index("scenario")[["expected_loss"]])

with tab_watch:
    st.subheader("Grade D and E accounts not yet in default (SQL view)")
    watch = db.query(REPORTS / "portfolio.db", "SELECT * FROM v_high_risk_watchlist LIMIT 200")
    st.dataframe(watch, hide_index=True, width="stretch")
