"""Fraud review dashboard: threshold trade-offs and per-transaction explanations.

Run after training:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fraud.data import FEATURE_COLUMNS, TARGET, add_features, load_raw, time_split  # noqa: E402
from fraud.explain import shap_values, top_reasons  # noqa: E402
from fraud.metrics import evaluate  # noqa: E402

MODEL_DIR = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

st.set_page_config(page_title="Fraud review dashboard", layout="wide")


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgboost.json")
    config = json.loads((MODEL_DIR / "config.json").read_text())
    return model, config


@st.cache_data
def score_test_period(_model) -> pd.DataFrame:
    test = time_split(add_features(load_raw()))["test"].reset_index(drop=True)
    return test.assign(score=_model.predict_proba(test[FEATURE_COLUMNS])[:, 1])


if not (MODEL_DIR / "xgboost.json").exists():
    st.error("No trained model found. Run `python scripts/download_data.py` and `python -m fraud.train` first.")
    st.stop()

model, config = load_model()
test = score_test_period(model)

st.title("Fraud review dashboard")
st.caption("XGBoost scores for the held-out test period: the latest 20% of transactions, never seen in training.")

st.sidebar.header("Alert policy")
threshold = st.sidebar.slider(
    "Alert threshold", 0.01, 0.99, float(round(config["xgboost_threshold"], 2)), 0.01,
    help="Default is the F1-best threshold chosen on the validation period.",
)
review_cost = st.sidebar.number_input("Cost to review one alert", min_value=0.0, value=5.0, step=1.0)

m = evaluate(test[TARGET], test["score"], threshold, test["Amount"])
missed_value = test.loc[(test[TARGET] == 1) & (test["score"] < threshold), "Amount"].sum()
total_cost = missed_value + review_cost * (m["tp"] + m["fp"])

cols = st.columns(5)
cols[0].metric("Frauds caught", f"{m['tp']} / {m['tp'] + m['fn']}")
cols[1].metric("Alert precision", f"{m['precision']:.1%}")
cols[2].metric("False alarms", f"{m['fp']:,}")
cols[3].metric("Fraud value caught", f"{m['fraud_amount_caught_pct']:.1%}")
cols[4].metric("Missed fraud + review cost", f"{total_cost:,.0f}")

st.subheader("Why was this transaction flagged?")
flagged = test[test["score"] >= threshold].sort_values("score", ascending=False)
if flagged.empty:
    st.info("No transactions are above this threshold.")
else:
    choice = st.selectbox(
        f"{len(flagged):,} flagged transactions, highest score first",
        flagged.index,
        format_func=lambda i: (
            f"#{i} | score {flagged.at[i, 'score']:.3f} | amount {flagged.at[i, 'Amount']:,.2f} | "
            f"{'fraud' if flagged.at[i, TARGET] else 'genuine'}"
        ),
    )
    shap_row, _ = shap_values(model, test.loc[[choice], FEATURE_COLUMNS])
    reasons = pd.DataFrame(top_reasons(shap_row.iloc[0], k=8), columns=["feature", "SHAP value"])
    chart = alt.Chart(reasons).mark_bar().encode(
        x=alt.X("SHAP value:Q"),
        y=alt.Y("feature:N", sort=list(reasons["feature"]), title=None),
        color=alt.condition("datum['SHAP value'] > 0", alt.value("#c0392b"), alt.value("#2a6f97")),
    )
    st.altair_chart(chart)
    st.caption(
        "Red bars push the score towards fraud, blue bars away from it. V1-V28 are anonymised PCA components released by "
        "the data provider, so reasons name components rather than raw fields such as merchant or location."
    )

left, right = st.columns(2)
for col, name, title in [
    (left, "pr_curve.png", "Precision-recall, test period"),
    (right, "shap_importance.png", "Global feature importance (SHAP)"),
]:
    if (FIGURES / name).exists():
        col.image(str(FIGURES / name), caption=title)
