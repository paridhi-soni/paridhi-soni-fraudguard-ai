"""
FraudGuard AI - Streamlit dashboard
Tabs:
  1. Predict Transaction - single-transaction fraud scoring with SHAP explanation
  2. Model Performance   - saved evaluation metrics & plots
  3. Batch Analysis      - CSV upload, batch scoring, flagged-transaction report
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from explain import explain_single, get_explainer  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    explainer = get_explainer(model)
    return model, scaler, feature_cols, explainer


def resources_available() -> bool:
    return all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_feature_row(feature_cols, amount, time_val, v_values: dict, scaler):
    """Build a single-row DataFrame matching the model's training columns."""
    raw = {col: 0.0 for col in feature_cols}
    for i, val in v_values.items():
        col = f"V{i}"
        if col in raw:
            raw[col] = val
    raw["Amount"] = amount
    raw["Time"] = time_val

    df = pd.DataFrame([raw])[feature_cols]

    scaled_amount_time = scaler.transform(df[["Amount", "Time"]])
    df[["Amount", "Time"]] = scaled_amount_time
    return df


def gauge_color(prob: float) -> str:
    if prob < 0.30:
        return "#2ecc71"  # green
    if prob < 0.70:
        return "#f1c40f"  # yellow
    return "#e74c3c"  # red


def render_gauge(prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Fraud Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color(prob)},
            "steps": [
                {"range": [0, 30], "color": "#eafaf1"},
                {"range": [30, 70], "color": "#fef9e7"},
                {"range": [70, 100], "color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def seconds_to_hour(time_val: float) -> int:
    return int((time_val % 86400) // 3600)


def plain_english_explanation(top_features: dict, amount: float, time_val: float) -> str:
    hour = seconds_to_hour(time_val)
    reasons = []

    fraud_drivers = {k: v for k, v in top_features.items() if v["direction"] == "fraud"}
    sorted_drivers = sorted(fraud_drivers.items(), key=lambda kv: abs(kv[1]["shap"]), reverse=True)

    if amount >= 800 or amount <= 2:
        reasons.append(f"unusual amount (${amount:,.2f})")
    if hour < 5:
        reasons.append(f"unusual hour ({hour:02d}:00)")

    for feat, info in sorted_drivers:
        if feat in ("Amount", "Time"):
            continue
        reasons.append(f"anomalous pattern in {feat} (value={info['value']:.2f})")
        if len(reasons) >= 4:
            break

    if not reasons:
        return "This transaction looks consistent with typical legitimate spending patterns."

    return "This transaction was flagged because: " + ", ".join(reasons) + "."


def render_shap_bar(top_features: dict):
    items = sorted(top_features.items(), key=lambda kv: abs(kv[1]["shap"]))
    features = [k for k, _ in items]
    values = [v["shap"] for _, v in items]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 5 Feature Contributions (red = towards fraud, green = towards legit)",
        xaxis_title="SHAP value",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.fg-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #7f1d1d 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 18px; color: #fff;
    box-shadow: 0 8px 24px rgba(0,0,0,.25);
}
.fg-hero h1 { margin: 0; font-size: 2.2rem; color: #fff; }
.fg-hero p  { margin: 4px 0 18px; opacity: .8; }
.fg-stats { display: flex; gap: 14px; flex-wrap: wrap; }
.fg-stat {
    flex: 1 1 150px; background: rgba(255,255,255,.1); border-radius: 12px;
    padding: 12px 16px; backdrop-filter: blur(4px);
}
.fg-stat .num { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.fg-stat .lbl { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; opacity: .75; }
.fg-stat.red .num { color: #ff6b6b; }

@keyframes fg-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(231,76,60,.75); border-color: #e74c3c; }
    70%  { box-shadow: 0 0 0 18px rgba(231,76,60,0); border-color: #ff9d94; }
    100% { box-shadow: 0 0 0 0 rgba(231,76,60,0);   border-color: #e74c3c; }
}
.fg-fraud-alert {
    border: 4px solid #e74c3c; border-radius: 14px; padding: 18px 22px;
    background: #fdedec; color: #922b21; animation: fg-pulse 1.3s infinite;
}
.fg-fraud-alert h2 { margin: 0; color: #c0392b; }
.fg-fraud-alert span { font-size: .95rem; }
.fg-legit-alert {
    border: 4px solid #2ecc71; border-radius: 14px; padding: 18px 22px;
    background: #eafaf1; color: #1e8449;
}
.fg-legit-alert h2 { margin: 0; color: #1e8449; }

.fg-feed-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; margin-bottom: 6px; border-radius: 10px;
    background: rgba(128,128,128,.08); border-left: 5px solid #999;
}
.fg-feed-row.fraud { border-left-color: #e74c3c; }
.fg-feed-row.legit { border-left-color: #2ecc71; }
.fg-badge { padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700; color: #fff; }
.fg-badge.fraud { background: #e74c3c; }
.fg-badge.legit { background: #27ae60; }
.fg-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#e74c3c;
          margin-right:6px; animation: fg-pulse 1.3s infinite; }

/* ---- Premium overrides ---- */
.stButton > button, .stDownloadButton > button {
    border-radius: 999px; border: none; padding: .6rem 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); color: #fff;
    box-shadow: 0 4px 14px rgba(37,99,235,.35); transition: transform .15s, box-shadow .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 20px rgba(124,58,237,.45); color: #fff;
}
.stButton > button:active { transform: translateY(0); }
[data-testid="stMetric"] {
    background: rgba(128,128,128,.06); border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px; padding: 14px 18px; box-shadow: 0 4px 12px rgba(0,0,0,.08);
}

.fg-hero3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 22px; }
@media (max-width: 800px) { .fg-hero3 { grid-template-columns: 1fr; } }
.fg-card {
    background: linear-gradient(145deg, #0b1220 0%, #16223b 100%); color: #fff;
    border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 22px 26px;
    box-shadow: 0 10px 28px rgba(0,0,0,.35); display: flex; align-items: center; gap: 18px;
}
.fg-card .ico {
    font-size: 2rem; width: 58px; height: 58px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
}
.fg-card .ico.blue  { background: rgba(59,130,246,.2);  box-shadow: 0 0 18px rgba(59,130,246,.45); }
.fg-card .ico.red   { background: rgba(239,68,68,.2);   box-shadow: 0 0 18px rgba(239,68,68,.45); }
.fg-card .ico.green { background: rgba(34,197,94,.2);   box-shadow: 0 0 18px rgba(34,197,94,.45); }
.fg-card .num { font-size: 2.3rem; font-weight: 800; line-height: 1.05; }
.fg-card .num.blue { color: #60a5fa; } .fg-card .num.red { color: #f87171; } .fg-card .num.green { color: #4ade80; }
.fg-card .lbl { font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; opacity: .7; }

.fg-banner {
    width: 100%; text-align: center; font-size: 2rem; font-weight: 800; letter-spacing: .02em;
    padding: 26px 16px; border-radius: 16px; color: #fff; margin: 6px 0 18px;
}
@keyframes fg-banner-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,.8), 0 0 30px rgba(239,68,68,.5); filter: brightness(1); }
    50%      { box-shadow: 0 0 0 22px rgba(239,68,68,0), 0 0 60px rgba(239,68,68,.9); filter: brightness(1.25); }
}
.fg-banner.fraud { background: linear-gradient(90deg, #991b1b, #dc2626, #991b1b); animation: fg-banner-pulse 1.2s infinite; }
.fg-banner.safe  { background: linear-gradient(90deg, #166534, #16a34a, #166534); box-shadow: 0 8px 24px rgba(22,163,74,.4); }
.fg-bigpct { text-align: center; font-size: 4.5rem; font-weight: 900; line-height: 1; margin: 4px 0 2px; }
.fg-bigpct.fraud { color: #ef4444; } .fg-bigpct.safe { color: #22c55e; }
.fg-bigpct-lbl { text-align: center; text-transform: uppercase; letter-spacing: .12em; opacity: .65; font-size: .8rem; }

.fg-mon-row {
    display: grid; grid-template-columns: 1.3fr 1fr 1fr .9fr 1.2fr; gap: 10px; align-items: center;
    padding: 12px 18px; margin-bottom: 8px; border-radius: 12px; color: #fff; font-weight: 600;
}
.fg-mon-row.fraud { background: linear-gradient(90deg, #7f1d1d, #b91c1c); }
.fg-mon-row.safe  { background: linear-gradient(90deg, #14532d, #15803d); }
.fg-mon-row.warn  { background: linear-gradient(90deg, #9a3412, #ea580c); }
.fg-mon-row .st { text-align: right; font-weight: 800; }

.fg-flagstat { text-align: center; padding: 22px; border-radius: 16px; margin: 10px 0 18px;
               border: 2px solid; background: rgba(128,128,128,.06); }
.fg-flagstat .num { font-size: 3.4rem; font-weight: 900; line-height: 1; }
.fg-flagstat.red   { border-color: #ef4444; } .fg-flagstat.red .num   { color: #ef4444; }
.fg-flagstat.green { border-color: #22c55e; } .fg-flagstat.green .num { color: #22c55e; }
.fg-flagstat .lbl { text-transform: uppercase; letter-spacing: .1em; opacity: .7; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def risk_tier(p: float) -> str:
    """green < 30%, orange 30-70%, red > 70%"""
    return "safe" if p < 0.30 else ("warn" if p <= 0.70 else "fraud")


def risk_dot(p: float) -> str:
    return {"safe": "🟢", "warn": "🟠", "fraud": "🔴"}[risk_tier(p)]


def describe_reason(feature: str, raw_value: float, shap_val: float, is_fraud: bool) -> str:
    """One human-readable bullet for a feature's contribution (raw_value is unscaled)."""
    if feature == "Amount":
        if is_fraud:
            if raw_value < 10:
                note = "unusually low amount (card testing pattern)"
            elif raw_value > 2000:
                note = "unusually high amount"
            else:
                note = "amount adds to the fraud risk"
        else:
            note = "normal range" if 10 <= raw_value <= 2000 else "amount is not a risk factor here"
        return f"Amount = ${raw_value:,.2f} → {note}"

    if feature == "Time":
        hour = seconds_to_hour(raw_value)
        if is_fraud:
            note = "unusual hour" if hour < 5 else "timing adds to the fraud risk"
        else:
            note = "normal business hours" if 8 <= hour < 20 else "typical timing"
        return f"Time = {hour:02d}:00 → {note}"

    if is_fraud:
        strength = "strongly pushes toward fraud" if abs(shap_val) >= 1.0 else "pushes toward fraud"
        return f"{feature} = {raw_value:+.1f} → anomalous pattern in feature {feature} ({strength})"
    return f"{feature} = {raw_value:+.1f} → typical behavior"


def feed_top_reasons(shap_row, raw_row, cols, is_fraud: bool, n: int = 3) -> list:
    """Top-n features by |SHAP|, restricted to those pushing the verdict's way when possible."""
    order = sorted(range(len(cols)), key=lambda i: abs(shap_row[i]), reverse=True)
    aligned = [i for i in order if (shap_row[i] > 0) == is_fraud]
    chosen = (aligned + [i for i in order if i not in aligned])[:n]
    return [describe_reason(cols[i], float(raw_row[cols[i]]), float(shap_row[i]), is_fraud)
            for i in chosen]


# Initialise all session-state keys before anything reads them
for _key, _default in {
    "analyzed_session": 0,
    "fraud_detected_session": 0,
}.items():
    st.session_state.setdefault(_key, _default)


@st.cache_data
def load_holdout():
    """Held-out test split (never seen in training) with Amount/Time back in raw units."""
    path = os.path.join(MODEL_DIR, "test_set.csv")
    if not (os.path.exists(path) and os.path.exists(SCALER_PATH)):
        return None
    df = pd.read_csv(path)
    scaler_ = joblib.load(SCALER_PATH)
    df[["Amount", "Time"]] = scaler_.inverse_transform(df[["Amount", "Time"]])
    return df


def render_hero():
    df = load_dataset()
    total = len(df) if df is not None else 0
    frauds = int(df["Class"].sum()) if df is not None else 0
    saved = float(df.loc[df["Class"] == 1, "Amount"].sum()) if df is not None else 0.0
    st.markdown(f"""
<div class="fg-hero3">
  <div class="fg-card"><div class="ico blue">📊</div>
    <div><div class="num blue">{total:,}</div><div class="lbl">Transactions Analyzed</div></div></div>
  <div class="fg-card"><div class="ico red">🚨</div>
    <div><div class="num red">{frauds:,}</div><div class="lbl">Frauds Detected</div></div></div>
  <div class="fg-card"><div class="ico green">💰</div>
    <div><div class="num green">${saved:,.0f}</div><div class="lbl">Saved</div></div></div>
</div>
""", unsafe_allow_html=True)


st.title("🛡️ FraudGuard AI")
st.caption("Explainable credit card fraud detection powered by XGBoost + SHAP")
render_hero()

if not resources_available():
    st.error(
        "Model artifacts not found. Please run the training pipeline first:\n\n"
        "```\npython generate_sample_data.py\npython src/train.py\npython src/evaluate.py\npython src/explain.py\n```"
    )
    st.stop()

model, scaler, feature_cols, explainer = load_resources()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Predict Transaction", "📊 Model Performance", "📁 Batch Analysis", "🔴 LIVE MONITOR"]
)

# ----------------------------------------------------------------------------
# TAB 1: Predict Transaction
# ----------------------------------------------------------------------------
with tab1:
    st.sidebar.header("Transaction Details")
    amount = st.sidebar.slider("Amount ($)", 0.0, 10000.0, 100.0, step=1.0)
    time_val = st.sidebar.slider("Time gap since previous transaction (seconds)", 0, 3600, 300, step=30)

    if time_val == 0:
        st.sidebar.caption("= just now")
    else:
        minutes = time_val / 60
        unit = "minute" if minutes == 1 else "minutes"
        st.sidebar.caption(f"= {minutes:g} {unit} ago")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**PCA Components (V1-V10)**")

    v_values = {}
    for i in range(1, 11):
        v_values[i] = st.sidebar.slider(f"V{i}", -5.0, 5.0, 0.0, step=0.1)

    analyze_clicked = st.button("🚨 Analyze Transaction", type="primary", use_container_width=True)
    demo_clicked = st.button("⚠️ Load Demo Fraud Case", use_container_width=True)

    row = None
    if demo_clicked:
        data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
        if not os.path.exists(data_path):
            st.error("data/creditcard.csv not found. Run `python generate_sample_data.py` first.")
        else:
            fraud_rows = pd.read_csv(data_path).query("Class == 1")
            if fraud_rows.empty:
                st.error("No fraud rows (Class == 1) found in data/creditcard.csv.")
            else:
                raw_row = fraud_rows.iloc[[0]][feature_cols].copy()
                amount = float(raw_row["Amount"].iloc[0])
                time_val = float(raw_row["Time"].iloc[0])
                st.warning(f"Loaded first fraud row from dataset — Amount: ${amount:,.2f}, "
                           f"Time: {time_val:.0f}s (sliders ignored)")
                row = raw_row.copy()
                row[["Amount", "Time"]] = scaler.transform(raw_row[["Amount", "Time"]])
    elif analyze_clicked:
        row = build_feature_row(feature_cols, amount, time_val, v_values, scaler)

    if row is not None:
        proba = float(model.predict_proba(row)[:, 1][0])

        st.session_state.analyzed_session += 1
        if proba >= 0.5:
            st.session_state.fraud_detected_session += 1

        is_fraud_pred = proba >= 0.5
        if is_fraud_pred:
            st.markdown('<div class="fg-banner fraud">🚨 FRAUD DETECTED — Transaction Blocked</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="fg-banner safe">✅ TRANSACTION APPROVED</div>',
                        unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.3])

        with col1:
            tone = "fraud" if is_fraud_pred else "safe"
            st.markdown(
                f'<div class="fg-bigpct {tone}">{proba * 100:.1f}%</div>'
                '<div class="fg-bigpct-lbl">Fraud probability</div>',
                unsafe_allow_html=True,
            )
            render_gauge(proba)

        top_features = explain_single(row, model=model, explainer=explainer, top_n=5)

        with col2:
            render_shap_bar(top_features)

        st.markdown("#### Plain-English Explanation")
        st.info(plain_english_explanation(top_features, amount, time_val))

        with st.expander("Raw feature contributions"):
            contrib_df = pd.DataFrame([
                {"Feature": k, "Value": v["value"], "SHAP Contribution": v["shap"],
                 "Direction": v["direction"]}
                for k, v in top_features.items()
            ])
            st.dataframe(contrib_df, use_container_width=True)
    else:
        st.info("Set transaction details in the sidebar, then click **Analyze Transaction**.")

    # ---- Live transaction feed (simulated from the dataset) ----
    st.markdown("---")
    head_l, head_r = st.columns([4, 1])
    head_l.markdown('#### <span class="fg-dot"></span>Live Transaction Feed', unsafe_allow_html=True)
    refresh_feed = head_r.button("🔄 New transactions", use_container_width=True)

    data_df = load_holdout()
    if data_df is None:
        st.caption("Feed unavailable: models/test_set.csv not found.")
    else:
        if refresh_feed or "feed_idx" not in st.session_state:
            fraud_idx = data_df.index[data_df["Class"] == 1].to_numpy()
            legit_idx = data_df.index[data_df["Class"] == 0].to_numpy()
            rng = np.random.default_rng()
            n_fraud_in_feed = int(rng.integers(1, 3))  # keep the feed lively: 1-2 frauds
            picks = np.concatenate([
                rng.choice(fraud_idx, size=min(n_fraud_in_feed, len(fraud_idx)), replace=False),
                rng.choice(legit_idx, size=5 - n_fraud_in_feed, replace=False),
            ])
            rng.shuffle(picks)
            st.session_state.feed_idx = picks.tolist()

        feed_raw = data_df.loc[st.session_state.feed_idx]
        feed_X = feed_raw[feature_cols].copy()
        feed_X[["Amount", "Time"]] = scaler.transform(feed_X[["Amount", "Time"]])
        feed_probs = model.predict_proba(feed_X)[:, 1]

        feed_shap = explainer.shap_values(feed_X)
        if isinstance(feed_shap, list):
            feed_shap = feed_shap[-1]
        feed_shap = np.asarray(feed_shap)

        for k, ((_, r), p) in enumerate(zip(feed_raw.iterrows(), feed_probs)):
            is_fraud = bool(p >= 0.5)
            icon = "🚨 FRAUD" if is_fraud else "✅ SAFE"
            header = (f"{icon}  ·  ${r['Amount']:,.2f}  ·  {seconds_to_hour(r['Time']):02d}:00"
                      f"  ·  risk {risk_dot(p)} {p * 100:.1f}%")
            with st.expander(header, expanded=False):
                reasons = feed_top_reasons(feed_shap[k], r, feature_cols, is_fraud)
                title = "🚨 Why this is fraud:" if is_fraud else "✅ Why this is safe:"
                st.markdown(f"**{title}**\n\n" + "\n".join(f"- {t}" for t in reasons))

# ----------------------------------------------------------------------------
# TAB 2: Model Performance
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Evaluation Metrics")

    metrics_path = os.path.join(MODEL_DIR, "metrics.pkl")
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
        m2.metric("F1 Score", f"{metrics['f1']:.4f}")
        m3.metric("Recall", f"{metrics['recall']:.4f}")
        m4.metric("Precision", f"{metrics['precision']:.4f}")

        m5, m6, m7 = st.columns(3)
        m5.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m6.metric("AUC-PR", f"{metrics['auc_pr']:.4f}")
        m7.metric("Optimal Threshold", f"{metrics['optimal_threshold']:.4f}")
    else:
        st.warning("No metrics found. Run `python src/evaluate.py` first.")

    st.markdown("---")
    st.subheader("Diagnostic Plots")

    col1, col2 = st.columns(2)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
    pr_path = os.path.join(MODEL_DIR, "pr_curve.png")

    with col1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    with col2:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve", use_container_width=True)

    if os.path.exists(pr_path):
        st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Global Feature Importance")

    shap_summary_path = os.path.join(MODEL_DIR, "shap_summary.png")
    shap_bar_path = os.path.join(MODEL_DIR, "shap_bar.png")

    if os.path.exists(shap_summary_path):
        st.image(shap_summary_path, caption="SHAP Summary (Beeswarm)", use_container_width=True)
    if os.path.exists(shap_bar_path):
        st.image(shap_bar_path, caption="SHAP Mean |Impact| by Feature", use_container_width=True)

    if not (os.path.exists(shap_summary_path) or os.path.exists(shap_bar_path)):
        st.warning("No SHAP plots found. Run `python src/explain.py` first.")

# ----------------------------------------------------------------------------
# TAB 3: Batch Analysis
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Upload a CSV of transactions")
    st.caption(f"Expected columns: {', '.join(feature_cols)}")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
        else:
            X_batch = batch_df[feature_cols].copy()
            X_batch[["Amount", "Time"]] = scaler.transform(X_batch[["Amount", "Time"]])

            probas = model.predict_proba(X_batch)[:, 1]
            predictions = (probas >= 0.5).astype(int)

            results_df = batch_df.copy()
            results_df["fraud_probability"] = probas
            results_df["prediction"] = np.where(predictions == 1, "FRAUD", "LEGIT")

            n_flagged = int(predictions.sum())
            pct_flagged = n_flagged / len(results_df) * 100

            flag_tone = "red" if pct_flagged > 1 else "green"
            st.markdown(
                f'<div class="fg-flagstat {flag_tone}"><div class="num">{pct_flagged:.2f}%</div>'
                '<div class="lbl">of transactions flagged</div></div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", len(results_df))
            c2.metric("Flagged as Fraud", n_flagged)
            c3.metric("% Flagged", f"{pct_flagged:.2f}%")

            def highlight_fraud(row):
                if row["prediction"] == "FRAUD":
                    return ["background-color: #fecaca; color: #7f1d1d; font-weight: 600"] * len(row)
                return [""] * len(row)

            st.markdown("#### Flagged Transactions")
            flagged_df = results_df[results_df["prediction"] == "FRAUD"].sort_values(
                "fraud_probability", ascending=False
            )

            if len(flagged_df) > 0:
                st.dataframe(
                    flagged_df.style.apply(highlight_fraud, axis=1),
                    use_container_width=True,
                )
            else:
                st.success("No fraudulent transactions detected in this batch.")

            st.markdown("#### Full Results (fraud rows highlighted)")
            st.dataframe(results_df.style.apply(highlight_fraud, axis=1), use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="fraudguard_batch_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a CSV with columns matching the training schema (Time, V1-V28, Amount) to run batch predictions.")

# ----------------------------------------------------------------------------
# TAB 4: Live Monitor
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("🔴 LIVE MONITOR")
    st.caption("Simulates 10 incoming transactions and scores each one with the trained model.")

    if st.button("▶️ Start Live Monitor", use_container_width=True):
        real_df = load_holdout()
        if real_df is None:
            st.error("models/test_set.csv not found. Run `python src/train.py` first.")
            st.stop()

        # Stream 10 held-out real transactions (unseen in training): 2-3 frauds + legit
        sim_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        pick_rng = np.random.default_rng(sim_seed)
        n_fraud_sim = int(pick_rng.integers(2, 4))
        fraud_pick = real_df[real_df["Class"] == 1].sample(n=n_fraud_sim, random_state=sim_seed % 10_000)
        legit_pick = real_df[real_df["Class"] == 0].sample(n=10 - n_fraud_sim, random_state=sim_seed % 10_000)
        sim = pd.concat([fraud_pick, legit_pick]).sample(frac=1.0, random_state=sim_seed % 10_000)
        sim = sim.reset_index(drop=True)
        # Slight random noise on V features so each transaction gets a unique risk score
        v_cols = [c for c in feature_cols if c.startswith("V")]
        sim[v_cols] = sim[v_cols] + np.random.normal(0, 0.3, size=(len(sim), len(v_cols)))

        X_sim = sim[feature_cols].copy()
        X_sim[["Amount", "Time"]] = scaler.transform(X_sim[["Amount", "Time"]])
        sim_probs = model.predict_proba(X_sim)[:, 1]

        sim_shap = explainer.shap_values(X_sim)
        if isinstance(sim_shap, list):
            sim_shap = sim_shap[-1]
        sim_shap = np.asarray(sim_shap)

        status_ph = st.empty()
        rows_ct = st.container()
        n_flagged_live = 0
        id_rng = np.random.default_rng(sim_seed)

        for i, (r, p) in enumerate(zip(sim.itertuples(index=False), sim_probs), start=1):
            flagged = bool(p >= 0.5)
            n_flagged_live += int(flagged)
            tone = risk_tier(p)
            status = "FRAUD 🚨" if flagged else "SAFE ✅"
            hh, mm = divmod(int(r.Time % 86400) // 60, 60)
            raw = sim.iloc[i - 1]
            reasons = feed_top_reasons(sim_shap[i - 1], raw, feature_cols, flagged)
            title = "🚨 Why this is fraud:" if flagged else "✅ Why this is safe:"
            status_ph.markdown(f"**Processing transaction {i} of 10…**")
            with rows_ct:
                st.markdown(
                    f'<div class="fg-mon-row {tone}" style="margin-bottom:2px">'
                    f'<span>TXN-{int(id_rng.integers(100000, 999999))}</span>'
                    f'<span>${r.Amount:,.2f}</span>'
                    f'<span>{hh:02d}:{mm:02d}</span>'
                    f'<span>{p * 100:.1f}%</span>'
                    f'<span class="st">{status}</span></div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"Details · risk {risk_dot(p)} {p * 100:.1f}%", expanded=False):
                    st.markdown(f"**{title}**\n\n" + "\n".join(f"- {t}" for t in reasons))
            time.sleep(0.8)

        status_ph.empty()
        tone = "red" if n_flagged_live else "green"
        st.markdown(
            f'<div class="fg-flagstat {tone}"><div class="num">{n_flagged_live} / 10</div>'
            f'<div class="lbl">flagged as fraud — {n_flagged_live} out of 10</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Click **Start Live Monitor** to stream 10 simulated transactions through the model.")
