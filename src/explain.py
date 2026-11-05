"""
SHAP-based explainability for the FraudGuard AI model:
- global summary (beeswarm) and bar (mean |SHAP|) plots
- explain_single(): per-transaction top-5 feature contributions
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud_model.pkl")
TEST_SET_PATH = os.path.join(MODEL_DIR, "test_set.csv")


def load_model():
    return joblib.load(MODEL_PATH)


def load_sample(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(TEST_SET_PATH)
    X = df.drop(columns=["Class"])
    if len(X) > n:
        X = X.sample(n=n, random_state=random_state)
    return X


def get_explainer(model):
    return shap.TreeExplainer(model)


def plot_summary(explainer, X, out_path):
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar(explainer, X, out_path):
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_single(transaction_df: pd.DataFrame, model=None, explainer=None, top_n: int = 5) -> dict:
    """
    Explain a single transaction (1-row DataFrame with the model's feature
    columns). Returns a dict: {feature_name: {"value": x, "shap": s, "direction": "fraud"/"legit"}}
    for the top_n features by absolute SHAP contribution.
    """
    if model is None:
        model = load_model()
    if explainer is None:
        explainer = get_explainer(model)

    shap_values = explainer.shap_values(transaction_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    row_shap = shap_values[0]
    row_values = transaction_df.iloc[0]

    contributions = []
    for feature, shap_val, feat_val in zip(transaction_df.columns, row_shap, row_values):
        contributions.append({
            "feature": feature,
            "value": float(feat_val),
            "shap": float(shap_val),
            "direction": "fraud" if shap_val > 0 else "legit",
        })

    contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
    top = contributions[:top_n]

    return {c["feature"]: {"value": c["value"], "shap": c["shap"], "direction": c["direction"]}
            for c in top}


def run():
    model = load_model()
    explainer = get_explainer(model)
    X_sample = load_sample()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Computing SHAP values and generating summary plot...")
    plot_summary(explainer, X_sample, os.path.join(MODEL_DIR, "shap_summary.png"))
    print("Generating SHAP bar (mean importance) plot...")
    plot_bar(explainer, X_sample, os.path.join(MODEL_DIR, "shap_bar.png"))
    print(f"SHAP plots saved to: {MODEL_DIR}")


if __name__ == "__main__":
    run()
