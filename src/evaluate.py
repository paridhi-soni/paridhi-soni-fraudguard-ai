"""
Evaluate the trained XGBoost fraud model on the held-out test set:
Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, classification report,
confusion matrix / ROC / Precision-Recall plots, and the F1-optimal
decision threshold.
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud_model.pkl")
TEST_SET_PATH = os.path.join(MODEL_DIR, "test_set.csv")


def load_test_set():
    df = pd.read_csv(TEST_SET_PATH)
    y = df["Class"]
    X = df.drop(columns=["Class"])
    return X, y


def find_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    return best_threshold, f1_scores[best_idx]


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_proba, out_path):
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recalls, precisions, label=f"PR curve (AUC-PR = {ap:.4f})", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate():
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_set()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    print("=" * 50)
    print("MODEL EVALUATION (threshold = 0.5)")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], zero_division=0))

    best_threshold, best_f1 = find_optimal_threshold(y_test, y_proba)
    print(f"Optimal F1 threshold: {best_threshold:.4f} (F1 = {best_f1:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba, os.path.join(MODEL_DIR, "roc_curve.png"))
    plot_pr_curve(y_test, y_proba, os.path.join(MODEL_DIR, "pr_curve.png"))
    print(f"Plots saved to: {MODEL_DIR}")

    metrics = dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        auc_roc=auc_roc, auc_pr=auc_pr, optimal_threshold=float(best_threshold),
    )
    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.pkl"))
    return metrics


if __name__ == "__main__":
    evaluate()
