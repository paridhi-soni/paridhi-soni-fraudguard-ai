"""
Train an XGBoost fraud classifier with 5-fold stratified cross-validation,
then fit on the full (SMOTE-balanced) training set and persist the model.
"""

import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud_model.pkl")

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=58,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)


def build_model() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def cross_validate(X, y, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = build_model()
    scores = cross_val_score(model, X, y, cv=skf, scoring="average_precision", n_jobs=-1)
    return scores


def train():
    print("Loading & preprocessing data (scaling + stratified split + SMOTE)...")
    X_train, X_test, y_train, y_test, scaler = preprocess()

    print(f"Training set after SMOTE: {X_train.shape}, fraud rate: {np.mean(y_train):.4f}")
    print("Running 5-fold StratifiedKFold cross-validation (scoring=average_precision/AUC-PR)...")
    cv_scores = cross_validate(X_train, y_train, n_splits=5)
    print(f"CV AUC-PR scores: {np.round(cv_scores, 4)}")
    print(f"CV AUC-PR mean: {cv_scores.mean():.4f}  std: {cv_scores.std():.4f}")

    print("Fitting final model on full training set...")
    model = build_model()
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))

    # Persist the held-out test set so evaluate.py / explain.py can reuse the
    # exact same split without retraining.
    X_test.assign(Class=y_test.values).to_csv(
        os.path.join(MODEL_DIR, "test_set.csv"), index=False
    )

    print(f"Model saved to: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
