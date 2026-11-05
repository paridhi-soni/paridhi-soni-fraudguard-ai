"""
Preprocessing pipeline for the FraudGuard AI dataset:
load -> clean -> scale Amount/Time -> stratified train/test split -> SMOTE
(applied to the training set only, to avoid leaking synthetic samples into
the test set).
"""

import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "creditcard.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def scale_features(df: pd.DataFrame):
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def preprocess(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42,
               use_smote: bool = True):
    """Full pipeline: returns X_train, X_test, y_train, y_test, scaler."""
    df = load_data(path)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess()
    print(f"X_train: {X_train.shape}, fraud rate: {np.mean(y_train):.4f}")
    print(f"X_test:  {X_test.shape}, fraud rate: {np.mean(y_test):.4f}")
