"""
Generate a synthetic credit-card-fraud dataset that mimics the schema and
statistical feel of the Kaggle "Credit Card Fraud Detection" dataset, so the
rest of the FraudGuard AI pipeline can run end-to-end without needing the
real data.

Schema: Time, V1..V28, Amount, Class
Rows: 10,000
Fraud rate: ~1.7%
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 10_000
FRAUD_RATE = 0.017
N_FEATURES = 28
SECONDS_IN_TWO_DAYS = 172_800  # dataset spans ~2 days


def generate_dataset(n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE,
                      seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    # ---- Legitimate transactions ----
    # V1..V28 behave like PCA components: roughly standard normal, mild correlation
    legit_v = rng.normal(loc=0.0, scale=1.0, size=(n_legit, N_FEATURES))

    # Amounts: right-skewed, mostly small purchases
    legit_amount = rng.gamma(shape=2.0, scale=35.0, size=n_legit)
    legit_amount = np.clip(legit_amount, 0.01, 5000)
    # ~6% of legitimate purchases are big-ticket (overlap with fraud amounts)
    big_mask = rng.random(n_legit) < 0.06
    legit_amount = np.where(big_mask, rng.uniform(2000, 6000, size=n_legit), legit_amount)

    # Time: transactions spread fairly uniformly across ~2 days, with a dip at night
    legit_time = rng.uniform(0, SECONDS_IN_TWO_DAYS, size=n_legit)

    legit_df = pd.DataFrame(legit_v, columns=[f"V{i}" for i in range(1, N_FEATURES + 1)])
    legit_df["Time"] = legit_time
    legit_df["Amount"] = legit_amount
    legit_df["Class"] = 0

    # ---- Fraudulent transactions ----
    # Fraud tends to shift several PCA components (simulate anomalous pattern)
    # Fraud signal is deliberately weak so fraud and legit overlap:
    # V1 / V4 mean shifts are 50% of the earlier strength (2.1-5.0 -> 1.05-2.5).
    fraud_v = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, N_FEATURES))
    fraud_v += rng.normal(loc=0.0, scale=1.5, size=fraud_v.shape)
    which = rng.integers(0, 3, size=n_fraud)  # 0: V1 only, 1: V4 only, 2: both
    fraud_v[:, 0] += np.where(which != 1, -rng.uniform(1.05, 2.5, size=n_fraud), 0.0)
    fraud_v[:, 3] += np.where(which != 0, rng.uniform(1.05, 2.5, size=n_fraud), 0.0)

    # Amounts: mostly the suspicious extremes (< $5 or > $2000), but ~30% of
    # fraud sits at ordinary / medium amounts so Amount alone doesn't give it away
    r_amt = rng.random(n_fraud)
    fraud_amount = np.where(
        r_amt < 0.35, rng.uniform(0.5, 4.99, size=n_fraud),         # card-testing micro-charges
        np.where(r_amt < 0.70, rng.uniform(2000.01, 9500, size=n_fraud),  # large purchases
                 rng.uniform(50, 1500, size=n_fraud)),                    # medium amounts
    )

    # Fraud happens at ALL hours; only a mild (15%) night-time lean as minor noise
    fraud_time = rng.uniform(0, SECONDS_IN_TWO_DAYS, size=n_fraud)
    night_mask = rng.random(n_fraud) < 0.15
    night_seconds = rng.uniform(0, 5 * 3600, size=n_fraud) + rng.choice([0, 1], size=n_fraud) * 86400
    fraud_time = np.where(night_mask, night_seconds, fraud_time)

    fraud_df = pd.DataFrame(fraud_v, columns=[f"V{i}" for i in range(1, N_FEATURES + 1)])
    fraud_df["Time"] = fraud_time
    fraud_df["Amount"] = fraud_amount
    fraud_df["Class"] = 1

    # ---- Combine & shuffle ----
    df = pd.concat([legit_df, fraud_df], ignore_index=True)

    # Gaussian noise (std=2.0) on ALL features of ALL rows to blur class boundaries
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, N_FEATURES + 1)] + ["Amount"]
    df[feature_cols] += rng.normal(loc=0.0, scale=2.0, size=(len(df), len(feature_cols)))
    df["Amount"] = df["Amount"].clip(lower=0.01)
    df["Time"] = df["Time"].clip(lower=0)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Reorder columns to match the real dataset: Time, V1..V28, Amount, Class
    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, N_FEATURES + 1)] + ["Amount", "Class"]
    df = df[ordered_cols]

    # Round for realism
    df["Time"] = df["Time"].round(0)
    df["Amount"] = df["Amount"].round(2)

    return df


def main():
    import os

    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "creditcard.csv")
    df.to_csv(out_path, index=False)

    n_fraud = int(df["Class"].sum())
    print(f"Synthetic dataset generated: {len(df)} rows")
    print(f"Fraud cases: {n_fraud} ({n_fraud / len(df) * 100:.2f}%)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
