# 🛡️ FraudGuard AI

Explainable, real-time credit card fraud detection — built for a hackathon.

FraudGuard AI trains an XGBoost classifier on highly imbalanced transaction
data, balances the training set with SMOTE, and pairs every prediction with a
SHAP-based explanation so a reviewer can see *why* a transaction was flagged,
not just a bare probability. A Streamlit dashboard ties it all together:
single-transaction scoring, model performance reporting, and CSV batch
analysis.

## Project overview

- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
  (ULB/Kaggle) — 284,807 transactions, 492 frauds (0.17%), with anonymized
  PCA features `V1`-`V28` plus `Time` and `Amount`.
- **Model**: XGBoost (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`,
  `scale_pos_weight=58`) trained on a SMOTE-balanced training set, validated
  with 5-fold stratified cross-validation (AUC-PR).
- **Explainability**: SHAP `TreeExplainer` for both global feature importance
  and per-transaction contribution breakdowns.
- **App**: a 3-tab Streamlit dashboard —
  1. 🔍 **Predict Transaction** — adjust sliders, get a fraud probability
     gauge, a FRAUD/LEGITIMATE banner, top-5 SHAP drivers, and a plain-English
     explanation.
  2. 📊 **Model Performance** — AUC-ROC/F1/Recall/Precision metric cards,
     confusion matrix, ROC curve, and SHAP summary plots.
  3. 📁 **Batch Analysis** — upload a CSV, score every row, review flagged
     transactions, and download the results.

## Project structure

```
fraudguard/
├── data/                       # creditcard.csv goes here (real or synthetic)
├── models/                     # trained model, scaler, and generated plots
├── src/
│   ├── preprocess.py           # load, scale, split, SMOTE
│   ├── train.py                # XGBoost training + cross-validation
│   ├── evaluate.py             # metrics, confusion matrix, ROC/PR curves
│   └── explain.py              # SHAP global + per-transaction explanations
├── notebooks/
│   └── fraud_detection_eda.ipynb
├── app.py                      # Streamlit dashboard
├── generate_sample_data.py     # synthetic dataset generator
├── download_instructions.md    # how to get the real Kaggle dataset
└── requirements.txt
```

## Install and run

```bash
# from the fraudguard/ directory
pip install -r requirements.txt

# if you don't have the real Kaggle dataset yet, generate a synthetic one
python generate_sample_data.py

# train the model, then generate evaluation + SHAP plots
python src/train.py
python src/evaluate.py
python src/explain.py

# launch the dashboard
streamlit run app.py
```

## Train from scratch

```bash
python src/train.py
```

This re-runs preprocessing (scaling + stratified split + SMOTE), 5-fold
cross-validation, fits the final model, and saves `models/xgb_fraud_model.pkl`
along with the fitted scaler, feature column order, and the held-out test
set (so `evaluate.py` / `explain.py` reuse the exact same split).

## Dataset

By default the pipeline looks for `data/creditcard.csv`. You have two
options:

1. **Real data** — download the Kaggle dataset. See
   [`download_instructions.md`](download_instructions.md) for step-by-step
   instructions (including the Kaggle API).
2. **Synthetic data** — run `python generate_sample_data.py` to create a
   10,000-row synthetic dataset with the same schema and a realistic ~1.7%
   fraud rate, so the whole project works immediately without Kaggle access.

Swap in the real dataset at any time and re-run the training pipeline to
retrain on real data — no code changes required.
