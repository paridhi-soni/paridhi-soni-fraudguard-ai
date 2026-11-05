# Downloading the Real Dataset

FraudGuard AI is designed around the **Credit Card Fraud Detection** dataset
published on Kaggle by the Machine Learning Group at ULB (Universite Libre de
Bruxelles). It contains 284,807 European credit card transactions made in
September 2013, of which 492 are fraudulent (0.172%).

Columns: `Time`, `V1`...`V28` (PCA-transformed, anonymized features), `Amount`,
and `Class` (1 = fraud, 0 = legitimate).

## Steps

1. Create a free Kaggle account at https://www.kaggle.com if you don't have one.
2. Go to the dataset page:
   https://www.kaggle.com/mlg-ulb/creditcardfraud
3. Click **Download** (requires login). This downloads `creditcard.csv.zip`.
4. Unzip it and place `creditcard.csv` into the `data/` folder of this
   project, so the path is `fraudguard/data/creditcard.csv`.

### Alternative: Kaggle API

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Don't have Kaggle access right now?

No problem — run `python generate_sample_data.py` from the project root. It
creates a realistic **synthetic** version of this dataset (10,000 rows, the
same column schema, ~1.7% fraud rate) and saves it to `data/creditcard.csv`,
so every part of the pipeline (training, evaluation, SHAP, and the Streamlit
app) works immediately without needing the real data. Swap in the real
`creditcard.csv` at any time and re-run `python src/train.py` to retrain on
real data.
