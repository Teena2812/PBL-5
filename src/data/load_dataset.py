"""
Loads and preprocesses the UCI Heart Disease dataset (Cleveland processed version).

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/45/heart+disease
Raw file used: processed.cleveland.data (303 rows, 14 columns, "?" for missing values)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Features that are genuinely continuous/ordinal-numeric and get standardized.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Features that are categorical codes and get one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Already-binary features, left as-is (0/1).
BINARY_FEATURES = ["sex", "fbs", "exang"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
PROCESSED_PATH = DATA_DIR / "processed" / "heart_disease_processed.csv"


def download_raw(force: bool = False) -> Path:
    """Downloads the raw UCI file to data/raw/ if not already present."""
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and not force:
        return RAW_PATH

    response = requests.get(RAW_URL, timeout=30)
    response.raise_for_status()
    RAW_PATH.write_text(response.text, encoding="utf-8")
    return RAW_PATH


def load_raw_dataframe() -> pd.DataFrame:
    """Reads the raw comma-separated file (no header, '?' = missing) into a DataFrame."""
    path = download_raw()
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing values and converts the multi-class target (0-4, severity)
    into a binary target (0 = no disease, 1 = disease present), matching the
    standard convention used by most published work on this dataset.
    """
    df = df.copy()

    # 'ca' and 'thal' have a handful of missing values in the raw file.
    # Drop rows with missing values rather than imputing, since it's <2% of rows
    # and this is a research prototype, not a production pipeline.
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with missing values "
              f"({n_before} -> {n_after} rows).")

    df["ca"] = df["ca"].astype(int)
    df["thal"] = df["thal"].astype(int)

    # Binarize target: 0 stays 0 (no disease), 1-4 (increasing severity) -> 1 (disease).
    df["target"] = (df["target"] > 0).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits into features (X) and label (y), one-hot encodes categorical
    features, and standardizes numeric features. Returns (X, y) where X is
    a fully numeric DataFrame ready for both sklearn and PyTorch.
    """
    y = df["target"].copy()

    x_numeric = df[NUMERIC_FEATURES].copy()
    x_binary = df[BINARY_FEATURES].copy()
    x_categorical = pd.get_dummies(
        df[CATEGORICAL_FEATURES].astype(str),
        prefix=CATEGORICAL_FEATURES,
    ).astype(int)

    # Standardize numeric features: (x - mean) / std, computed on the full
    # dataset here for reproducible preprocessing. Note: in later phases,
    # per-client normalization stats may be considered to more faithfully
    # simulate hospitals that never see each other's data.
    means = x_numeric.mean()
    stds = x_numeric.std()
    x_numeric_scaled = (x_numeric - means) / stds

    x = pd.concat([x_numeric_scaled, x_binary, x_categorical], axis=1)
    return x, y


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Full pipeline: download -> clean -> engineer -> feature matrix.

    Returns:
        X: numeric feature DataFrame (ready for ML models)
        y: binary target Series (0 = no disease, 1 = disease)
        df_clean: cleaned but not-yet-encoded DataFrame (kept for splitting
                  logic that needs the original 'age', 'cp', 'target' columns)
    """
    df_raw = load_raw_dataframe()
    df_clean = clean_and_engineer(df_raw)
    x, y = build_feature_matrix(df_clean)
    return x, y, df_clean


def save_processed(x: pd.DataFrame, y: pd.Series) -> Path:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = x.copy()
    out["target"] = y.values
    out.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH


if __name__ == "__main__":
    X, y, df_clean = load_and_preprocess()
    print(f"Loaded {len(df_clean)} patients, {X.shape[1]} features after encoding.")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    path = save_processed(X, y)
    print(f"Saved processed dataset to {path}")
