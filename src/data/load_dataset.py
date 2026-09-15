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

    # Cast every categorical/count code column to int (not just ca/thal,
    # which needed it to fix their missing-value float upcast). Without
    # this, pd.get_dummies produces inconsistent column names ("cp_4.0" vs
    # "thal_3") depending on which columns pandas happened to read as
    # float64 -- harmless for training (the column names are just labels),
    # but it makes the categorical_categories mapping in the preprocessing
    # artifact (see compute_preprocessing_artifact) ambiguous for
    # transforming a new raw patient record at inference time (Phase 7).
    for col in ["ca", "thal", "cp", "restecg", "slope"]:
        df[col] = df[col].astype(int)

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


PREPROCESSING_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "results" / "models" / "preprocessing.json"
)


def compute_preprocessing_artifact(df_clean: pd.DataFrame, x: pd.DataFrame) -> dict:
    """
    Captures everything needed to transform a NEW raw patient record into
    the exact numeric vector a trained model expects (Phase 7's live
    prediction endpoint): standardization constants, the categorical
    dummy-column mapping, and the final column order. Without this, a live
    endpoint has no way to reproduce build_feature_matrix()'s encoding for
    a single new patient outside the training pipeline.
    """
    means = df_clean[NUMERIC_FEATURES].mean()
    stds = df_clean[NUMERIC_FEATURES].std()

    categorical_categories = {
        feature: sorted(df_clean[feature].unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }

    return {
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_categories": categorical_categories,
        "means": {f: float(means[f]) for f in NUMERIC_FEATURES},
        "stds": {f: float(stds[f]) for f in NUMERIC_FEATURES},
        "feature_order": list(x.columns),
    }


def save_preprocessing_artifact(artifact: dict, path: Path = PREPROCESSING_ARTIFACT_PATH) -> Path:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return path


def load_preprocessing_artifact(path: Path = PREPROCESSING_ARTIFACT_PATH) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def transform_new_patient(raw: dict, artifact: dict) -> pd.DataFrame:
    """
    Transforms ONE new raw patient record (a flat dict with keys age, sex,
    cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
    thal -- the original UCI feature codes, e.g. cp one of {1,2,3,4}) into
    a single-row DataFrame matching the exact column order and encoding a
    trained model expects. Used only for live inference (Phase 7) -- never
    for anything that would need to be reproduced exactly for training.

    Raises KeyError if a required raw feature is missing, and ValueError if
    a categorical feature's value was never seen during training (the
    artifact's categorical_categories lists what's valid).
    """
    row = {}

    for f in artifact["numeric_features"]:
        row[f] = (float(raw[f]) - artifact["means"][f]) / artifact["stds"][f]

    for f in artifact["binary_features"]:
        row[f] = float(raw[f])

    for f in artifact["categorical_features"]:
        value = raw[f]
        categories = artifact["categorical_categories"][f]
        if value not in categories:
            raise ValueError(
                f"'{f}'={value!r} was never seen during training (valid values: {categories})"
            )
        for c in categories:
            row[f"{f}_{c}"] = 1.0 if value == c else 0.0

    return pd.DataFrame([row])[artifact["feature_order"]]


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
