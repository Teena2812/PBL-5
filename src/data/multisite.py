"""
Real multi-institutional heart disease data: the four UCI Heart Disease
sites, each used directly as one federated client (no synthetic
partitioning). Replaces the earlier Dirichlet split of Cleveland alone.

Source: UCI Machine Learning Repository, "Heart Disease" (Janosi, Steinbrunn,
Pfisterer & Detrano, 1988), CC BY 4.0, DOI 10.24432/C52P4X. The data's
principal investigators ask that publications name them:
  Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
  University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
  University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
  V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
  Robert Detrano, M.D., Ph.D.

Cleaning ("Option A"), every step verified against the raw files:
  - binary target: 0 = no disease, 1-4 = disease
  - physiologically impossible zeros are missing values recorded as 0
    (trestbps == 0: 1 VA row; cholesterol == 0: all 123 Swiss rows and 49
    VA rows -- cholesterol is not used at all, see below)
  - exact duplicate rows within a site dropped (1 in Hungary, 1 in VA)
  - 8 features shared by all four sites; ca, thal (83-99% missing at three
    sites), slope (51-65% missing at two), chol (100% missing in
    Switzerland) and fbs (61% missing in Switzerland) are excluded
  - complete cases only on those 8 features -- no imputation
Result: Cleveland 303, Hungary 291, Switzerland 116, VA Long Beach 139
(849 patients).

The Statlog (Heart) dataset is NOT a fifth site: all 270 of its rows are
exact copies of Cleveland rows, so including it would double-count
patients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.partition import HospitalPartition

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Client id -> UCI file. Order fixes the client/partition ids (0..3).
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}
SITE_LABELS = {
    "cleveland": "Cleveland Clinic",
    "hungary": "Hungarian Institute of Cardiology",
    "switzerland": "University Hospitals Zurich & Basel",
    "va_long_beach": "VA Long Beach",
}
EXPECTED_SITE_COUNTS = {"cleveland": 303, "hungary": 291, "switzerland": 116, "va_long_beach": 139}

NUMERIC_FEATURES = ["age", "trestbps", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg"]
USED_FEATURES = ["age", "sex", "cp", "trestbps", "restecg", "thalach", "exang", "oldpeak"]


def download_site_files(force: bool = False) -> dict[str, Path]:
    """Downloads the four processed UCI site files to data/raw/ if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for site, filename in SITES.items():
        path = RAW_DIR / filename
        if force or not path.exists():
            response = requests.get(UCI_BASE + filename, timeout=30)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
        paths[site] = path
    return paths


def load_clean_sites() -> pd.DataFrame:
    """All four sites, cleaned per Option A, with a 'site' column. Asserts
    the verified per-site counts so any silent change fails loudly."""
    frames = []
    for site, path in download_site_files().items():
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
        df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
        df = df.drop_duplicates()
        df = df.dropna(subset=USED_FEATURES).copy()
        for col in ["sex", "cp", "restecg", "exang"]:
            df[col] = df[col].astype(int)
        df["target"] = (df["target"] > 0).astype(int)
        df["site"] = site
        frames.append(df[USED_FEATURES + ["target", "site"]])
    df_clean = pd.concat(frames, ignore_index=True)

    counts = df_clean["site"].value_counts().to_dict()
    if counts != EXPECTED_SITE_COUNTS:
        raise ValueError(f"Site counts {counts} differ from the verified {EXPECTED_SITE_COUNTS}")
    return df_clean


def build_feature_matrix(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Same encoding approach as src/data/load_dataset.py: standardized
    numerics (pooled statistics, as in the earlier phases), binaries as-is,
    one-hot categoricals."""
    y = df_clean["target"].copy()
    x_numeric = df_clean[NUMERIC_FEATURES]
    x_numeric = (x_numeric - x_numeric.mean()) / x_numeric.std()
    x_categorical = pd.get_dummies(
        df_clean[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES
    ).astype(int)
    x = pd.concat([x_numeric, df_clean[BINARY_FEATURES], x_categorical], axis=1).astype(float)
    return x, y


def load_multisite() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Drop-in counterpart of load_dataset.load_and_preprocess():
    returns (x, y, df_clean) for all 849 patients."""
    df_clean = load_clean_sites()
    x, y = build_feature_matrix(df_clean)
    return x, y, df_clean


def create_site_partitions(
    x: pd.DataFrame, y: pd.Series, df_clean: pd.DataFrame
) -> list[HospitalPartition]:
    """One partition per real site -- the counterpart of
    partition.create_hospital_partitions() without any synthetic split."""
    partitions = []
    for client_id, site in enumerate(SITES):
        indices = np.flatnonzero(df_clean["site"].to_numpy() == site)
        subset = df_clean.iloc[indices]
        partitions.append(HospitalPartition(
            client_id=client_id,
            name=site,
            indices=indices,
            x=x.iloc[indices].reset_index(drop=True),
            y=y.iloc[indices].reset_index(drop=True),
            demographics={
                "n_patients": int(len(indices)),
                "disease_rate": round(float(subset["target"].mean()), 3),
                "mean_age": round(float(subset["age"].mean()), 1),
                "pct_female": round(float((subset["sex"] == 0).mean()), 3),
            },
        ))
    return partitions


def summarize_sites(partitions: list[HospitalPartition], df_clean: pd.DataFrame) -> pd.DataFrame:
    """Per-site summary table (rows = hospitals) for the write-up."""
    rows = []
    for p in partitions:
        subset = df_clean.iloc[p.indices]
        rows.append({
            "hospital": p.name,
            "institution": SITE_LABELS[p.name],
            "n_patients": len(p),
            "n_disease": int(subset["target"].sum()),
            "n_no_disease": int((subset["target"] == 0).sum()),
            "disease_rate": round(float(subset["target"].mean()), 3),
            "mean_age": round(float(subset["age"].mean()), 1),
            "pct_female": round(float((subset["sex"] == 0).mean()), 3),
            "dominant_cp_type": int(subset["cp"].mode().iloc[0]),
        })
    return pd.DataFrame(rows)


def compute_preprocessing_artifact(df_clean: pd.DataFrame, x: pd.DataFrame) -> dict:
    """Same schema as load_dataset.compute_preprocessing_artifact(), so
    load_dataset.transform_new_patient() encodes a new raw patient exactly
    like build_feature_matrix() above."""
    means = df_clean[NUMERIC_FEATURES].mean()
    stds = df_clean[NUMERIC_FEATURES].std()
    return {
        "dataset": "uci_heart_disease_4_sites_option_a",
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_categories": {f: sorted(df_clean[f].unique().tolist()) for f in CATEGORICAL_FEATURES},
        "means": {f: float(means[f]) for f in NUMERIC_FEATURES},
        "stds": {f: float(stds[f]) for f in NUMERIC_FEATURES},
        "feature_order": list(x.columns),
    }
