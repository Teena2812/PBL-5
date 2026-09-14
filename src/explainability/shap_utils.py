"""
SHAP explainability helpers (Phase 6): TreeExplainer for the Random Forest
baseline (exact, fast) and KernelExplainer for the PyTorch personalized NN
(model-agnostic, since SHAP has no exact/fast method for an arbitrary small
feed-forward net the way it does for trees).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn


def compute_tree_shap(model: RandomForestClassifier, x_explain: pd.DataFrame):
    """
    Exact SHAP values for a Random Forest via shap.TreeExplainer.
    Returns (explainer, shap_values_for_positive_class) -- shap_values has
    shape (n_samples, n_features), one row per patient in x_explain.
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(x_explain)
    # Newer shap versions return an array of shape (n_samples, n_features, n_classes)
    # for binary classifiers instead of a list of two (n_samples, n_features) arrays;
    # both forms need slicing down to just the positive class (index 1).
    if isinstance(raw, list):
        shap_values = raw[1]
    elif raw.ndim == 3:
        shap_values = raw[:, :, 1]
    else:
        shap_values = raw
    return explainer, shap_values


def compute_kernel_shap_nn(
    model: nn.Module,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    n_background: int = 20,
    nsamples: int = 100,
    seed: int = 42,
):
    """
    Model-agnostic SHAP values for the PyTorch NN via shap.KernelExplainer.
    Much slower than TreeExplainer (it perturbs features and re-queries the
    model many times per sample), so n_background/nsamples are kept small --
    fine for a handful of example patients, not meant for the full test set.
    """
    model.eval()

    def predict_proba(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            return torch.sigmoid(logits).numpy()

    background = shap.sample(x_background, min(n_background, len(x_background)), random_state=seed)
    explainer = shap.KernelExplainer(predict_proba, background)
    shap_values = explainer.shap_values(x_explain, nsamples=nsamples, silent=True)
    return explainer, shap_values


def summarize_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Global feature importance: mean |SHAP value| per feature, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_single_patient(
    shap_values: np.ndarray, x_explain: pd.DataFrame, patient_index: int, base_value: float, top_n: int = 6
) -> pd.DataFrame:
    """
    Ranked feature contributions for ONE patient (by |SHAP value|), the
    "Cholesterol +23%, Age +17%"-style panel described in the proposal.
    """
    row_shap = shap_values[patient_index]
    row_values = x_explain.iloc[patient_index]
    df = pd.DataFrame({
        "feature": x_explain.columns,
        "feature_value": row_values.values,
        "shap_value": row_shap,
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns=["abs_shap"])
    df.attrs["base_value"] = base_value
    df.attrs["predicted_value"] = float(base_value + row_shap.sum())
    return df
