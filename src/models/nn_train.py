"""
Local training and evaluation helpers for HeartDiseaseNet, shared by the
FedAvg (Phase 4) and Personalized FL / FedProx (Phase 5) experiments.

local_train() accepts an optional FedProx proximal term (mu > 0), unused in
Phase 4 (plain FedAvg, mu=0) but kept here so Phase 5 can reuse this exact
training loop instead of duplicating it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


def _to_tensors(x: pd.DataFrame, y: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.tensor(x.to_numpy(), dtype=torch.float32)
    y_t = torch.tensor(y.to_numpy(), dtype=torch.float32)
    return x_t, y_t


def local_train(
    model: nn.Module,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    epochs: int = 5,
    lr: float = 0.01,
    mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> nn.Module:
    """
    Trains `model` in place on local data for `epochs` full-batch steps
    (each hospital has too few samples to justify mini-batching).

    If mu > 0 and global_params is given (FedProx, Phase 5), adds the
    proximal term (mu/2) * ||local_params - global_params||^2 to the loss
    to keep local updates from drifting too far from the global model.
    """
    x_t, y_t = _to_tensors(x_train, y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)

        if mu > 0 and global_params is not None:
            proximal_term = sum(
                (local_p - global_p).pow(2).sum()
                for local_p, global_p in zip(model.parameters(), global_params)
            )
            loss = loss + (mu / 2) * proximal_term

        loss.backward()
        optimizer.step()

    return model


def evaluate_nn(model: nn.Module, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Same metric set/semantics as src/models/baseline.py's evaluate_model,
    so Local ML / Centralized ML / FedAvg / Personalized FL results are
    directly comparable in the final report.
    """
    n_test = len(y_test)
    if n_test == 0:
        return {"n_test": 0, "accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan, "auc": np.nan}

    x_t, y_t = _to_tensors(x_test, y_test)
    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        y_proba = torch.sigmoid(logits).numpy()
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = y_t.numpy().astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = roc_auc_score(y_true, y_proba)

    return {
        "n_test": n_test,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
    }
