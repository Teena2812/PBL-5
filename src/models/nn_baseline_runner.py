"""
Reusable Local NN / Centralized NN baseline runners, extracted from
experiments/phase4_nn_baselines.py so both the single-seed report script
and experiments/phase4_multiseed_comparison.py share one implementation.
"""

from __future__ import annotations

import pandas as pd
import torch

from src.data.partition import HospitalPartition
from src.models.nn_model import HeartDiseaseNet
from src.models.nn_train import evaluate_nn, local_train


def run_local_nn(
    partitions: list[HospitalPartition], n_features: int, seed: int, epochs: int, lr: float
) -> pd.DataFrame:
    """Each hospital trains its own HeartDiseaseNet in isolation."""
    rows = []
    for p in partitions:
        torch.manual_seed(seed)
        model = HeartDiseaseNet(n_features)
        local_train(model, p.x_train, p.y_train, epochs=epochs, lr=lr)
        metrics = evaluate_nn(model, p.x_test, p.y_test)
        rows.append({"hospital": p.name, "n_train": len(p.x_train), **metrics})
    return pd.DataFrame(rows)


def run_centralized_nn(
    partitions: list[HospitalPartition], n_features: int, seed: int, epochs: int, lr: float
) -> tuple[dict, pd.DataFrame]:
    """Pools every hospital's local training data into one HeartDiseaseNet."""
    x_train_pooled = pd.concat([p.x_train for p in partitions], ignore_index=True)
    y_train_pooled = pd.concat([p.y_train for p in partitions], ignore_index=True)
    x_test_pooled = pd.concat([p.x_test for p in partitions], ignore_index=True)
    y_test_pooled = pd.concat([p.y_test for p in partitions], ignore_index=True)

    torch.manual_seed(seed)
    model = HeartDiseaseNet(n_features)
    local_train(model, x_train_pooled, y_train_pooled, epochs=epochs, lr=lr)

    overall_metrics = evaluate_nn(model, x_test_pooled, y_test_pooled)
    overall_metrics["n_train"] = len(x_train_pooled)

    per_hospital_rows = []
    for p in partitions:
        metrics = evaluate_nn(model, p.x_test, p.y_test)
        per_hospital_rows.append({"hospital": p.name, **metrics})
    per_hospital_df = pd.DataFrame(per_hospital_rows)

    return overall_metrics, per_hospital_df
