"""
Phase 5 personalization, layer 2: local fine-tuning.

Starting from the trained FedProx global model, each hospital fine-tunes
its OWN copy for a few more epochs on only its own local training data
(no proximal term -- mu=0, nothing anchoring it to the global model
anymore, since the point here is to let it adapt) before being evaluated
on its own local test set. This is the standard "personalize by
fine-tuning the global model locally" approach, and matches the proposal's
stated strategy: "FedProx (proximal term) + local fine-tuning per
hospital, to adapt the global model to each hospital's population."
"""

from __future__ import annotations

import pandas as pd
from flwr.common import NDArrays
from torch import nn

from src.data.partition import HospitalPartition
from src.models.nn_model import HeartDiseaseNet, set_model_parameters
from src.models.nn_train import evaluate_nn, local_train


def personalize_per_hospital_with_models(
    global_ndarrays: NDArrays,
    partitions: list[HospitalPartition],
    n_features: int,
    fine_tune_epochs: int = 10,
    lr: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, nn.Module]]:
    """
    Same as personalize_per_hospital, but also returns the actual trained
    model objects (keyed by hospital name) -- needed to save checkpoints
    for Phase 7's live inference endpoint, not just their metrics.

    No seed argument -- fine-tuning starts from the already-trained global
    weights (not a fresh random init), so there is nothing to seed here;
    the only randomness (initial weights) was already fixed upstream, when
    the FedProx run that produced global_ndarrays was seeded.
    """
    rows = []
    models = {}
    for p in partitions:
        model = HeartDiseaseNet(n_features)
        set_model_parameters(model, global_ndarrays)
        local_train(model, p.x_train, p.y_train, epochs=fine_tune_epochs, lr=lr)  # mu=0: plain fine-tuning
        metrics = evaluate_nn(model, p.x_test, p.y_test)
        rows.append({"hospital": p.name, "n_train": len(p.x_train), **metrics})
        models[p.name] = model
    return pd.DataFrame(rows), models


def personalize_per_hospital(
    global_ndarrays: NDArrays,
    partitions: list[HospitalPartition],
    n_features: int,
    fine_tune_epochs: int = 10,
    lr: float = 0.01,
) -> pd.DataFrame:
    """
    For each hospital: load the trained global model, fine-tune a copy on
    that hospital's own local training data (mu=0, plain local training),
    evaluate on that hospital's own local test data. See
    personalize_per_hospital_with_models if you also need the trained
    model objects (e.g. to save checkpoints), not just their metrics.
    """
    metrics_df, _models = personalize_per_hospital_with_models(
        global_ndarrays, partitions, n_features, fine_tune_epochs=fine_tune_epochs, lr=lr
    )
    return metrics_df
