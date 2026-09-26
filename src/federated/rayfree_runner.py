"""
Ray-free federated training loop for the hosted live-training feature.

Runs the same algorithm as src/federated/fedavg_runner.py and
fedprox_runner.py -- the same HospitalFlowerClient.fit()/evaluate() and
Flower's own aggregate() -- in a plain Python loop instead of Flower's Ray
simulation engine. Measured on 2026-09-26 (see README, "Live-training
feasibility"): Flower + Ray peaked at ~2.4 GB and was OOM-killed under
512 MB / 0.1 CPU (Render's free instance), while this loop peaked at
~320 MB, finished 20 rounds in ~61 s under those limits, and reproduced the
Flower runs' per-round, per-hospital accuracies 80/80 for both FedAvg and
FedProx.

Also deterministic: every round visits clients in partition order, so the
floating-point summation order in aggregate() never changes (Flower under
Ray can differ by a patient or two between identical runs).

Mirrors Flower's behaviour step for step, per round:
  1. every client is created fresh, loads the current global weights and
     trains `local_epochs` full-batch Adam steps (new optimizer each round);
     FedProx adds the proximal term when "proximal_mu" is in the config
  2. aggregate(): average weighted by each client's training-set size
  3. every client evaluates the new global model on its own test set
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from flwr.server.strategy.aggregate import aggregate

from src.data.partition import HospitalPartition
from src.federated.client import HospitalFlowerClient
from src.federated.personalize import personalize_per_hospital
from src.models.nn_model import HeartDiseaseNet, get_model_parameters

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "auc")


def _weighted(rows: list[dict], key: str) -> float:
    total = sum(r["n_test"] for r in rows)
    return sum(r["n_test"] * r[key] for r in rows) / total if total else float("nan")


def iter_federated_training(
    partitions: list[HospitalPartition],
    n_features: int,
    *,
    model_init_seed: int = 42,
    n_rounds: int = 20,
    local_epochs: int = 5,
    lr: float = 0.01,
    proximal_mu: float = 0.0,
    personalize: bool = False,
    fine_tune_epochs: int = 10,
) -> Iterator[dict]:
    """
    Yields one event dict per round:
        {"type": "round", "round": r, "hospitals": [...], "global_accuracy": g}
    then, if personalize=True, one {"type": "personalized", ...} event with
    each hospital's accuracy after local fine-tuning of the final global
    model (the same step as src/federated/personalize.py in Phase 5).

    proximal_mu=0 is FedAvg; proximal_mu>0 is FedProx.
    """
    torch.manual_seed(model_init_seed)
    params = get_model_parameters(HeartDiseaseNet(n_features))
    config = {"epochs": local_epochs, "lr": lr}
    if proximal_mu > 0:
        config["proximal_mu"] = proximal_mu

    for rnd in range(1, n_rounds + 1):
        fit_results = []
        for p in partitions:
            new_params, n_train, _ = HospitalFlowerClient(p, n_features, seed=model_init_seed).fit(params, config)
            fit_results.append((new_params, n_train))
        params = aggregate(fit_results)

        hospitals = []
        for p in partitions:
            _, n_test, metrics = HospitalFlowerClient(p, n_features, seed=model_init_seed).evaluate(params, {})
            row = {"hospital": metrics["hospital"], "n_test": int(n_test)}
            for key in METRIC_KEYS:
                value = float(metrics[key])
                row[key] = None if key == "auc" and value == -1.0 else value  # -1.0 = AUC undefined
            hospitals.append(row)
        yield {
            "type": "round",
            "round": rnd,
            "hospitals": hospitals,
            "global_accuracy": _weighted(hospitals, "accuracy"),
        }

    if personalize:
        df = personalize_per_hospital(params, partitions, n_features, fine_tune_epochs=fine_tune_epochs, lr=lr)
        hospitals = [
            {"hospital": r["hospital"], "n_test": int(r["n_test"]),
             **{k: (None if r[k] != r[k] else float(r[k])) for k in METRIC_KEYS}}
            for r in df.to_dict("records")
        ]
        yield {"type": "personalized", "hospitals": hospitals, "global_accuracy": _weighted(hospitals, "accuracy")}
