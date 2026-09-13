"""
Reusable FedAvg simulation runner (Flower's flwr.simulation.run_simulation),
extracted from experiments/phase4_fedavg.py so both the single-run report
script and the multi-seed comparison script
(experiments/phase4_multiseed_comparison.py) share exactly one
implementation instead of two copies that could drift apart.
"""

from __future__ import annotations

import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from src.data.partition import HospitalPartition
from src.federated.client import make_client_fn
from src.federated.strategy_utils import MetricsRecorder
from src.models.nn_model import HeartDiseaseNet, get_model_parameters


def run_fedavg_simulation(
    partitions: list[HospitalPartition],
    n_features: int,
    model_init_seed: int,
    n_rounds: int = 20,
    local_epochs: int = 5,
    lr: float = 0.01,
) -> pd.DataFrame:
    """
    Runs one FedAvg simulation and returns the per-round, per-hospital
    evaluation results as a DataFrame (columns: round, hospital, n_test,
    accuracy, precision, recall, f1, auc).

    model_init_seed controls the only source of run-to-run variation here
    (the initial network weights) -- data partitioning, local train/test
    splits, client sampling (fraction_fit=fraction_evaluate=1.0, all
    clients every round), and the full-batch/no-dropout optimization path
    are otherwise deterministic given the same partitions.
    """
    n_clients = len(partitions)

    def fit_config_fn(server_round: int) -> dict:
        return {"epochs": local_epochs, "lr": lr}

    torch.manual_seed(model_init_seed)
    initial_model = HeartDiseaseNet(n_features)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    recorder = MetricsRecorder()

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=recorder.record_fit,
        evaluate_metrics_aggregation_fn=recorder.record_evaluate,
    )

    client_app = ClientApp(client_fn=make_client_fn(partitions, n_features, seed=model_init_seed))

    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=n_rounds))

    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}},
        verbose_logging=False,
    )

    return recorder.to_dataframe()


def weighted_accuracy(df: pd.DataFrame) -> float:
    """Sample-size-weighted accuracy across the hospital rows in `df`."""
    return float((df["accuracy"] * df["n_test"]).sum() / df["n_test"].sum())
