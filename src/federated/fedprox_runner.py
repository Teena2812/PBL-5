"""
FedProx simulation runner (Phase 5 personalization, layer 1: the proximal
term). Mirrors src/federated/fedavg_runner.py's structure, but:

  1. Uses Flower's built-in FedProx strategy, which sends "proximal_mu" in
     the fit config each round -- src/federated/client.py's fit() picks
     this up and adds (mu/2)||local - global||^2 to the local loss
     (src/models/nn_train.local_train), keeping each hospital's local
     update from drifting too far from the shared global model.
  2. Captures the final round's aggregated global parameters (via a thin
     FedProx subclass) and returns them as ndarrays, so a second
     personalization step (src/federated/personalize.py, Phase 5 layer 2:
     local fine-tuning) can start from the trained global model and adapt
     it further to each hospital -- matching the proposal's stated
     personalization strategy of "FedProx + local fine-tuning per
     hospital".
"""

from __future__ import annotations

import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.common import Context, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from flwr.simulation import run_simulation

from src.data.partition import HospitalPartition
from src.federated.client import make_client_fn
from src.federated.strategy_utils import MetricsRecorder
from src.models.nn_model import HeartDiseaseNet, get_model_parameters


class _RecordingFedProx(FedProx):
    """FedProx that also stashes the latest aggregated global parameters,
    so the caller can read them back after run_simulation() returns."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, metrics


def run_fedprox_simulation(
    partitions: list[HospitalPartition],
    n_features: int,
    model_init_seed: int,
    proximal_mu: float,
    n_rounds: int = 20,
    local_epochs: int = 5,
    lr: float = 0.01,
) -> tuple[pd.DataFrame, NDArrays]:
    """
    Runs one FedProx simulation. Returns (per_round_df, final_global_ndarrays)
    -- the per-round/per-hospital evaluation DataFrame (same shape as
    run_fedavg_simulation's) plus the final global model's weights as a
    list of numpy arrays, ready for src/federated/personalize.py.
    """
    n_clients = len(partitions)

    def fit_config_fn(server_round: int) -> dict:
        return {"epochs": local_epochs, "lr": lr}

    torch.manual_seed(model_init_seed)
    initial_model = HeartDiseaseNet(n_features)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(initial_model))

    recorder = MetricsRecorder()

    strategy = _RecordingFedProx(
        proximal_mu=proximal_mu,
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

    if strategy.final_parameters is None:
        raise RuntimeError("FedProx simulation finished without ever aggregating parameters.")

    final_ndarrays = parameters_to_ndarrays(strategy.final_parameters)
    return recorder.to_dataframe(), final_ndarrays
