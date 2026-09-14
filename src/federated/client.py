"""
Flower NumPyClient wrapping HeartDiseaseNet for one simulated hospital.

Each client trains locally with src.models.nn_train.local_train and
evaluates with evaluate_nn, using the SAME local train/test split produced
by src.data.partition for Local ML / Centralized ML (Phase 3), so results
stay comparable across phases.

Plain FedAvg (Phase 4) vs FedProx (Phase 5) is controlled entirely by
whether the server sends a "proximal_mu" key in the fit config: Flower's
built-in FedProx strategy adds it automatically (see
src/federated/fedprox_runner.py); FedAvg's strategy never does, so
config.get("proximal_mu", 0.0) defaults to plain FedAvg here.
"""

from __future__ import annotations

import torch
from flwr.client import Client, NumPyClient
from flwr.common import Context

from src.data.partition import HospitalPartition
from src.models.nn_model import HeartDiseaseNet, get_model_parameters, set_model_parameters
from src.models.nn_train import evaluate_nn, local_train


class HospitalFlowerClient(NumPyClient):
    def __init__(self, partition: HospitalPartition, n_features: int, seed: int = 42):
        self.partition = partition
        self.n_features = n_features
        self.seed = seed
        self.model = HeartDiseaseNet(n_features)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        # Capture the global weights AS RECEIVED (before local training moves
        # them) as the FedProx proximal anchor point -- ||local - global||^2
        # is measured against this, not against whatever the local model
        # drifts to during training.
        global_params = [torch.tensor(p) for p in parameters]
        set_model_parameters(self.model, parameters)

        epochs = int(config.get("epochs", 5))
        lr = float(config.get("lr", 0.01))
        mu = float(config.get("proximal_mu", 0.0))

        local_train(
            self.model, self.partition.x_train, self.partition.y_train,
            epochs=epochs, lr=lr, mu=mu, global_params=global_params,
        )
        return get_model_parameters(self.model), len(self.partition.x_train), {
            "hospital": self.partition.name,
        }

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        metrics = evaluate_nn(self.model, self.partition.x_test, self.partition.y_test)
        # loss = 1 - accuracy as a simple scalar Flower can aggregate; the
        # real metrics of interest (accuracy/precision/recall/f1/auc) travel
        # in the metrics dict and are what we report in the final results.
        loss = 1.0 - metrics["accuracy"]
        auc = metrics["auc"]
        return float(loss), metrics["n_test"], {
            "hospital": self.partition.name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": float(auc) if auc == auc else -1.0,  # NaN -> -1.0 sentinel (Flower Scalar can't carry NaN cleanly)
        }


def make_client_fn(partitions: list[HospitalPartition], n_features: int, seed: int = 42):
    """
    Builds the client_fn Flower's simulation engine calls once per virtual
    client, mapping context.node_config['partition-id'] (0..n_clients-1,
    assigned by Flower) to our hospital partitions list in order.
    """
    def client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        partition = partitions[partition_id]
        return HospitalFlowerClient(partition, n_features, seed=seed).to_client()

    return client_fn
