"""
Flower NumPyClient wrapping HeartDiseaseNet for one simulated hospital.

Each client trains locally with src.models.nn_train.local_train (plain
FedAvg here: mu=0, no proximal term) and evaluates with evaluate_nn, using
the SAME local train/test split produced by src.data.partition for Local ML
/ Centralized ML (Phase 3), so results stay comparable across phases.
"""

from __future__ import annotations

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
        set_model_parameters(self.model, parameters)
        epochs = int(config.get("epochs", 5))
        lr = float(config.get("lr", 0.01))
        local_train(self.model, self.partition.x_train, self.partition.y_train, epochs=epochs, lr=lr)
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
