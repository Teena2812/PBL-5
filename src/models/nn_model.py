"""
Small PyTorch feed-forward NN used as the shared model architecture for
FedAvg (Phase 4) and Personalized FL / FedProx (Phase 5). Kept small (2
hidden layers) since each simulated hospital only has 20-60 local training
examples -- a larger network would just overfit.
"""

from __future__ import annotations

import torch
from torch import nn


class HeartDiseaseNet(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: tuple[int, int] = (16, 8)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logits, shape (batch,)


def get_model_parameters(model: nn.Module) -> list:
    """Returns model weights as a list of numpy arrays (Flower's parameter format)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list) -> None:
    """Loads a list of numpy arrays (Flower's parameter format) into a model in place."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
