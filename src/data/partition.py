"""
Simulates non-IID hospital data splits from the single UCI Heart Disease dataset.

Strategy: Dirichlet label-skew partitioning (Hsu et al., 2019 -- the standard
non-IID simulation method in the federated learning literature). Each
simulated hospital gets a different class-balance profile controlled by a
concentration parameter `alpha`:
    - low alpha  (e.g. 0.1-0.3)  -> highly skewed hospitals (some almost
      entirely healthy or almost entirely diseased patients)
    - high alpha (e.g. 5-10)     -> close to IID / uniform across hospitals

This is preferred over splitting by a single raw feature (e.g. age band)
because it is (a) the standard, citable approach so reviewers/evaluators
recognize it, (b) tunable via one parameter to dial heterogeneity up or down
for ablation experiments, and (c) scales cleanly to any number of clients.

We additionally record each client's age/sex/cp distribution purely for
descriptive reporting in the research write-up (to show that label-skew
partitioning also produces plausible demographic heterogeneity across
hospitals, not just an abstract class imbalance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HospitalPartition:
    client_id: int
    name: str
    indices: np.ndarray
    x: pd.DataFrame
    y: pd.Series
    demographics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)


def dirichlet_label_partition(
    y: pd.Series,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    min_size: int = 10,
) -> list[np.ndarray]:
    """
    Partitions sample indices into n_clients groups with non-IID class
    proportions drawn from a Dirichlet(alpha) distribution per class.

    Returns a list of length n_clients, each an array of row indices (into
    the original y) assigned to that client. Retries with a fresh draw if
    any resulting client would be smaller than `min_size`, since
    highly-skewed low-alpha draws can occasionally starve a client.
    """
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    classes = np.unique(y_arr)

    for attempt in range(50):
        client_indices: list[list[int]] = [[] for _ in range(n_clients)]

        for c in classes:
            class_idx = np.where(y_arr == c)[0]
            rng.shuffle(class_idx)

            proportions = rng.dirichlet(alpha=[alpha] * n_clients)
            # Convert proportions to cumulative split points over this class's samples.
            split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            class_splits = np.split(class_idx, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_size:
            return [np.array(sorted(idx)) for idx in client_indices]

    raise RuntimeError(
        f"Could not produce a partition with every client >= {min_size} samples "
        f"after 50 attempts (alpha={alpha}, n_clients={n_clients}). Try a larger "
        f"alpha, fewer clients, or a lower min_size."
    )


def summarize_partition(
    partitions: list[HospitalPartition], df_clean: pd.DataFrame
) -> pd.DataFrame:
    """Builds a summary table (rows = hospitals) for the research write-up."""
    rows = []
    for p in partitions:
        subset = df_clean.iloc[p.indices]
        disease_rate = (subset["target"] > 0).mean()
        rows.append({
            "hospital": p.name,
            "n_patients": len(p),
            "disease_rate": round(float(disease_rate), 3),
            "mean_age": round(float(subset["age"].mean()), 1),
            "pct_female": round(float((subset["sex"] == 0).mean()), 3),
            "dominant_cp_type": int(subset["cp"].mode().iloc[0]),
        })
    return pd.DataFrame(rows)


def create_hospital_partitions(
    x: pd.DataFrame,
    y: pd.Series,
    df_clean: pd.DataFrame,
    n_clients: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[HospitalPartition]:
    """
    Main entry point: produces n_clients non-IID HospitalPartition objects
    from the full preprocessed dataset (x, y) plus the un-encoded df_clean
    (used only to compute descriptive demographics per hospital).
    """
    index_groups = dirichlet_label_partition(y, n_clients, alpha, seed)

    partitions = []
    for i, indices in enumerate(index_groups):
        name = f"hospital_{i + 1}"
        subset_x = x.iloc[indices].reset_index(drop=True)
        subset_y = y.iloc[indices].reset_index(drop=True)
        partitions.append(HospitalPartition(
            client_id=i,
            name=name,
            indices=indices,
            x=subset_x,
            y=subset_y,
        ))
    return partitions


if __name__ == "__main__":
    try:
        from load_dataset import load_and_preprocess
    except ImportError:
        from src.data.load_dataset import load_and_preprocess

    X, y, df_clean = load_and_preprocess()

    for alpha in [0.1, 0.5, 5.0]:
        print(f"\n=== Dirichlet alpha={alpha} ===")
        parts = create_hospital_partitions(X, y, df_clean, n_clients=5, alpha=alpha)
        summary = summarize_partition(parts, df_clean)
        print(summary.to_string(index=False))
