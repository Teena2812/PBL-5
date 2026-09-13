"""
Phase 4 deliverable: Federated Learning (FedAvg) via Flower's simulation
engine (flwr.simulation.run_simulation, in-process virtual clients -- no
separate OS processes per hospital, matching the locked tech stack).

Uses the SAME hospital partition and local train/test split as Phase 3
(seed=117 Dirichlet partition, seed=42 local 75/25 split), so FedAvg's
per-hospital results are directly comparable to Local ML and Centralized ML.

FedAvg here is plain (mu=0, no FedProx proximal term) -- personalization is
Phase 5. Aggregation is Flower's default: each round, every client trains
HeartDiseaseNet locally for a few epochs, and the server averages weights
weighted by each client's number of local training examples
(theta_global = sum(n_k / n * theta_k)).

Run from the project root:
    venv\\Scripts\\python.exe experiments\\phase4_fedavg.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_dataset import load_and_preprocess
from src.data.partition import add_local_train_test_split, create_hospital_partitions
from src.federated.fedavg_runner import run_fedavg_simulation, weighted_accuracy

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

N_CLIENTS = 5
ALPHA = 0.5
PARTITION_SEED = 117   # same as Phase 3: balanced sizes, no single-class hospitals
SPLIT_SEED = 42         # same local train/test split as Phase 3
MODEL_INIT_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.01


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_and_preprocess()
    partitions = create_hospital_partitions(
        x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=PARTITION_SEED
    )
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    print("Hospital local train/test sizes (same partition as Phase 3):")
    for p in partitions:
        print(f"  {p.name}: train={len(p.x_train)}, test={len(p.x_test)}")

    print(f"\nRunning FedAvg simulation: {N_ROUNDS} rounds, {N_CLIENTS} clients, "
          f"{LOCAL_EPOCHS} local epochs/round, lr={LEARNING_RATE}\n")

    per_round_df = run_fedavg_simulation(
        partitions, n_features, model_init_seed=MODEL_INIT_SEED,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE,
    )
    per_round_path = RESULTS_DIR / "phase4_fedavg_per_round_per_hospital.csv"
    per_round_df.to_csv(per_round_path, index=False)
    print(f"\nSaved per-round, per-hospital results to {per_round_path}")

    # Global weighted accuracy per round (sample-size weighted, same
    # convention as FedAvg's own aggregation), for the learning-curve chart.
    round_summary = (
        per_round_df.groupby("round")
        .apply(weighted_accuracy, include_groups=False)
        .reset_index(name="global_weighted_accuracy")
    )
    round_summary_path = RESULTS_DIR / "phase4_fedavg_round_summary.csv"
    round_summary.to_csv(round_summary_path, index=False)
    print(f"Saved per-round global accuracy to {round_summary_path}")
    print(round_summary.to_string(index=False))

    final_round = per_round_df["round"].max()
    final_per_hospital = per_round_df[per_round_df["round"] == final_round].drop(columns=["round"])
    final_path = RESULTS_DIR / "phase4_fedavg_final_per_hospital.csv"
    final_per_hospital.to_csv(final_path, index=False)
    print(f"\n[FedAvg] final-round (round {final_round}) per-hospital results:")
    print(final_per_hospital.to_string(index=False))
    print(f"Saved to {final_path}")

    final_global_accuracy = weighted_accuracy(final_per_hospital)
    print(f"\n[FedAvg] final global weighted accuracy: {final_global_accuracy:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(round_summary["round"], round_summary["global_weighted_accuracy"],
              marker="o", color="#4C72B0")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Global weighted accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Phase 4: FedAvg learning curve ({N_ROUNDS} rounds)")
    ax1.grid(alpha=0.3)

    ax2.bar(final_per_hospital["hospital"], final_per_hospital["accuracy"], color="#55A868")
    ax2.axhline(final_global_accuracy, color="black", linestyle="--", linewidth=1,
                label=f"global weighted = {final_global_accuracy:.3f}")
    ax2.set_ylabel("Accuracy (final round)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Per-hospital accuracy, final round")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend()

    fig.tight_layout()
    fig_path = RESULTS_DIR / "phase4_fedavg_learning_curve.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
