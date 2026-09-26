"""
Phase 4 (addendum): Local ML and Centralized ML baselines using the SAME
PyTorch NN architecture (HeartDiseaseNet) as FedAvg, so the
Local <= FedAvg <= Centralized hypothesis is tested like-for-like.

Why this exists: Phase 3's Local ML / Centralized ML baselines use sklearn
Random Forest / Logistic Regression, while Phase 4's FedAvg necessarily
uses a PyTorch NN (FedAvg averages weight tensors, which only makes sense
for a fixed neural network architecture -- it has no RF/LR equivalent).
Comparing FedAvg's NN accuracy directly against the RF/LR Local ML number
mixes two variables (algorithm choice AND collaboration strategy) into one
comparison, so an apparent "Local ML beats FedAvg" result could just be
"Random Forest beats a small NN on this tiny dataset" and tell us nothing
about federation. This script isolates the collaboration-strategy variable
by holding the model architecture fixed at HeartDiseaseNet for all three
settings (Local NN, FedAvg NN, Centralized NN).

Fairness of the comparison: each hospital's Local NN and the pooled
Centralized NN are trained for TOTAL_EPOCHS epochs (== N_ROUNDS *
LOCAL_EPOCHS_PER_ROUND from experiments/phase4_fedavg.py), the same total
number of local gradient-update epochs any single FedAvg client
experiences over the full run, with the same learning rate and the same
4 real UCI sites / seed=42 local split as Phase 3-4.

Run from the project root, after experiments/phase4_fedavg.py:
    venv\\Scripts\\python.exe experiments\\phase4_nn_baselines.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.multisite import create_site_partitions, load_multisite
from src.data.partition import add_local_train_test_split
from src.models.nn_baseline_runner import run_centralized_nn, run_local_nn

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# Clients are the 4 real UCI sites (src/data/multisite.py) -- no synthetic partition.
N_CLIENTS = 4
SPLIT_SEED = 42
MODEL_SEED = 42

N_ROUNDS = 20            # must match experiments/phase4_fedavg.py
LOCAL_EPOCHS_PER_ROUND = 5
TOTAL_EPOCHS = N_ROUNDS * LOCAL_EPOCHS_PER_ROUND  # 100: same total local compute as one FedAvg client
LEARNING_RATE = 0.01


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_multisite()
    partitions = create_site_partitions(x, y, df_clean)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    print(f"Training Local NN and Centralized NN for {TOTAL_EPOCHS} epochs each "
          f"(= {N_ROUNDS} rounds x {LOCAL_EPOCHS_PER_ROUND} local epochs, matching FedAvg's per-client compute)\n")

    local_df = run_local_nn(partitions, n_features, seed=MODEL_SEED, epochs=TOTAL_EPOCHS, lr=LEARNING_RATE)
    local_path = RESULTS_DIR / "phase4_local_nn.csv"
    local_df.to_csv(local_path, index=False)
    print("[Local NN] per-hospital results:")
    print(local_df.to_string(index=False))
    print(f"Saved to {local_path}\n")

    central_overall, central_per_hospital = run_centralized_nn(partitions, n_features, seed=MODEL_SEED, epochs=TOTAL_EPOCHS, lr=LEARNING_RATE)
    central_path = RESULTS_DIR / "phase4_centralized_nn_per_hospital.csv"
    central_per_hospital.to_csv(central_path, index=False)
    print(f"[Centralized NN] overall (pooled test set): {central_overall}")
    print("[Centralized NN] per-hospital breakdown:")
    print(central_per_hospital.to_string(index=False))
    print(f"Saved to {central_path}\n")

    # Pull in the already-computed FedAvg result (experiments/phase4_fedavg.py)
    # for a single NN-only, apples-to-apples comparison across all three settings.
    fedavg_final_path = RESULTS_DIR / "phase4_fedavg_final_per_hospital.csv"
    if not fedavg_final_path.exists():
        raise FileNotFoundError(
            f"{fedavg_final_path} not found -- run experiments/phase4_fedavg.py first."
        )
    fedavg_df = pd.read_csv(fedavg_final_path)
    fedavg_weighted_acc = (fedavg_df["accuracy"] * fedavg_df["n_test"]).sum() / fedavg_df["n_test"].sum()

    local_weighted_acc = (local_df["accuracy"] * local_df["n_test"]).sum() / local_df["n_test"].sum()

    summary = pd.DataFrame([
        {"experiment": "Local NN (per-hospital, isolated)", "accuracy": round(float(local_weighted_acc), 4)},
        {"experiment": "FedAvg NN (collaborative, no personalization)", "accuracy": round(float(fedavg_weighted_acc), 4)},
        {"experiment": "Centralized NN (pooled, not privacy-preserving)", "accuracy": central_overall["accuracy"]},
    ])
    summary_path = RESULTS_DIR / "phase4_nn_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Apples-to-apples NN-only comparison (same HeartDiseaseNet architecture throughout):")
    print(summary.to_string(index=False))
    print(f"Saved to {summary_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C44E52", "#4C72B0", "#55A868"]
    ax.bar(summary["experiment"], summary["accuracy"], color=colors)
    ax.set_ylabel("Sample-weighted accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 4: Local vs FedAvg vs Centralized\n(same NN architecture, apples-to-apples)")
    ax.tick_params(axis="x", rotation=15)
    for i, v in enumerate(summary["accuracy"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()

    fig_path = RESULTS_DIR / "phase4_nn_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
