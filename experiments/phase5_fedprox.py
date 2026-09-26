"""
Phase 5 deliverable: Personalization via FedProx (proximal term during
federated training) + local fine-tuning (post-training per-hospital
adaptation), compared against Phase 4's plain FedAvg.

Uses the SAME hospital partition, local train/test split, model
architecture, round count, and local epoch count as Phase 4
(4 real sites / seed=42 / HeartDiseaseNet / 20 rounds / 5 local epochs), so
FedAvg vs FedProx vs Personalized are directly comparable -- proximal_mu
is the only new variable, plus the fine-tuning step afterward.

Run from the project root, ideally after experiments/phase4_fedavg.py and
experiments/phase4_nn_baselines.py (for the comparison table):
    venv\\Scripts\\python.exe experiments\\phase5_fedprox.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.multisite import create_site_partitions, load_multisite
from src.data.partition import add_local_train_test_split
from src.federated.fedavg_runner import weighted_accuracy
from src.federated.fedprox_runner import run_fedprox_simulation
from src.federated.personalize import personalize_per_hospital

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# Clients are the 4 real UCI sites (src/data/multisite.py) -- no synthetic partition.
N_CLIENTS = 4
SPLIT_SEED = 42
MODEL_INIT_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.01
PROXIMAL_MU = 0.1   # chosen after an informal sweep over {0, 0.01, 0.1, 1.0} -- see README

FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 0.01


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_multisite()
    partitions = create_site_partitions(x, y, df_clean)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    print(f"Running FedProx simulation: {N_ROUNDS} rounds, {N_CLIENTS} clients, "
          f"{LOCAL_EPOCHS} local epochs/round, lr={LEARNING_RATE}, proximal_mu={PROXIMAL_MU}\n")

    per_round_df, final_global_ndarrays = run_fedprox_simulation(
        partitions, n_features, model_init_seed=MODEL_INIT_SEED, proximal_mu=PROXIMAL_MU,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE,
    )
    per_round_path = RESULTS_DIR / "phase5_fedprox_per_round_per_hospital.csv"
    per_round_df.to_csv(per_round_path, index=False)
    print(f"Saved per-round, per-hospital FedProx results to {per_round_path}")

    round_summary = (
        per_round_df.groupby("round").apply(weighted_accuracy, include_groups=False)
        .reset_index(name="global_weighted_accuracy")
    )
    round_summary_path = RESULTS_DIR / "phase5_fedprox_round_summary.csv"
    round_summary.to_csv(round_summary_path, index=False)
    print(round_summary.to_string(index=False))

    final_round = per_round_df["round"].max()
    fedprox_final = per_round_df[per_round_df["round"] == final_round].drop(columns=["round"])
    fedprox_final_path = RESULTS_DIR / "phase5_fedprox_final_per_hospital.csv"
    fedprox_final.to_csv(fedprox_final_path, index=False)
    fedprox_global_acc = weighted_accuracy(fedprox_final)
    print(f"\n[FedProx] final-round (round {final_round}) per-hospital results:")
    print(fedprox_final.to_string(index=False))
    print(f"[FedProx] final global weighted accuracy: {fedprox_global_acc:.4f}")

    # --- Personalization: fine-tune the trained global model per hospital ---
    print(f"\nFine-tuning the FedProx global model per hospital "
          f"({FINE_TUNE_EPOCHS} epochs, lr={FINE_TUNE_LR}, local data only, no proximal term)\n")
    personalized_df = personalize_per_hospital(
        final_global_ndarrays, partitions, n_features,
        fine_tune_epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR,
    )
    personalized_path = RESULTS_DIR / "phase5_personalized_per_hospital.csv"
    personalized_df.to_csv(personalized_path, index=False)
    print("[Personalized FL] per-hospital results:")
    print(personalized_df.to_string(index=False))
    personalized_acc = weighted_accuracy(personalized_df)
    print(f"[Personalized FL] global weighted accuracy (mean of personalized models): {personalized_acc:.4f}")
    print(f"Saved to {personalized_path}")

    # --- Comparison chart: FedProx (shared global model) vs Personalized, per hospital ---
    merged = fedprox_final[["hospital", "accuracy"]].merge(
        personalized_df[["hospital", "accuracy"]], on="hospital", suffixes=("_fedprox", "_personalized")
    )
    merged_path = RESULTS_DIR / "phase5_fedprox_vs_personalized.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\nFedProx (shared) vs Personalized (fine-tuned), per hospital:")
    print(merged.to_string(index=False))
    print(f"Saved to {merged_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(round_summary["round"], round_summary["global_weighted_accuracy"],
              marker="o", color="#8172B2", label=f"FedProx (mu={PROXIMAL_MU})")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Global weighted accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Phase 5: FedProx learning curve ({N_ROUNDS} rounds)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    x_pos = np.arange(len(merged))
    width = 0.35
    ax2.bar(x_pos - width / 2, merged["accuracy_fedprox"], width, label="FedProx (shared global)", color="#8172B2")
    ax2.bar(x_pos + width / 2, merged["accuracy_personalized"], width, label="Personalized (fine-tuned)", color="#CCB974")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(merged["hospital"], rotation=30)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("Per-hospital: shared global vs personalized")
    ax2.legend()

    fig.tight_layout()
    fig_path = RESULTS_DIR / "phase5_fedprox_personalization.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
