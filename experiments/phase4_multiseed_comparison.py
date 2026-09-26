"""
Phase 4 robustness check: reruns the apples-to-apples NN comparison (Local
NN / FedAvg NN / Centralized NN, all HeartDiseaseNet) across 5 different
MODEL_INIT_SEED values to check whether FedAvg's slight edge seen at
seed=42 (0.829 vs 0.816/0.816) holds consistently, or was noise from that
one seed's random weight initialization.

Only the model-initialization seed varies across runs -- the hospitals
(the 4 real UCI sites) and local train/test split (seed=42) stay
fixed throughout, matching every other Phase 3-4 script, so the ONLY
variable being tested here is sensitivity to initial NN weights.

Run from the project root (takes a few minutes: 5 seeds x (1 FedAvg
simulation + 2 NN baseline trainings)):
    venv\\Scripts\\python.exe experiments\\phase4_multiseed_comparison.py
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
from src.federated.fedavg_runner import run_fedavg_simulation, weighted_accuracy
from src.models.nn_baseline_runner import run_centralized_nn, run_local_nn

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# Clients are the 4 real UCI sites (src/data/multisite.py) -- no synthetic partition.
N_CLIENTS = 4
SPLIT_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS_PER_ROUND = 5
TOTAL_EPOCHS = N_ROUNDS * LOCAL_EPOCHS_PER_ROUND
LEARNING_RATE = 0.01

MODEL_INIT_SEEDS = [42, 1, 7, 123, 2024]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_multisite()
    partitions = create_site_partitions(x, y, df_clean)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    print(f"Running Local NN / FedAvg NN / Centralized NN across "
          f"{len(MODEL_INIT_SEEDS)} model-init seeds: {MODEL_INIT_SEEDS}\n"
          f"(hospital partition and local train/test split held fixed throughout)\n")

    rows = []
    for seed in MODEL_INIT_SEEDS:
        print(f"--- seed={seed} ---")

        local_df = run_local_nn(partitions, n_features, seed=seed, epochs=TOTAL_EPOCHS, lr=LEARNING_RATE)
        local_acc = weighted_accuracy(local_df)

        per_round_df = run_fedavg_simulation(
            partitions, n_features, model_init_seed=seed,
            n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS_PER_ROUND, lr=LEARNING_RATE,
        )
        final_round = per_round_df["round"].max()
        fedavg_final = per_round_df[per_round_df["round"] == final_round]
        fedavg_acc = weighted_accuracy(fedavg_final)

        central_overall, _ = run_centralized_nn(partitions, n_features, seed=seed, epochs=TOTAL_EPOCHS, lr=LEARNING_RATE)
        central_acc = central_overall["accuracy"]

        print(f"  Local NN={local_acc:.4f}  FedAvg NN={fedavg_acc:.4f}  Centralized NN={central_acc:.4f}")
        rows.append({
            "seed": seed,
            "local_nn_accuracy": local_acc,
            "fedavg_nn_accuracy": fedavg_acc,
            "centralized_nn_accuracy": central_acc,
        })

    results_df = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "phase4_multiseed_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nPer-seed results saved to {results_path}")
    print(results_df.to_string(index=False))

    experiments = {
        "Local NN": "local_nn_accuracy",
        "FedAvg NN": "fedavg_nn_accuracy",
        "Centralized NN": "centralized_nn_accuracy",
    }
    summary_rows = []
    for label, col in experiments.items():
        values = results_df[col]
        summary_rows.append({
            "experiment": label,
            "mean_accuracy": round(float(values.mean()), 4),
            "std_accuracy": round(float(values.std(ddof=1)), 4),
            "min_accuracy": round(float(values.min()), 4),
            "max_accuracy": round(float(values.max()), 4),
            "n_seeds": len(values),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "phase4_multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMean +/- std across {len(MODEL_INIT_SEEDS)} seeds:")
    print(summary_df.to_string(index=False))
    print(f"Saved to {summary_path}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = ["#C44E52", "#4C72B0", "#55A868"]
    x_pos = np.arange(len(summary_df))
    ax.bar(x_pos, summary_df["mean_accuracy"], yerr=summary_df["std_accuracy"],
           capsize=6, color=colors)

    # overlay individual seed results as scatter points to show the spread directly
    for i, (label, col) in enumerate(experiments.items()):
        jitter = np.random.default_rng(0).uniform(-0.05, 0.05, size=len(results_df))
        ax.scatter(np.full(len(results_df), i) + jitter, results_df[col],
                   color="black", zorder=3, s=25, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df["experiment"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Phase 4 robustness check: mean ± std across {len(MODEL_INIT_SEEDS)} seeds\n"
                 f"(black dots = individual seed results, same architecture throughout)")
    for i, row in summary_df.iterrows():
        ax.text(i, row["mean_accuracy"] + row["std_accuracy"] + 0.02,
                f"{row['mean_accuracy']:.3f}±{row['std_accuracy']:.3f}", ha="center", fontsize=9)
    fig.tight_layout()

    fig_path = RESULTS_DIR / "phase4_multiseed_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
