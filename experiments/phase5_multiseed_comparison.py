"""
Phase 5 robustness check: reruns Local NN / FedAvg NN / FedProx NN /
Personalized (FedProx + fine-tuning) / Centralized NN across the SAME 5
MODEL_INIT_SEED values used in experiments/phase4_multiseed_comparison.py
(42, 1, 7, 123, 2024), reusing that verified structure so the eventual
5-way comparison stays on the same footing Phase 4 established.

Only the model-initialization seed varies -- the hospitals (the 4 real
UCI sites) and local train/test split (seed=42) stay fixed.

Run from the project root (takes several minutes: 5 seeds x (1 FedAvg sim +
1 FedProx sim + 1 fine-tuning pass + 2 NN baseline trainings)):
    venv\\Scripts\\python.exe experiments\\phase5_multiseed_comparison.py
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
from src.federated.fedprox_runner import run_fedprox_simulation
from src.federated.personalize import personalize_per_hospital
from src.models.nn_baseline_runner import run_centralized_nn, run_local_nn

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# Clients are the 4 real UCI sites (src/data/multisite.py) -- no synthetic partition.
N_CLIENTS = 4
SPLIT_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS_PER_ROUND = 5
TOTAL_EPOCHS = N_ROUNDS * LOCAL_EPOCHS_PER_ROUND
LEARNING_RATE = 0.01
PROXIMAL_MU = 0.1
FINE_TUNE_EPOCHS = 10

MODEL_INIT_SEEDS = [42, 1, 7, 123, 2024]  # same seeds as experiments/phase4_multiseed_comparison.py


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_multisite()
    partitions = create_site_partitions(x, y, df_clean)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    print(f"Running Local / FedAvg / FedProx / Personalized / Centralized NN across "
          f"{len(MODEL_INIT_SEEDS)} model-init seeds: {MODEL_INIT_SEEDS}\n")

    rows = []
    for seed in MODEL_INIT_SEEDS:
        print(f"--- seed={seed} ---")

        local_df = run_local_nn(partitions, n_features, seed=seed, epochs=TOTAL_EPOCHS, lr=LEARNING_RATE)
        local_acc = weighted_accuracy(local_df)

        fedavg_round_df = run_fedavg_simulation(
            partitions, n_features, model_init_seed=seed,
            n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS_PER_ROUND, lr=LEARNING_RATE,
        )
        fedavg_final = fedavg_round_df[fedavg_round_df["round"] == fedavg_round_df["round"].max()]
        fedavg_acc = weighted_accuracy(fedavg_final)

        fedprox_round_df, final_global_ndarrays = run_fedprox_simulation(
            partitions, n_features, model_init_seed=seed, proximal_mu=PROXIMAL_MU,
            n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS_PER_ROUND, lr=LEARNING_RATE,
        )
        fedprox_final = fedprox_round_df[fedprox_round_df["round"] == fedprox_round_df["round"].max()]
        fedprox_acc = weighted_accuracy(fedprox_final)

        personalized_df = personalize_per_hospital(
            final_global_ndarrays, partitions, n_features,
            fine_tune_epochs=FINE_TUNE_EPOCHS, lr=LEARNING_RATE,
        )
        personalized_acc = weighted_accuracy(personalized_df)

        central_overall, _ = run_centralized_nn(partitions, n_features, seed=seed, epochs=TOTAL_EPOCHS, lr=LEARNING_RATE)
        central_acc = central_overall["accuracy"]

        print(f"  Local={local_acc:.4f}  FedAvg={fedavg_acc:.4f}  FedProx={fedprox_acc:.4f}  "
              f"Personalized={personalized_acc:.4f}  Centralized={central_acc:.4f}")
        rows.append({
            "seed": seed,
            "local_nn_accuracy": local_acc,
            "fedavg_nn_accuracy": fedavg_acc,
            "fedprox_nn_accuracy": fedprox_acc,
            "personalized_accuracy": personalized_acc,
            "centralized_nn_accuracy": central_acc,
        })

    results_df = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "phase5_multiseed_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nPer-seed results saved to {results_path}")
    print(results_df.to_string(index=False))

    experiments = {
        "Local NN": "local_nn_accuracy",
        "FedAvg NN": "fedavg_nn_accuracy",
        "FedProx NN": "fedprox_nn_accuracy",
        "Personalized\n(FedProx+FT)": "personalized_accuracy",
        "Centralized NN": "centralized_nn_accuracy",
    }
    summary_rows = []
    for label, col in experiments.items():
        values = results_df[col]
        summary_rows.append({
            "experiment": label.replace("\n", " "),
            "mean_accuracy": round(float(values.mean()), 4),
            "std_accuracy": round(float(values.std(ddof=1)), 4),
            "min_accuracy": round(float(values.min()), 4),
            "max_accuracy": round(float(values.max()), 4),
            "n_seeds": len(values),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "phase5_multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMean +/- std across {len(MODEL_INIT_SEEDS)} seeds:")
    print(summary_df.to_string(index=False))
    print(f"Saved to {summary_path}")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#C44E52", "#4C72B0", "#8172B2", "#CCB974", "#55A868"]
    x_pos = np.arange(len(summary_df))
    ax.bar(x_pos, summary_df["mean_accuracy"], yerr=summary_df["std_accuracy"], capsize=6, color=colors)

    for i, (label, col) in enumerate(experiments.items()):
        jitter = np.random.default_rng(0).uniform(-0.05, 0.05, size=len(results_df))
        ax.scatter(np.full(len(results_df), i) + jitter, results_df[col],
                   color="black", zorder=3, s=25, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([l.replace("\n", " ") for l in experiments.keys()])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Phase 5: 5-way comparison, mean ± std across {len(MODEL_INIT_SEEDS)} seeds\n"
                 f"(black dots = individual seed results, same NN architecture throughout)")
    for i, row in summary_df.iterrows():
        ax.text(i, row["mean_accuracy"] + row["std_accuracy"] + 0.02,
                f"{row['mean_accuracy']:.3f}±{row['std_accuracy']:.3f}", ha="center", fontsize=8)
    fig.tight_layout()

    fig_path = RESULTS_DIR / "phase5_multiseed_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
