"""
Phase 5 equity analysis: for each of the same 5 seeds used in
experiments/phase5_multiseed_comparison.py, identifies the worst-served
hospital under plain FedAvg (lowest per-hospital accuracy) and checks
whether FedProx + local fine-tuning (personalization) improved THAT
SPECIFIC hospital's accuracy -- the actual hypothesis from the original
problem statement (docs/proposal/01_Problem_Definition.md): personalization
should help the worst-served hospital, not necessarily move the global
average.

Note: experiments/phase5_multiseed_comparison.py only saved each seed's
GLOBAL weighted accuracy, not the per-hospital breakdown, so that data
isn't available to re-derive without rerunning -- this script captures the
per-hospital numbers this time. Same partition (seed=117), same local
split (seed=42), same 5 MODEL_INIT_SEEDs (42, 1, 7, 123, 2024) as every
other Phase 4-5 script.

Run from the project root:
    venv\\Scripts\\python.exe experiments\\phase5_equity_analysis.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.load_dataset import load_and_preprocess
from src.data.partition import add_local_train_test_split, create_hospital_partitions
from src.federated.fedavg_runner import run_fedavg_simulation
from src.federated.fedprox_runner import run_fedprox_simulation
from src.federated.personalize import personalize_per_hospital

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

N_CLIENTS = 5
ALPHA = 0.5
PARTITION_SEED = 117
SPLIT_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS_PER_ROUND = 5
LEARNING_RATE = 0.01
PROXIMAL_MU = 0.1
FINE_TUNE_EPOCHS = 10

MODEL_INIT_SEEDS = [42, 1, 7, 123, 2024]  # same seeds as phase4/phase5_multiseed_comparison.py


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_and_preprocess()
    partitions = create_hospital_partitions(x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=PARTITION_SEED)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    print(f"Equity analysis across {len(MODEL_INIT_SEEDS)} seeds: {MODEL_INIT_SEEDS}\n"
          f"For each seed: find the worst-served hospital under FedAvg, "
          f"check whether personalization (FedProx+fine-tuning) helped THAT hospital.\n")

    rows = []
    for seed in MODEL_INIT_SEEDS:
        print(f"--- seed={seed} ---")

        fedavg_round_df = run_fedavg_simulation(
            partitions, n_features, model_init_seed=seed,
            n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS_PER_ROUND, lr=LEARNING_RATE,
        )
        fedavg_final = fedavg_round_df[fedavg_round_df["round"] == fedavg_round_df["round"].max()]

        fedprox_round_df, final_global_ndarrays = run_fedprox_simulation(
            partitions, n_features, model_init_seed=seed, proximal_mu=PROXIMAL_MU,
            n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS_PER_ROUND, lr=LEARNING_RATE,
        )
        personalized_df = personalize_per_hospital(
            final_global_ndarrays, partitions, n_features,
            fine_tune_epochs=FINE_TUNE_EPOCHS, lr=LEARNING_RATE,
        )

        worst_row = fedavg_final.loc[fedavg_final["accuracy"].idxmin()]
        worst_hospital = worst_row["hospital"]
        fedavg_worst_acc = float(worst_row["accuracy"])

        personalized_worst_acc = float(
            personalized_df.loc[personalized_df["hospital"] == worst_hospital, "accuracy"].iloc[0]
        )
        delta = personalized_worst_acc - fedavg_worst_acc

        print(f"  worst-served hospital under FedAvg: {worst_hospital} (accuracy={fedavg_worst_acc:.4f})")
        print(f"  same hospital after personalization: accuracy={personalized_worst_acc:.4f}  "
              f"(delta={delta:+.4f})")

        rows.append({
            "seed": seed,
            "worst_hospital": worst_hospital,
            "fedavg_accuracy": round(fedavg_worst_acc, 4),
            "personalized_accuracy": round(personalized_worst_acc, 4),
            "delta": round(delta, 4),
            "improved": delta > 0,
        })

    results_df = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "phase5_equity_analysis.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 70}")
    print("Equity metric: does personalization help the worst-served hospital?")
    print(results_df.to_string(index=False))

    n_improved = int(results_df["improved"].sum())
    n_seeds = len(results_df)
    mean_delta_all = results_df["delta"].mean()
    improved_df = results_df[results_df["improved"]]
    mean_delta_improved = improved_df["delta"].mean() if len(improved_df) > 0 else float("nan")

    print(f"\nPersonalization improved the worst-performing hospital in "
          f"{n_improved}/{n_seeds} seeds.")
    print(f"Mean delta across ALL {n_seeds} seeds (improved + not): "
          f"{mean_delta_all:+.4f} ({mean_delta_all*100:+.2f} pp)")
    if n_improved > 0:
        print(f"Mean delta among the {n_improved} seed(s) where it improved: "
              f"{mean_delta_improved:+.4f} ({mean_delta_improved*100:+.2f} pp)")

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
