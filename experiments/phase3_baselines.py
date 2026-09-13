"""
Phase 3 deliverable: Local ML and Centralized ML baseline experiments.

- Local ML: each simulated hospital trains its own model on only its own
  local training data, evaluated on its own local held-out test data.
  No collaboration between hospitals -- this is the "no FL" lower-bound
  reference for the eventual 4-way comparison (Local < FedAvg <
  Personalized FL < Centralized, per the project hypothesis to be tested).
- Centralized ML: all hospitals' local training data is pooled into one
  training set and one model is trained -- this is the privacy-violating
  upper-bound reference (all data pooled, not privacy-preserving), used
  only for comparison purposes.

Both experiments use the SAME hospital partition and local train/test
split (seed=117 for the Dirichlet partition, chosen in Phase 2 for balanced
hospital sizes with no single-class hospitals; seed=42 for the local
train/test split) so that later FedAvg and Personalized FL experiments
(Phase 4-5) can reuse this exact protocol and produce directly comparable
numbers.

Run from the project root:
    venv\\Scripts\\python.exe experiments\\phase3_baselines.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_dataset import load_and_preprocess
from src.data.partition import add_local_train_test_split, create_hospital_partitions
from src.models.baseline import MODEL_FACTORIES, evaluate_model

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

N_CLIENTS = 5
ALPHA = 0.5
PARTITION_SEED = 117     # selected in Phase 2: balanced sizes AND every hospital
                          # has >= 5 samples of both classes (no single-class hospitals)
SPLIT_SEED = 42          # local train/test split seed
LOCAL_TEST_SIZE = 0.25
MODEL_SEED = 42


def run_local_ml(partitions, model_name: str) -> pd.DataFrame:
    """Each hospital trains and evaluates its own model, in isolation."""
    train_fn = MODEL_FACTORIES[model_name]
    rows = []
    for p in partitions:
        model = train_fn(p.x_train, p.y_train, seed=MODEL_SEED)
        metrics = evaluate_model(model, p.x_test, p.y_test)
        rows.append({"hospital": p.name, "n_train": len(p.x_train), **metrics})
    df = pd.DataFrame(rows)
    return df


def run_centralized_ml(partitions, model_name: str) -> tuple[dict, pd.DataFrame]:
    """
    Pools every hospital's local training data into one training set,
    trains a single model, then evaluates it two ways:
      1. overall, on the union of every hospital's local test set
      2. per-hospital, on each hospital's own local test set separately
         (shows how well one centralized model serves each hospital's
         population -- the gap this motivates personalization to close)
    """
    train_fn = MODEL_FACTORIES[model_name]

    x_train_pooled = pd.concat([p.x_train for p in partitions], ignore_index=True)
    y_train_pooled = pd.concat([p.y_train for p in partitions], ignore_index=True)
    x_test_pooled = pd.concat([p.x_test for p in partitions], ignore_index=True)
    y_test_pooled = pd.concat([p.y_test for p in partitions], ignore_index=True)

    model = train_fn(x_train_pooled, y_train_pooled, seed=MODEL_SEED)

    overall_metrics = evaluate_model(model, x_test_pooled, y_test_pooled)
    overall_metrics["n_train"] = len(x_train_pooled)

    per_hospital_rows = []
    for p in partitions:
        metrics = evaluate_model(model, p.x_test, p.y_test)
        per_hospital_rows.append({"hospital": p.name, **metrics})
    per_hospital_df = pd.DataFrame(per_hospital_rows)

    return overall_metrics, per_hospital_df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_and_preprocess()
    partitions = create_hospital_partitions(
        x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=PARTITION_SEED
    )
    partitions = add_local_train_test_split(
        partitions, test_size=LOCAL_TEST_SIZE, seed=SPLIT_SEED
    )

    print("Hospital local train/test sizes:")
    for p in partitions:
        print(f"  {p.name}: train={len(p.x_train)}, test={len(p.x_test)}")

    summary_rows = []

    for model_name in MODEL_FACTORIES:
        print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")

        local_df = run_local_ml(partitions, model_name)
        local_path = RESULTS_DIR / f"phase3_local_ml_{model_name}.csv"
        local_df.to_csv(local_path, index=False)
        print(f"\n[Local ML] per-hospital results:")
        print(local_df.to_string(index=False))
        print(f"Saved to {local_path}")

        central_overall, central_per_hospital = run_centralized_ml(partitions, model_name)
        central_path = RESULTS_DIR / f"phase3_centralized_ml_{model_name}_per_hospital.csv"
        central_per_hospital.to_csv(central_path, index=False)
        print(f"\n[Centralized ML] overall (pooled test set): {central_overall}")
        print(f"[Centralized ML] per-hospital breakdown:")
        print(central_per_hospital.to_string(index=False))
        print(f"Saved to {central_path}")

        local_mean_accuracy = local_df["accuracy"].mean()
        summary_rows.append({
            "model": model_name,
            "experiment": "Local ML (mean across hospitals)",
            "accuracy": round(float(local_mean_accuracy), 4),
        })
        summary_rows.append({
            "model": model_name,
            "experiment": "Centralized ML (pooled test set)",
            "accuracy": central_overall["accuracy"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "phase3_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nPhase 3 summary (Local ML vs Centralized ML):")
    print(summary_df.to_string(index=False))
    print(f"Saved to {summary_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(MODEL_FACTORIES.keys())
    x_pos = range(len(models))
    width = 0.35

    local_acc = [summary_df[(summary_df.model == m) & (summary_df.experiment.str.startswith("Local"))]["accuracy"].iloc[0] for m in models]
    central_acc = [summary_df[(summary_df.model == m) & (summary_df.experiment.str.startswith("Centralized"))]["accuracy"].iloc[0] for m in models]

    ax.bar([p - width / 2 for p in x_pos], local_acc, width, label="Local ML (mean/hospital)", color="#C44E52")
    ax.bar([p + width / 2 for p in x_pos], central_acc, width, label="Centralized ML (pooled)", color="#55A868")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 3: Local ML vs Centralized ML Baseline Accuracy")
    ax.legend()
    fig.tight_layout()

    fig_path = RESULTS_DIR / "phase3_local_vs_centralized.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
