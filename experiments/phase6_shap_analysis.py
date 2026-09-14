"""
Phase 6 deliverable: SHAP explainability on top of the best-performing
models from Phases 3-5.

Two analyses, per the proposal's stated plan (TreeExplainer for the
sklearn baseline, KernelExplainer for the PyTorch model):

1. Global explainability: SHAP TreeExplainer on the Centralized ML Random
   Forest (Phase 3's RF baseline, retrained here identically) over the
   full pooled test set -- global feature importance ranking + one example
   patient's ranked contributions ("Cholesterol +23%, Age +17%"-style).
2. Personalized explainability: SHAP KernelExplainer on hospital_1's
   Phase 5 personalized NN model (the hospital whose accuracy personalization
   most reliably improved -- see the Phase 5 equity analysis) over a small
   sample of its own local test patients.

Both use the SAME seed=117 partition / seed=42 local split as every prior
phase. Clinical-plausibility commentary (do the top features make medical
sense?) is written in the README after inspecting this script's actual
output -- not assumed in advance.

Run from the project root, ideally after Phase 3-5 scripts:
    venv\\Scripts\\python.exe experiments\\phase6_shap_analysis.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_dataset import load_and_preprocess
from src.data.partition import add_local_train_test_split, create_hospital_partitions
from src.explainability.shap_utils import (
    compute_kernel_shap_nn,
    compute_tree_shap,
    explain_single_patient,
    summarize_importance,
)
from src.federated.fedprox_runner import run_fedprox_simulation
from src.federated.personalize import personalize_per_hospital
from src.models.baseline import train_random_forest

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

N_CLIENTS = 5
ALPHA = 0.5
PARTITION_SEED = 117
SPLIT_SEED = 42
MODEL_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.01
PROXIMAL_MU = 0.1
FINE_TUNE_EPOCHS = 10

TARGET_HOSPITAL = "hospital_1"  # most reliably helped by personalization, per Phase 5's equity analysis
EXAMPLE_PATIENT_INDEX = 0        # first patient in the pooled test set / target hospital's test set


def run_rf_global_explainability(partitions, n_features: int) -> None:
    print(f"\n{'=' * 70}\n1. Global explainability: SHAP TreeExplainer on Centralized ML Random Forest\n{'=' * 70}")

    x_train_pooled = pd.concat([p.x_train for p in partitions], ignore_index=True)
    y_train_pooled = pd.concat([p.y_train for p in partitions], ignore_index=True)
    x_test_pooled = pd.concat([p.x_test for p in partitions], ignore_index=True)

    model = train_random_forest(x_train_pooled, y_train_pooled, seed=MODEL_SEED)
    explainer, shap_values = compute_tree_shap(model, x_test_pooled)

    importance_df = summarize_importance(shap_values, list(x_test_pooled.columns))
    importance_path = RESULTS_DIR / "phase6_shap_rf_global_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print("\nGlobal feature importance (mean |SHAP value|), top 10:")
    print(importance_df.head(10).to_string(index=False))
    print(f"Saved to {importance_path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance_df.head(12).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title("Phase 6: Global feature importance\n(SHAP TreeExplainer, Centralized ML Random Forest)")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "phase6_shap_rf_global_importance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")
    plt.close(fig)

    base_value = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)
    patient_df = explain_single_patient(shap_values, x_test_pooled, EXAMPLE_PATIENT_INDEX, base_value)
    patient_path = RESULTS_DIR / "phase6_shap_rf_example_patient.csv"
    patient_df.to_csv(patient_path, index=False)
    print(f"\nExample patient #{EXAMPLE_PATIENT_INDEX} (pooled test set) -- "
          f"base rate={patient_df.attrs['base_value']:.3f}, "
          f"predicted={patient_df.attrs['predicted_value']:.3f}:")
    print(patient_df.to_string(index=False))
    print(f"Saved to {patient_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in patient_df["shap_value"]]
    ax.barh(patient_df["feature"], patient_df["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on predicted disease probability)")
    ax.set_title(f"Phase 6: Example patient explanation\n(red = pushes toward disease, blue = pushes away)")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "phase6_shap_rf_example_patient.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")
    plt.close(fig)


def run_personalized_nn_explainability(partitions, n_features: int) -> None:
    print(f"\n{'=' * 70}\n2. Personalized explainability: SHAP KernelExplainer on {TARGET_HOSPITAL}'s personalized NN\n{'=' * 70}")

    print(f"Retraining FedProx (mu={PROXIMAL_MU}) + personalizing {TARGET_HOSPITAL} "
          f"(same as Phase 5, seed={MODEL_SEED})...")
    _, final_global_ndarrays = run_fedprox_simulation(
        partitions, n_features, model_init_seed=MODEL_SEED, proximal_mu=PROXIMAL_MU,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE,
    )

    from src.models.nn_model import HeartDiseaseNet, set_model_parameters
    from src.models.nn_train import local_train

    target_partition = next(p for p in partitions if p.name == TARGET_HOSPITAL)
    model = HeartDiseaseNet(n_features)
    set_model_parameters(model, final_global_ndarrays)
    local_train(model, target_partition.x_train, target_partition.y_train, epochs=FINE_TUNE_EPOCHS, lr=LEARNING_RATE)

    explainer, shap_values = compute_kernel_shap_nn(
        model, target_partition.x_train, target_partition.x_test,
        n_background=min(20, len(target_partition.x_train)), nsamples=100, seed=MODEL_SEED,
    )

    importance_df = summarize_importance(shap_values, list(target_partition.x_test.columns))
    importance_path = RESULTS_DIR / f"phase6_shap_nn_{TARGET_HOSPITAL}_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\n{TARGET_HOSPITAL} personalized NN -- feature importance (mean |SHAP value|), top 10:")
    print(importance_df.head(10).to_string(index=False))
    print(f"Saved to {importance_path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    top = importance_df.head(12).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#CCB974")
    ax.set_xlabel("Mean |SHAP value| (average impact on predicted probability)")
    ax.set_title(f"Phase 6: {TARGET_HOSPITAL} personalized NN feature importance\n(SHAP KernelExplainer)")
    fig.tight_layout()
    fig_path = RESULTS_DIR / f"phase6_shap_nn_{TARGET_HOSPITAL}_importance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_and_preprocess()
    partitions = create_hospital_partitions(x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=PARTITION_SEED)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
    n_features = x.shape[1]

    run_rf_global_explainability(partitions, n_features)
    run_personalized_nn_explainability(partitions, n_features)

    print(f"\n{'=' * 70}\nPhase 6 complete. See experiments/results/phase6_*.csv and *.png")


if __name__ == "__main__":
    main()
