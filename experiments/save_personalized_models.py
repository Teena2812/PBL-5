"""
Phase 7 prep: saves each hospital's final PERSONALIZED model (Phase 5:
FedProx + local fine-tuning) to disk as a PyTorch state_dict, plus the
preprocessing artifact needed to transform a new raw patient record into
the model's expected input vector.

Nothing computed here is new -- it's the exact same FedProx + personalize
pipeline as experiments/phase5_fedprox.py (same 4 real UCI sites,
seed=42 local split, seed=42 model init, mu=0.1, 20 rounds, 10 fine-tune
epochs), just also saving the trained model objects instead of discarding
them after computing metrics. The printed personalized metrics below
should exactly match phase5_fedprox.py's [Personalized FL] results as a
sanity check that nothing changed.

These checkpoints back Phase 7's live single-patient prediction endpoint:
inference and single-instance SHAP explanation only -- NO retraining ever
happens in that endpoint, matching this project's "no live training" rule.

Run from the project root (needs torch + flwr, so on Colab -- see
notebooks/phase6_colab.ipynb):
    venv\\Scripts\\python.exe experiments\\save_personalized_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import torch

from src.data.load_dataset import (
    save_preprocessing_artifact,
)
from src.data.multisite import compute_preprocessing_artifact, create_site_partitions, load_multisite
from src.data.partition import add_local_train_test_split
from src.federated.fedprox_runner import run_fedprox_simulation
from src.federated.personalize import personalize_per_hospital_with_models
from src.models.nn_model import HeartDiseaseNet

MODELS_DIR = PROJECT_ROOT / "experiments" / "results" / "models"

# Clients are the 4 real UCI sites (src/data/multisite.py) -- no synthetic partition.
N_CLIENTS = 4
SPLIT_SEED = 42
MODEL_INIT_SEED = 42

N_ROUNDS = 20
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.01
PROXIMAL_MU = 0.1
FINE_TUNE_EPOCHS = 10

HIDDEN_SIZES = (16, 8)  # must match src/models/nn_model.py's HeartDiseaseNet default


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_multisite()
    n_features = x.shape[1]

    artifact = compute_preprocessing_artifact(df_clean, x)
    artifact_path = save_preprocessing_artifact(artifact)
    print(f"Saved preprocessing artifact to {artifact_path}")

    partitions = create_site_partitions(x, y, df_clean)
    partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)

    print(f"\nRunning FedProx (mu={PROXIMAL_MU}, {N_ROUNDS} rounds) then personalizing each hospital "
          f"({FINE_TUNE_EPOCHS} fine-tune epochs) -- same as experiments/phase5_fedprox.py...\n")

    _, final_global_ndarrays = run_fedprox_simulation(
        partitions, n_features, model_init_seed=MODEL_INIT_SEED, proximal_mu=PROXIMAL_MU,
        n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE,
    )

    metrics_df, models = personalize_per_hospital_with_models(
        final_global_ndarrays, partitions, n_features,
        fine_tune_epochs=FINE_TUNE_EPOCHS, lr=LEARNING_RATE,
    )
    print("[Personalized FL] per-hospital results (should match phase5_fedprox.py exactly):")
    print(metrics_df.to_string(index=False))

    manifest = {
        "model_class": "HeartDiseaseNet",
        "n_features": n_features,
        "hidden_sizes": list(HIDDEN_SIZES),
        "feature_order": artifact["feature_order"],
        "hospitals": {},
        "source": {
            "partition": "uci_4_real_sites_option_a", "split_seed": SPLIT_SEED,
            "model_init_seed": MODEL_INIT_SEED, "proximal_mu": PROXIMAL_MU,
            "n_rounds": N_ROUNDS, "local_epochs": LOCAL_EPOCHS,
            "fine_tune_epochs": FINE_TUNE_EPOCHS, "lr": LEARNING_RATE,
        },
    }

    for hospital_name, model in models.items():
        checkpoint_path = MODELS_DIR / f"{hospital_name}_personalized.pt"
        torch.save(model.state_dict(), checkpoint_path)
        row = metrics_df[metrics_df["hospital"] == hospital_name].iloc[0]
        manifest["hospitals"][hospital_name] = {
            "checkpoint": checkpoint_path.name,
            "test_accuracy": float(row["accuracy"]),
            "n_train": int(row["n_train"]),
        }
        print(f"Saved {hospital_name} -> {checkpoint_path}")

    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest to {manifest_path}")
    print(f"\nDone. {len(models)} model checkpoints + preprocessing artifact + manifest "
          f"saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
