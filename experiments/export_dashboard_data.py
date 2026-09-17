"""
Phase 7 step 1: consolidates every phase's result CSVs (Phases 2-6) into a
small set of clean, dashboard-ready JSON files for the FastAPI backend to
serve directly (precomputed -- the backend never recomputes these at
request time, only the live single-patient prediction endpoint does real
work).

Reads only from experiments/results/*.csv and experiments/results/models/
(manifest.json, preprocessing.json) -- no torch/sklearn/flwr needed, so
this runs fine on the local machine despite the Smart App Control block.

Run from the project root:
    venv\\Scripts\\python.exe experiments\\export_dashboard_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MODELS_DIR = RESULTS_DIR / "models"
OUTPUT_DIR = RESULTS_DIR / "dashboard_data"


def _records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records")


def build_hospitals() -> dict:
    """Per-hospital demographics (Phase 2) + local train/test sizes + the
    personalized model's test accuracy (Phase 5/7) -- what each hospital
    "card" on the dashboard needs."""
    demographics = pd.read_csv(RESULTS_DIR / "phase2_hospital_partition_summary.csv")
    local_sizes = pd.read_csv(RESULTS_DIR / "phase3_local_ml_random_forest.csv")[["hospital", "n_train", "n_test"]]
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())

    df = demographics.merge(local_sizes, on="hospital")
    hospitals = []
    for row in df.to_dict(orient="records"):
        h = row["hospital"]
        hospitals.append({
            **row,
            "personalized_accuracy": manifest["hospitals"][h]["test_accuracy"],
            "checkpoint": manifest["hospitals"][h]["checkpoint"],
        })
    return {"hospitals": hospitals, "n_hospitals": len(hospitals)}


def build_experiment_comparison() -> dict:
    """
    The core apples-to-apples NN comparison (Local / FedAvg / FedProx /
    Personalized / Centralized -- all HeartDiseaseNet) plus each
    experiment's per-hospital breakdown, for the dashboard's main
    comparison chart.
    """
    nn_summary = pd.read_csv(RESULTS_DIR / "phase4_nn_comparison_summary.csv")
    label_map = {
        "Local NN (per-hospital, isolated)": "local",
        "FedAvg NN (collaborative, no personalization)": "fedavg",
        "Centralized NN (pooled, not privacy-preserving)": "centralized",
    }
    global_accuracy = {label_map[row["experiment"]]: row["accuracy"] for _, row in nn_summary.iterrows()}

    fedprox_final = pd.read_csv(RESULTS_DIR / "phase5_fedprox_final_per_hospital.csv")
    global_accuracy["fedprox"] = float((fedprox_final["accuracy"] * fedprox_final["n_test"]).sum() / fedprox_final["n_test"].sum())

    personalized = pd.read_csv(RESULTS_DIR / "phase5_personalized_per_hospital.csv")
    global_accuracy["personalized"] = float((personalized["accuracy"] * personalized["n_test"]).sum() / personalized["n_test"].sum())

    per_hospital = {
        "local": _records(RESULTS_DIR / "phase4_local_nn.csv"),
        "fedavg": _records(RESULTS_DIR / "phase4_fedavg_final_per_hospital.csv"),
        "fedprox": _records(RESULTS_DIR / "phase5_fedprox_final_per_hospital.csv"),
        "personalized": _records(RESULTS_DIR / "phase5_personalized_per_hospital.csv"),
        "centralized": _records(RESULTS_DIR / "phase4_centralized_nn_per_hospital.csv"),
    }

    return {
        "global_accuracy": {k: round(v, 4) for k, v in global_accuracy.items()},
        "per_hospital": per_hospital,
        "note": (
            "All five settings use the identical HeartDiseaseNet architecture "
            "(apples-to-apples). See README.md 'Phase 4' section for why the "
            "Phase 3 sklearn RF/LR baselines are NOT included here -- they use "
            "a different model and are not directly comparable."
        ),
    }


def build_training_curves() -> dict:
    """Round-by-round accuracy for FedAvg and FedProx (Phase 4/5), for the
    dashboard's live-looking training-progress chart."""
    return {
        "fedavg": {
            "global_by_round": _records(RESULTS_DIR / "phase4_fedavg_round_summary.csv"),
            "per_hospital_by_round": _records(RESULTS_DIR / "phase4_fedavg_per_round_per_hospital.csv"),
        },
        "fedprox": {
            "global_by_round": _records(RESULTS_DIR / "phase5_fedprox_round_summary.csv"),
            "per_hospital_by_round": _records(RESULTS_DIR / "phase5_fedprox_per_round_per_hospital.csv"),
        },
    }


def build_equity_analysis() -> dict:
    """Phase 5's headline finding: does personalization help the
    worst-served hospital? Plus the multi-seed robustness summaries."""
    return {
        "per_seed": _records(RESULTS_DIR / "phase5_equity_analysis.csv"),
        "phase4_multiseed_summary": _records(RESULTS_DIR / "phase4_multiseed_summary.csv"),
        "phase5_multiseed_summary": _records(RESULTS_DIR / "phase5_multiseed_summary.csv"),
        "headline": (
            "Personalization improved the worst-performing hospital in 4/5 seeds, "
            "by an average of +11.67 percentage points among the seeds where it "
            "improved (+9.34 pp across all 5, including one flat case)."
        ),
    }


def build_explainability() -> dict:
    """Phase 6 SHAP results: global RF feature importance, hospital_1's
    personalized NN importance, and one example patient explanation."""
    return {
        "rf_global_importance": _records(RESULTS_DIR / "phase6_shap_rf_global_importance.csv"),
        "nn_hospital_1_importance": _records(RESULTS_DIR / "phase6_shap_nn_hospital_1_importance.csv"),
        "example_patient": _records(RESULTS_DIR / "phase6_shap_rf_example_patient.csv"),
    }


def build_dashboard_summary(hospitals: dict, comparison: dict) -> dict:
    """Top-level numbers for the dashboard's home/overview screen."""
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    return {
        "dataset": {
            "name": "UCI Heart Disease (Cleveland processed)",
            "n_patients": sum(h["n_patients"] for h in hospitals["hospitals"]),
            "n_features": manifest["n_features"],
        },
        "n_hospitals": hospitals["n_hospitals"],
        "model_architecture": {
            "class": manifest["model_class"],
            "hidden_sizes": manifest["hidden_sizes"],
        },
        "training": {
            "n_rounds": manifest["source"]["n_rounds"],
            "local_epochs": manifest["source"]["local_epochs"],
            "fine_tune_epochs": manifest["source"]["fine_tune_epochs"],
            "proximal_mu": manifest["source"]["proximal_mu"],
        },
        "global_accuracy": comparison["global_accuracy"],
        "privacy_statement": (
            "Only model weight updates are shared during training -- no raw "
            "patient data ever left any simulated hospital. This is a research "
            "prototype using non-IID partitions of one public dataset, not a "
            "real multi-hospital deployment."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hospitals = build_hospitals()
    comparison = build_experiment_comparison()
    outputs = {
        "hospitals.json": hospitals,
        "experiment_comparison.json": comparison,
        "training_curves.json": build_training_curves(),
        "equity_analysis.json": build_equity_analysis(),
        "explainability.json": build_explainability(),
        "dashboard_summary.json": build_dashboard_summary(hospitals, comparison),
    }

    for filename, data in outputs.items():
        path = OUTPUT_DIR / filename
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {path} ({path.stat().st_size} bytes)")

    print(f"\nDone. {len(outputs)} JSON files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
