"""
Phase 7 step 1 (updated for the 4 real sites): consolidates every phase's result CSVs (Phases 2-6) into a
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


PER_HOSPITAL_RESULT_FILES = {
    "local": "phase4_local_nn.csv",
    "fedavg": "phase4_fedavg_final_per_hospital.csv",
    "fedprox": "phase5_fedprox_final_per_hospital.csv",
    "personalized": "phase5_personalized_per_hospital.csv",
    "centralized": "phase4_centralized_nn_per_hospital.csv",
}


def derive_test_class_counts() -> dict[str, dict]:
    """
    Each site's test-set class counts, recovered from the saved metrics
    (accuracy, precision, recall, n_test) -- the CSVs don't store them.
    With P positives among n: TP = recall*P, FP = TP*(1/precision - 1),
    TN = n - P - FP and accuracy*n = TP + TN, which gives
    P = n*(accuracy - 1) / (2*recall - 1 - recall/precision).
    Every model's results must agree on the same integer P per site, or
    this fails loudly.
    """
    found: dict[str, set[int]] = {}
    for filename in PER_HOSPITAL_RESULT_FILES.values():
        for r in _records(RESULTS_DIR / filename):
            n, acc, prec, rec = r["n_test"], r["accuracy"], r["precision"], r["recall"]
            if prec <= 0 or rec <= 0:
                continue
            denom = 2 * rec - 1 - rec / prec
            if abs(denom) < 1e-9:
                continue
            p_est = n * (acc - 1) / denom
            if abs(p_est - round(p_est)) > 0.05:
                raise ValueError(f"{filename} {r['hospital']}: non-integer positive count {p_est:.3f}")
            found.setdefault(r["hospital"], set()).add(int(round(p_est)))
    sizes = {r["hospital"]: r["n_test"] for r in _records(RESULTS_DIR / "phase3_local_ml_random_forest.csv")}
    counts = {}
    for site, values in found.items():
        if len(values) != 1:
            raise ValueError(f"{site}: result files disagree on the test positive count: {values}")
        pos = values.pop()
        counts[site] = {"n_test": sizes[site], "test_disease": pos, "test_no_disease": sizes[site] - pos}
    return counts


def confusion_extras(row: dict, test_counts: dict) -> dict:
    """Specificity and balanced accuracy from the recovered class counts."""
    c = test_counts[row["hospital"]]
    # Counts are whole patients; the CSVs' 4-decimal rounding leaves tiny residue.
    tp = round(row["recall"] * c["test_disease"])
    tn = round(row["accuracy"] * c["n_test"]) - tp
    specificity = tn / c["test_no_disease"] if c["test_no_disease"] else float("nan")
    return {
        "specificity": round(specificity, 4),
        "balanced_accuracy": round((row["recall"] + specificity) / 2, 4),
    }


def build_hospitals(test_counts: dict) -> dict:
    """The 4 real sites: size and case mix (Phase 2), local train/test sizes,
    test-set class balance, the accuracy of always predicting the majority
    class there, and the personalized model's results."""
    sites = pd.read_csv(RESULTS_DIR / "phase2_site_summary.csv")
    local_sizes = pd.read_csv(RESULTS_DIR / "phase3_local_ml_random_forest.csv")[["hospital", "n_train", "n_test"]]
    personalized = {r["hospital"]: r for r in _records(RESULTS_DIR / "phase5_personalized_per_hospital.csv")}
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())

    df = sites.merge(local_sizes, on="hospital")
    hospitals = []
    for row in df.to_dict(orient="records"):
        h = row["hospital"]
        c = test_counts[h]
        pers = personalized[h]
        hospitals.append({
            **row,
            "test_disease": c["test_disease"],
            "test_no_disease": c["test_no_disease"],
            "majority_class_accuracy": round(max(c["test_disease"], c["test_no_disease"]) / c["n_test"], 4),
            "personalized_accuracy": manifest["hospitals"][h]["test_accuracy"],
            "personalized_recall": pers["recall"],
            "personalized_auc": pers["auc"],
            **{f"personalized_{k}": v for k, v in confusion_extras(pers, test_counts).items()},
            "checkpoint": manifest["hospitals"][h]["checkpoint"],
        })
    return {"hospitals": hospitals, "n_hospitals": len(hospitals)}


def build_experiment_comparison(test_counts: dict) -> dict:
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
        key: [{**r, **confusion_extras(r, test_counts)} for r in _records(RESULTS_DIR / filename)]
        for key, filename in PER_HOSPITAL_RESULT_FILES.items()
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
    """
    The project's headline (locked wording, 2026-09-27), computed from the
    5-seed Phase 4 runs so every number in it traces to a CSV, plus the
    per-seed worst-served-hospital analysis (personalization's mixed result)
    and the multi-seed summaries.
    """
    seeds = pd.read_csv(RESULTS_DIR / "phase4_multiseed_results.csv")
    n = len(seeds)
    beats_local = int((seeds["fedavg_nn_accuracy"] > seeds["local_nn_accuracy"]).sum())
    ge_central = int((seeds["fedavg_nn_accuracy"] >= seeds["centralized_nn_accuracy"]).sum())
    fedavg_mean = float(seeds["fedavg_nn_accuracy"].mean())
    local_mean = float(seeds["local_nn_accuracy"].mean())
    central_mean = float(seeds["centralized_nn_accuracy"].mean())
    delta_pp = round((fedavg_mean - local_mean) * 100, 1)
    text = (
        f"Federated training beats hospitals training alone in {beats_local}/{n} seeds "
        f"({fedavg_mean * 100:.1f}% vs {local_mean * 100:.1f}% overall accuracy, +{delta_pp}pp average) "
        f"and matches or beats full data pooling in {ge_central}/{n} seeds \u2014 without sharing patient records."
    )

    equity = pd.read_csv(RESULTS_DIR / "phase5_equity_analysis.csv")
    return {
        "headline": {
            "text": text,
            "n_seeds": n,
            "seeds_fedavg_beats_local": beats_local,
            "seeds_fedavg_matches_or_beats_centralized": ge_central,
            "fedavg_mean_accuracy": round(fedavg_mean, 4),
            "local_mean_accuracy": round(local_mean, 4),
            "centralized_mean_accuracy": round(central_mean, 4),
            "delta_pp": delta_pp,
            "per_seed": seeds.to_dict(orient="records"),
        },
        "worst_served": {
            "per_seed": equity.to_dict(orient="records"),
            "seeds_improved": int((equity["delta"] > 0).sum()),
            "seeds_flat": int((equity["delta"] == 0).sum()),
            "seeds_worse": int((equity["delta"] < 0).sum()),
            "mean_delta_pp": round(float(equity["delta"].mean()) * 100, 2),
        },
        "phase4_multiseed_summary": _records(RESULTS_DIR / "phase4_multiseed_summary.csv"),
        "phase5_multiseed_summary": _records(RESULTS_DIR / "phase5_multiseed_summary.csv"),
    }


def build_explainability() -> dict:
    """Phase 6 SHAP: pooled Random Forest global importance, the
    personalized NN importance of the hospital Phase 6 analysed (the most
    often worst-served one, chosen by that script), and one example patient."""
    nn_files = sorted(RESULTS_DIR.glob("phase6_shap_nn_*_importance.csv"))
    if len(nn_files) != 1:
        raise ValueError(f"expected exactly one phase6_shap_nn_*_importance.csv, found {[f.name for f in nn_files]}")
    target = nn_files[0].name[len("phase6_shap_nn_"):-len("_importance.csv")]
    return {
        "rf_global_importance": _records(RESULTS_DIR / "phase6_shap_rf_global_importance.csv"),
        "nn_target_hospital": target,
        "nn_target_importance": _records(nn_files[0]),
        "example_patient": _records(RESULTS_DIR / "phase6_shap_rf_example_patient.csv"),
    }


def build_dashboard_summary(hospitals: dict, comparison: dict) -> dict:
    """Top-level numbers for the dashboard's home/overview screen."""
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    return {
        "dataset": {
            "name": "UCI Heart Disease, 4 sites",
            "sites": "Cleveland, Hungary, Switzerland, VA Long Beach",
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
            "Only model weight updates are shared during training -- no patient "
            "record leaves its hospital. This is a research simulation on the "
            "public UCI Heart Disease data from four real institutions "
            "(collected in 1988); those hospitals did not actually run "
            "federated training."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_counts = derive_test_class_counts()
    hospitals = build_hospitals(test_counts)
    comparison = build_experiment_comparison(test_counts)
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
