"""
Static dashboard data endpoints -- all served from the precomputed JSON
files in experiments/results/dashboard_data/ (see
experiments/export_dashboard_data.py). No computation happens here beyond
loading and returning JSON; no torch/sklearn/shap dependency, so this
router works regardless of what ML libraries are installed.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

DASHBOARD_DATA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results" / "dashboard_data"

router = APIRouter(prefix="/api", tags=["dashboard-data"])

_FILES = {
    "summary": "dashboard_summary.json",
    "hospitals": "hospitals.json",
    "experiments/comparison": "experiment_comparison.json",
    "experiments/training-curves": "training_curves.json",
    "equity": "equity_analysis.json",
    "explainability": "explainability.json",
    "explainability/samples": "sample_patients.json",
    "models/weights": "model_weights.json",
}


def _load(filename: str) -> dict:
    path = DASHBOARD_DATA_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{filename} not found in {DASHBOARD_DATA_DIR}. "
                "Run `python experiments/export_dashboard_data.py` first."
            ),
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/dashboard/summary")
def get_dashboard_summary() -> dict:
    return _load(_FILES["summary"])


@router.get("/hospitals")
def get_hospitals() -> dict:
    return _load(_FILES["hospitals"])


@router.get("/experiments/comparison")
def get_experiment_comparison() -> dict:
    return _load(_FILES["experiments/comparison"])


@router.get("/experiments/training-curves")
def get_training_curves() -> dict:
    return _load(_FILES["experiments/training-curves"])


@router.get("/equity")
def get_equity_analysis() -> dict:
    return _load(_FILES["equity"])


@router.get("/explainability")
def get_explainability() -> dict:
    return _load(_FILES["explainability"])


@router.get("/models/weights")
def get_model_weights() -> dict:
    # The 5 personalized models' weights + preprocessing, for in-browser
    # inference (experiments/export_model_weights.py). Aggregate hospital
    # means only; no individual patient records.
    return _load(_FILES["models/weights"])


@router.get("/explainability/samples")
def get_sample_patients() -> dict:
    # Needs torch/shap to generate, so it comes from a separate Colab-run
    # script rather than export_dashboard_data.py. Not having run it yet is
    # an expected state (e.g. a fresh local checkout), so it's an empty list
    # rather than an error.
    if not (DASHBOARD_DATA_DIR / _FILES["explainability/samples"]).exists():
        return {
            "samples": [],
            "generated": False,
            "detail": "Run experiments/export_sample_patients.py on Colab to generate sample patients.",
        }
    return {"generated": True, **_load(_FILES["explainability/samples"])}
