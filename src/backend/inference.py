"""
Live single-patient inference: load a hospital's saved personalized model
(no retraining, ever), run one forward pass, and compute a single-instance
SHAP explanation.

torch/shap are imported defensively (try/except at module load) so the
REST of the backend (dashboard data endpoints) still works in an
environment without them -- e.g. this project's local dev machine, where
Smart App Control blocks torch/shap's compiled extensions. Only calling
predict_and_explain() (i.e. the POST /api/predict endpoint) actually needs
them; every other endpoint is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import torch

    from src.explainability.shap_utils import compute_kernel_shap_nn
    from src.models.nn_model import HeartDiseaseNet

    TORCH_AVAILABLE = True
    _import_error: Exception | None = None
except ImportError as e:  # pragma: no cover - exercised only where torch is missing
    TORCH_AVAILABLE = False
    _import_error = e

from src.data.load_dataset import load_preprocessing_artifact, transform_new_patient
from src.data.multisite import create_site_partitions, load_multisite
from src.data.partition import add_local_train_test_split

MODELS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results" / "models"

# Must match every other phase's protocol (Phase 2-5) -- this is
# deterministic data loading/partitioning, NOT model training, so
# recomputing it at backend startup does not violate the "no live
# training" rule.
# Clients are the 4 real UCI sites (src/data/multisite.py) -- no synthetic partition.
N_CLIENTS = 4
SPLIT_SEED = 42

_cache: dict = {}


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "torch/shap are not available in this environment "
            f"({_import_error}). The live prediction endpoint needs them -- "
            "run the backend somewhere they're installed (e.g. Colab), not "
            "on a machine where they're blocked."
        )


def get_partitions() -> tuple[dict, int]:
    """Cached {hospital_name: HospitalPartition}, plus n_features. Needed
    only to supply each hospital's own x_train as the SHAP background
    sample -- no model training happens here."""
    if "partitions" not in _cache:
        x, y, df_clean = load_multisite()
        partitions = create_site_partitions(x, y, df_clean)
        partitions = add_local_train_test_split(partitions, test_size=0.25, seed=SPLIT_SEED)
        _cache["partitions"] = {p.name: p for p in partitions}
        _cache["n_features"] = x.shape[1]
    return _cache["partitions"], _cache["n_features"]


def get_artifact() -> dict:
    if "artifact" not in _cache:
        _cache["artifact"] = load_preprocessing_artifact()
    return _cache["artifact"]


def get_manifest() -> dict:
    if "manifest" not in _cache:
        _cache["manifest"] = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return _cache["manifest"]


def load_model(hospital_id: str):
    _ensure_torch()
    manifest = get_manifest()
    if hospital_id not in manifest["hospitals"]:
        raise ValueError(f"Unknown hospital_id {hospital_id!r}; valid: {sorted(manifest['hospitals'])}")

    models = _cache.setdefault("models", {})
    if hospital_id not in models:
        model = HeartDiseaseNet(manifest["n_features"], hidden_sizes=tuple(manifest["hidden_sizes"]))
        checkpoint_path = MODELS_DIR / manifest["hospitals"][hospital_id]["checkpoint"]
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        models[hospital_id] = model
    return models[hospital_id]


def _risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "HIGH"
    if probability >= 0.33:
        return "MODERATE"
    return "LOW"


def predict_and_explain(raw_patient: dict, hospital_id: str, top_n_features: int = 8) -> dict:
    """
    Inference + single-instance SHAP explanation only -- no gradient
    updates, no retraining. raw_patient must have the original UCI feature
    codes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
    oldpeak, slope, ca, thal).
    """
    _ensure_torch()

    artifact = get_artifact()
    manifest = get_manifest()
    model = load_model(hospital_id)

    x_row = transform_new_patient(raw_patient, artifact)  # 1-row DataFrame, raises ValueError on bad category

    with torch.no_grad():
        logit = model(torch.tensor(x_row.to_numpy(), dtype=torch.float32))
        probability = float(torch.sigmoid(logit).item())
    label = int(probability >= 0.5)

    partitions, _ = get_partitions()
    background = partitions[hospital_id].x_train
    explainer, shap_values = compute_kernel_shap_nn(
        model, background, x_row, n_background=min(20, len(background)), nsamples=100, seed=42
    )

    base_value = explainer.expected_value
    base_value = float(base_value[0]) if hasattr(base_value, "__len__") else float(base_value)

    row_shap = shap_values[0]
    contributions = sorted(
        [
            {"feature": f, "feature_value": float(x_row.iloc[0][f]), "shap_value": float(v)}
            for f, v in zip(x_row.columns, row_shap)
        ],
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )[:top_n_features]

    return {
        "hospital_id": hospital_id,
        "predicted_probability": round(probability, 4),
        "predicted_label": label,
        "risk_level": _risk_level(probability),
        "model_test_accuracy": manifest["hospitals"][hospital_id]["test_accuracy"],
        "base_value": round(base_value, 4),
        "explanation": contributions,
    }
