"""
Phase 7: export the 5 saved personalized models to a JSON file the React
dashboard can run in the browser (forward pass in JavaScript), so live
predictions work with no backend and no torch -- including on this
project's local machine, where Smart App Control blocks torch.

Reads the .pt checkpoints WITHOUT torch: a torch.save() file is a zip of a
pickled state_dict whose tensors point at raw little-endian storage blobs
(data/0, data/1, ...). A small custom unpickler rebuilds each tensor from
its storage with numpy. Inference only -- nothing is trained or changed.

Also exports what the browser needs around the models:
  - preprocessing (standardization constants + one-hot categories, from
    experiments/results/models/preprocessing.json) so raw clinical inputs
    are encoded exactly like src/data/load_dataset.py:transform_new_patient
  - each hospital's average encoded patient, over ALL its patients from the
    seed=117 partition (aggregate means only, no individual records), used
    as the baseline for the dashboard's approximate per-feature breakdown
  - the observed min/max of each numeric input across the 297 patients,
    which bounds the dashboard's what-if sliders

Verification: re-runs the forward pass in numpy on the 5 precomputed sample
patients (real predict_and_explain() outputs from Colab) and fails loudly if
any probability differs by more than the 4-dp rounding they were saved at.

Run: venv\\Scripts\\python experiments\\export_model_weights.py
"""

from __future__ import annotations

import json
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_dataset import load_and_preprocess, load_preprocessing_artifact, transform_new_patient  # noqa: E402
from src.data.partition import create_hospital_partitions  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = RESULTS_DIR / "models"
OUT_PATH = RESULTS_DIR / "dashboard_data" / "model_weights.json"
SAMPLES_PATH = RESULTS_DIR / "dashboard_data" / "sample_patients.json"

# Same partition protocol as every other phase (src/backend/inference.py).
N_CLIENTS, ALPHA, PARTITION_SEED = 5, 0.5, 117

_STORAGE_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64}


def _load_state_dict(path: Path) -> "OrderedDict[str, np.ndarray]":
    """torch.load() for a plain float state_dict, using numpy only."""
    with zipfile.ZipFile(path) as zf:
        root = zf.namelist()[0].split("/")[0]
        storages: dict[str, tuple[bytes, type]] = {}

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor
                if module == "torch" and name in _STORAGE_DTYPES:
                    return name
                raise pickle.UnpicklingError(f"unexpected pickle global {module}.{name}")

            def persistent_load(self, pid):
                _, storage_type, key, _location, _numel = pid
                if key not in storages:
                    storages[key] = (zf.read(f"{root}/data/{key}"), _STORAGE_DTYPES[storage_type])
                return key

        def _rebuild_tensor(key, offset, size, stride, *_):
            raw, dtype = storages[key]
            flat = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
            itemsize = np.dtype(dtype).itemsize
            view = np.lib.stride_tricks.as_strided(
                flat[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
            )
            return np.array(view, dtype=np.float64)

        byteorder = zf.read(f"{root}/byteorder").decode().strip() if f"{root}/byteorder" in zf.namelist() else "little"
        if byteorder != "little":
            raise ValueError(f"{path.name}: unsupported byteorder {byteorder}")
        return _Unpickler(zf.open(f"{root}/data.pkl")).load()


def _layers(state: dict) -> list[dict]:
    """HeartDiseaseNet = Sequential(Linear, ReLU, Linear, ReLU, Linear):
    Linear layers are net.0, net.2, net.4."""
    return [{"weight": state[f"net.{i}.weight"], "bias": state[f"net.{i}.bias"]} for i in (0, 2, 4)]


def forward_prob(layers: list[dict], x: np.ndarray) -> float:
    h = x
    for i, layer in enumerate(layers):
        h = layer["weight"] @ h + layer["bias"]
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return float(1.0 / (1.0 + np.exp(-h[0])))


def main() -> None:
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    artifact = load_preprocessing_artifact()
    feature_order = artifact["feature_order"]
    if feature_order != manifest["feature_order"]:
        raise ValueError("preprocessing.json and manifest.json disagree on feature order")

    models = {}
    for hospital_id, info in sorted(manifest["hospitals"].items()):
        layers = _layers(_load_state_dict(MODELS_DIR / info["checkpoint"]))
        shapes = [layer["weight"].shape for layer in layers]
        expected = [(16, 22), (8, 16), (1, 8)]
        if shapes != expected:
            raise ValueError(f"{hospital_id}: layer shapes {shapes}, expected {expected}")
        models[hospital_id] = layers

    n_params = sum(l["weight"].size + l["bias"].size for l in next(iter(models.values())))

    # Verify against real predict_and_explain() outputs before writing anything.
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]
    print("Verification against sample_patients.json (real backend outputs):")
    for s in samples:
        x = transform_new_patient(s["patient"], artifact).iloc[0][feature_order].to_numpy(dtype=np.float64)
        p = forward_prob(models[s["hospital_id"]], x)
        diff = abs(p - s["predicted_probability"])
        print(f"  {s['id']:<20} backend={s['predicted_probability']:.4f}  numpy={p:.6f}  |diff|={diff:.2e}")
        if diff > 5e-5:  # saved at 4 dp, so max rounding error is 5e-5
            raise ValueError(f"{s['id']}: exported model disagrees with the real backend output")

    # Each hospital's average encoded patient over its full partition.
    x_all, y_all, df_clean = load_and_preprocess()
    partitions = create_hospital_partitions(
        x_all, y_all, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=PARTITION_SEED
    )
    hospitals_json = json.loads((RESULTS_DIR / "dashboard_data" / "hospitals.json").read_text(encoding="utf-8"))
    expected_sizes = {h["hospital"]: h["n_patients"] for h in hospitals_json["hospitals"]}
    baselines = {}
    for p in partitions:
        if len(p.x) != expected_sizes[p.name]:
            raise ValueError(f"{p.name}: partition has {len(p.x)} patients, hospitals.json says {expected_sizes[p.name]}")
        baselines[p.name] = {"n_patients": len(p.x), "mean_encoded": p.x[feature_order].mean().round(6).tolist()}

    # Raw-unit range of each numeric input across all 297 patients, so the
    # dashboard's sliders stay inside values the models actually saw.
    observed_range = {
        f: [float(df_clean[f].min()), float(df_clean[f].max())] for f in artifact["numeric_features"]
    }

    def as_lists(layers):
        return [{"weight": np.round(l["weight"], 9).tolist(), "bias": np.round(l["bias"], 9).tolist()} for l in layers]

    OUT_PATH.write_text(json.dumps({
        "architecture": {
            "class": manifest["model_class"],
            "layers": "Linear(22,16) -> ReLU -> Linear(16,8) -> ReLU -> Linear(8,1) -> sigmoid",
            "n_parameters": int(n_params),
        },
        "feature_order": feature_order,
        "preprocessing": {
            "numeric_features": artifact["numeric_features"],
            "binary_features": artifact["binary_features"],
            "categorical_features": artifact["categorical_features"],
            "categorical_categories": artifact["categorical_categories"],
            "means": artifact["means"],
            "stds": artifact["stds"],
            "observed_range": observed_range,
            "n_patients_total": int(len(df_clean)),
        },
        "hospitals": {
            h: {
                "test_accuracy": manifest["hospitals"][h]["test_accuracy"],
                "n_train": manifest["hospitals"][h]["n_train"],
                "baseline": baselines[h],
                "layers": as_lists(models[h]),
            }
            for h in models
        },
        "verification": {
            "reference": "sample_patients.json (predict_and_explain() on Colab)",
            "cases": [
                {"id": s["id"], "hospital_id": s["hospital_id"], "patient": s["patient"],
                 "expected_probability": s["predicted_probability"]}
                for s in samples
            ],
        },
    }, indent=1), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(models)} models x {n_params} parameters)")


if __name__ == "__main__":
    main()
