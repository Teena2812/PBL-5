"""
Phase 7: precompute a small set of sample patients for the dashboard's
Explainability screen -- one real held-out test patient per hospital, run
through that hospital's saved personalized model with the SAME
predict_and_explain() the live POST /api/predict endpoint uses (inference +
single-instance KernelSHAP only, no retraining).

Writes experiments/results/dashboard_data/sample_patients.json, so the
dashboard can show real predictions + SHAP explanations even where torch
can't run (this project's local machine, where Smart App Control blocks
torch/shap/sklearn). Needs torch + shap + sklearn, so run it on Colab:

    !python experiments/export_sample_patients.py

Patient choice is deterministic: for hospital i, the first test patient
(in local-test-split order) whose true label alternates disease / no
disease across hospitals, so the samples cover both outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.backend import inference  # noqa: E402

OUT_PATH = (
    Path(__file__).resolve().parent / "results" / "dashboard_data" / "sample_patients.json"
)

# Integer-valued raw fields; oldpeak is the only non-integer clinical input.
INTEGER_FIELDS = ["age", "trestbps", "chol", "thalach", "ca"]


def raw_from_encoded(row: pd.Series, artifact: dict) -> dict:
    """Inverts transform_new_patient(): standardized numerics back to
    clinical units, one-hot columns back to their category code."""
    raw: dict = {}
    for f in artifact["numeric_features"]:
        value = row[f] * artifact["stds"][f] + artifact["means"][f]
        raw[f] = int(round(value)) if f in INTEGER_FIELDS else round(float(value), 1)
    for f in artifact["binary_features"]:
        raw[f] = int(round(row[f]))
    for f in artifact["categorical_features"]:
        hot = [c for c in artifact["categorical_categories"][f] if row[f"{f}_{c}"] == 1]
        if len(hot) != 1:
            raise ValueError(f"row has {len(hot)} active '{f}' columns, expected exactly 1")
        raw[f] = int(hot[0])
    return raw


def main() -> None:
    inference._ensure_torch()
    artifact = inference.get_artifact()
    partitions, _ = inference.get_partitions()

    samples = []
    for i, hospital_id in enumerate(sorted(partitions)):
        p = partitions[hospital_id]
        wanted_label = 1 if i % 2 == 0 else 0
        matches = p.y_test[p.y_test == wanted_label]
        if matches.empty:  # every hospital has both classes (Phase 2 seed choice), but be safe
            matches = p.y_test
        idx = int(matches.index[0])

        raw = raw_from_encoded(p.x_test.iloc[idx], artifact)

        # Round-trip check: re-encoding the recovered raw record must give
        # back the exact model input, or the sample isn't the real patient.
        re_encoded = inference.transform_new_patient(raw, artifact).iloc[0]
        max_diff = float((re_encoded - p.x_test.iloc[idx][re_encoded.index]).abs().max())
        if max_diff > 1e-6:
            raise ValueError(f"{hospital_id} test row {idx}: round-trip mismatch {max_diff:.4f}")

        result = inference.predict_and_explain(raw, hospital_id)
        samples.append({
            "id": f"{hospital_id}_test_{idx}",
            "hospital_id": hospital_id,
            "test_row": idx,
            "actual_label": int(p.y_test.iloc[idx]),
            "patient": raw,
            **result,
        })
        print(
            f"{hospital_id}: test row {idx}, actual={samples[-1]['actual_label']}, "
            f"p={result['predicted_probability']:.3f} ({result['risk_level']})"
        )

    OUT_PATH.write_text(
        json.dumps({
            "samples": samples,
            "note": (
                "One held-out local-test patient per hospital, predicted by that "
                "hospital's saved personalized model via the same "
                "predict_and_explain() as POST /api/predict (KernelSHAP, 20 "
                "background samples from the hospital's own training data, "
                "nsamples=100, seed=42)."
            ),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
