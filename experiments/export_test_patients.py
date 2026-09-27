"""
Phase 7 (real data): exports every held-out local-test patient (all 4 sites,
seed=42 local split -- the same split every experiment evaluates on) with
the real personalized models' predictions, for the dashboard's Model
Divergence Explorer and the in-browser models' self-check.

For each test patient: site, true label, raw clinical values (recovered by
inverting the encoding, round-trip checked), and the disease probability
from EACH hospital's saved personalized model (torch, same load path as
src/backend/inference.py). No training -- inference only.

Needs torch + sklearn (the local split), so run it on Linux/Colab:
    python experiments/export_test_patients.py
Writes experiments/results/dashboard_data/test_patients.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiments.export_sample_patients import raw_from_encoded  # noqa: E402
from src.backend import inference  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "results" / "dashboard_data" / "test_patients.json"


def main() -> None:
    inference._ensure_torch()
    artifact = inference.get_artifact()
    manifest = inference.get_manifest()
    partitions, _ = inference.get_partitions()
    hospital_ids = sorted(manifest["hospitals"])
    models = {h: inference.load_model(h) for h in hospital_ids}

    patients = []
    for site in sorted(partitions):
        p = partitions[site]
        for row in range(len(p.x_test)):
            encoded = p.x_test.iloc[row]
            raw = raw_from_encoded(encoded, artifact)
            re_encoded = inference.transform_new_patient(raw, artifact).iloc[0]
            diff = float((re_encoded - encoded[re_encoded.index]).abs().max())
            if diff > 1e-6:
                raise ValueError(f"{site} test row {row}: round-trip mismatch {diff}")
            x = torch.tensor(re_encoded[artifact["feature_order"]].to_numpy(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = {h: round(float(torch.sigmoid(models[h](x)).item()), 6) for h in hospital_ids}
            patients.append({
                "id": f"{site}_test_{row}",
                "site": site,
                "test_row": row,
                "actual_label": int(p.y_test.iloc[row]),
                "patient": raw,
                "probabilities": probs,
            })

    counts = {s: sum(1 for x in patients if x["site"] == s) for s in sorted(partitions)}
    OUT_PATH.write_text(json.dumps({
        "patients": patients,
        "n_patients": len(patients),
        "per_site": counts,
        "note": (
            "Every held-out local-test patient of the 4 UCI sites (seed=42 local split), "
            "with each hospital's saved personalized model's disease probability (torch)."
        ),
    }, indent=1), encoding="utf-8")
    print(f"Wrote {len(patients)} test patients {counts} to {OUT_PATH}")


if __name__ == "__main__":
    main()
