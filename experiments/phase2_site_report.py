"""
Phase 2 (real data): the 4 real UCI Heart Disease sites used directly as
federated clients (src/data/multisite.py) -- replaces the earlier
Dirichlet split of Cleveland alone (experiments/phase2_partition_report.py,
kept for the record).

Writes:
  experiments/results/phase2_site_summary.csv  -- per-site counts & demographics
  experiments/results/phase2_site_missingness.csv -- per-site, per-feature
      missingness in the raw UCI files (why ca/thal/slope/chol/fbs are excluded)
  experiments/results/phase2_site_summary.png

Runs locally (pandas + matplotlib only).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.multisite import (  # noqa: E402
    COLUMN_NAMES,
    SITE_LABELS,
    create_site_partitions,
    download_site_files,
    load_multisite,
    summarize_sites,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def raw_missingness() -> pd.DataFrame:
    """Share of each feature missing per site in the raw files, counting
    physiologically impossible zeros (chol, trestbps) as missing."""
    rows = []
    for site, path in download_site_files().items():
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
        df.loc[df["chol"] == 0, "chol"] = np.nan
        df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
        row = {"hospital": site, "n_raw_rows": len(df)}
        for col in COLUMN_NAMES[:-1]:
            row[col] = round(float(df[col].isna().mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x, y, df_clean = load_multisite()
    partitions = create_site_partitions(x, y, df_clean)

    summary = summarize_sites(partitions, df_clean)
    summary.to_csv(RESULTS_DIR / "phase2_site_summary.csv", index=False)
    missing = raw_missingness()
    missing.to_csv(RESULTS_DIR / "phase2_site_missingness.csv", index=False)

    print(f"Real sites as clients: {len(partitions)} hospitals, {len(df_clean)} patients, "
          f"{x.shape[1]} encoded features\n")
    print(summary.to_string(index=False))
    print("\nRaw missingness (share of rows; chol/trestbps zeros counted as missing):")
    print(missing.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [SITE_LABELS[h] for h in summary["hospital"]]
    axes[0].bar(labels, summary["n_patients"], color="#2563eb")
    axes[0].set_title("Patients per site (after cleaning)")
    axes[1].bar(labels, summary["disease_rate"], color="#dc2626")
    axes[1].axhline(df_clean["target"].mean(), color="black", linestyle="--", label="pooled")
    axes[1].set_title("Disease rate per site")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "phase2_site_summary.png", dpi=150)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
