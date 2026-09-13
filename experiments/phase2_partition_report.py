"""
Phase 2 deliverable: loads the UCI Heart Disease dataset, builds non-IID
hospital partitions (Dirichlet label-skew, alpha=0.5, 5 hospitals), and
saves a summary table + bar chart to experiments/results/ for use in the
research write-up.

Run from the project root:
    venv\\Scripts\\python.exe experiments\\phase2_partition_report.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.data.load_dataset import load_and_preprocess
from src.data.partition import create_hospital_partitions, summarize_partition

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

N_CLIENTS = 5
ALPHA = 0.5
# seed=117 was selected by experiments/phase2_seed_search.py as the most
# size-balanced non-IID split (max hospital share 27.6% of patients) among
# seeds where every hospital has >= 5 samples of BOTH classes (min_class_count=8
# at this seed). An earlier pass (seed=8) optimized only for size balance and
# left hospital_2 with zero diseased patients -- a single-class hospital that
# degenerates FL/baseline training for that client. See phase2_seed_search.py's
# module docstring for the full reasoning.
SEED = 117


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y, df_clean = load_and_preprocess()
    print(f"Total patients after cleaning: {len(df_clean)}")
    print(f"Feature count after encoding: {x.shape[1]}")

    partitions = create_hospital_partitions(
        x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=SEED
    )
    summary = summarize_partition(partitions, df_clean)

    print(f"\nNon-IID hospital partitions (Dirichlet alpha={ALPHA}, n={N_CLIENTS}):")
    print(summary.to_string(index=False))

    csv_path = RESULTS_DIR / "phase2_hospital_partition_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary table to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(summary["hospital"], summary["n_patients"], color="#4C72B0")
    axes[0].set_title("Patients per simulated hospital")
    axes[0].set_ylabel("n_patients")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(summary["hospital"], summary["disease_rate"], color="#C44E52")
    axes[1].axhline(y.mean(), color="black", linestyle="--", linewidth=1,
                     label=f"global rate = {y.mean():.2f}")
    axes[1].set_title(f"Disease rate per hospital (non-IID, alpha={ALPHA})")
    axes[1].set_ylabel("disease_rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()

    fig.suptitle("Phase 2: Simulated Hospital Heterogeneity (UCI Heart Disease)")
    fig.tight_layout()

    fig_path = RESULTS_DIR / "phase2_hospital_heterogeneity.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
