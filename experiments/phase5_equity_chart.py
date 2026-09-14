"""
Renders the equity chart from experiments/results/phase5_equity_analysis.csv
(produced by experiments/phase5_equity_analysis.py) -- no simulation rerun,
just plotting the already-saved per-seed results.

Run from the project root:
    venv\\Scripts\\python.exe experiments\\phase5_equity_chart.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "phase5_equity_analysis.csv")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_pos = np.arange(len(df))
    width = 0.35

    ax.bar(x_pos - width / 2, df["fedavg_accuracy"], width, label="FedAvg (worst-served hospital)", color="#4C72B0")
    ax.bar(x_pos + width / 2, df["personalized_accuracy"], width, label="Personalized (same hospital)", color="#CCB974")

    labels = [f"seed={s}\n({h})" for s, h in zip(df["seed"], df["worst_hospital"])]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 5 equity check: does personalization help the\nworst-served hospital under FedAvg? (4/5 seeds: yes)")
    ax.legend()

    for i, row in df.iterrows():
        color = "green" if row["improved"] else "gray"
        ax.text(i, max(row["fedavg_accuracy"], row["personalized_accuracy"]) + 0.03,
                f"{row['delta']*100:+.1f}pp", ha="center", color=color, fontweight="bold")

    fig.tight_layout()
    fig_path = RESULTS_DIR / "phase5_equity_analysis.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    main()
