"""
Phase 2 seed selection: searches random seeds for the 5-hospital, alpha=0.5
Dirichlet partition to find one with balanced hospital sizes.

Label-skew Dirichlet partitioning controls class *proportions* per hospital,
not hospital *size* -- at some seeds, one hospital can end up holding 55%+
of all patients, which would let that hospital dominate FedAvg's
sample-weighted aggregation and make the FedAvg-vs-personalized-FL
comparison less meaningful. This script searches seeds 1-50 and reports the
most size-balanced ones (by lowest max hospital share of total patients).

Result: seed=8 was selected (max_share=0.239, min_share=0.125) and is now
the default SEED in experiments/phase2_partition_report.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_dataset import load_and_preprocess
from src.data.partition import create_hospital_partitions

N_CLIENTS = 5
ALPHA = 0.5
SEED_RANGE = range(1, 51)
MAX_SHARE_THRESHOLD = 0.55  # avoid any hospital holding >55% of patients


def main() -> None:
    x, y, df_clean = load_and_preprocess()

    results = []
    for seed in SEED_RANGE:
        parts = create_hospital_partitions(x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=seed)
        sizes = [len(p) for p in parts]
        total = sum(sizes)
        max_share = max(sizes) / total
        min_share = min(sizes) / total
        results.append((seed, max_share, min_share, sizes))

    results.sort(key=lambda r: r[1])

    print(f"Seeds exceeding max_share > {MAX_SHARE_THRESHOLD} "
          f"(single hospital dominance risk):")
    bad = [r for r in results if r[1] > MAX_SHARE_THRESHOLD]
    print(f"  {len(bad)}/{len(results)} seeds")

    print(f"\nTop 10 most balanced seeds (alpha={ALPHA}, n={N_CLIENTS}):")
    for seed, max_share, min_share, sizes in results[:10]:
        print(f"seed={seed:3d}  max_share={max_share:.3f}  min_share={min_share:.3f}  sizes={sizes}")


if __name__ == "__main__":
    main()
