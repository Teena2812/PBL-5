"""
Phase 2 seed selection: searches random seeds for the 5-hospital, alpha=0.5
Dirichlet partition to find one that is both size-balanced AND gives every
hospital a usable number of samples of both classes.

Label-skew Dirichlet partitioning controls class *proportions* per hospital,
not hospital *size* or *minimum class count* -- so a given seed can (a) let
one hospital hold 55%+ of all patients (dominating FedAvg's sample-weighted
aggregation) and/or (b) leave some hospital with zero examples of one class.

(b) turned out to matter more than originally realized: a hospital with a
single-class training set cannot fit Logistic Regression at all and forces
Random Forest into a degenerate majority-class predictor (accuracy looks
fine, but F1=0 and AUC=NaN -- see the Phase 3 README caveat). That's a
Phase-3-only annoyance for a *baseline* metric, but for Phase 4 (FedAvg) and
Phase 5 (Personalized FL) it's worse: a single-class client can't meaningfully
contribute a local gradient update for the minority class at all, which would
distort federated aggregation, not just one baseline's reported accuracy.

So this search now applies two constraints together:
    1. max hospital share of total patients kept low (balanced size, as before)
    2. every hospital has >= MIN_CLASS_COUNT samples of BOTH classes

Result: seed=117 was selected (max_share=0.276, min_class_count=8) and is
now the default SEED in experiments/phase2_partition_report.py and
experiments/phase3_baselines.py. The earlier seed=8 (chosen for size balance
alone in the first pass of this search) is superseded -- it left hospital_2
with disease_rate=0.0 (zero diseased patients at all), which is exactly the
single-class problem this constraint rules out.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_dataset import load_and_preprocess
from src.data.partition import create_hospital_partitions

N_CLIENTS = 5
ALPHA = 0.5
SEED_RANGE = range(1, 501)
MAX_SHARE_THRESHOLD = 0.55   # avoid any hospital holding >55% of patients
MIN_CLASS_COUNT = 5          # every hospital must have >= this many of EACH class


def class_counts(partition) -> tuple[int, int]:
    vc = partition.y.value_counts()
    return int(vc.get(0, 0)), int(vc.get(1, 0))


def main() -> None:
    x, y, df_clean = load_and_preprocess()

    all_results = []
    valid_results = []
    for seed in SEED_RANGE:
        parts = create_hospital_partitions(x, y, df_clean, n_clients=N_CLIENTS, alpha=ALPHA, seed=seed)
        sizes = [len(p) for p in parts]
        total = sum(sizes)
        max_share = max(sizes) / total
        counts = [class_counts(p) for p in parts]
        min_class_count = min(min(c0, c1) for c0, c1 in counts)

        result = (seed, max_share, min_class_count, sizes, counts)
        all_results.append(result)
        if min_class_count >= MIN_CLASS_COUNT:
            valid_results.append(result)

    print(f"Searched seeds 1-{SEED_RANGE.stop - 1} at alpha={ALPHA}, n={N_CLIENTS}.")
    print(f"{len(valid_results)}/{len(all_results)} seeds give every hospital "
          f">= {MIN_CLASS_COUNT} samples of both classes.")

    # Among valid seeds, prefer the most size-balanced (lowest max_share),
    # then prefer a larger min_class_count as a tiebreak (safety margin).
    valid_results.sort(key=lambda r: (r[1], -r[2]))

    print(f"\nTop 10 valid seeds (size-balanced first, then class-count margin):")
    for seed, max_share, min_class_count, sizes, counts in valid_results[:10]:
        print(f"seed={seed:3d}  max_share={max_share:.3f}  min_class_count={min_class_count:2d}  sizes={sizes}")

    chosen = next(r for r in valid_results if r[0] == 117)
    seed, max_share, min_class_count, sizes, counts = chosen
    print(f"\nSelected seed=117: max_share={max_share:.3f}, min_class_count={min_class_count}, "
          f"sizes={sizes}, per-hospital (n_class0, n_class1)={counts}")


if __name__ == "__main__":
    main()
