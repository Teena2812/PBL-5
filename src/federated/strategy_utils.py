"""
Metric aggregation + per-round/per-hospital recording for the Flower FedAvg
strategy. Flower's evaluate_metrics_aggregation_fn only returns ONE
aggregated dict per round; we need the raw per-hospital breakdown too (the
project's "per-hospital accuracy breakdown" metric), so MetricsRecorder
stashes every client's raw metrics as a side effect while still returning
the weighted-average dict Flower expects.
"""

from __future__ import annotations

import pandas as pd


def _weighted_average(results: list[tuple[int, dict]], key: str) -> float:
    total_examples = sum(n for n, _ in results)
    if total_examples == 0:
        return float("nan")
    return sum(n * m[key] for n, m in results) / total_examples


class MetricsRecorder:
    """
    Records every client's per-round fit/evaluate metrics for later
    per-hospital reporting, while also acting as Flower's
    fit_metrics_aggregation_fn / evaluate_metrics_aggregation_fn.
    """

    def __init__(self):
        self.eval_records: list[dict] = []   # one row per (round, hospital)
        self._eval_round = 0
        self._fit_round = 0

    def record_fit(self, results: list[tuple[int, dict]]) -> dict:
        self._fit_round += 1
        return {}  # nothing to aggregate for fit; hospital name only

    def record_evaluate(self, results: list[tuple[int, dict]]) -> dict:
        self._eval_round += 1
        for n_examples, metrics in results:
            auc = metrics["auc"]
            self.eval_records.append({
                "round": self._eval_round,
                "hospital": metrics["hospital"],
                "n_test": n_examples,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                # -1.0 sentinel (see client.py) means AUC was undefined
                # (single-class test set) -> restore as NaN for reporting.
                "auc": float("nan") if auc == -1.0 else auc,
            })

        return {
            "accuracy": _weighted_average(results, "accuracy"),
            "precision": _weighted_average(results, "precision"),
            "recall": _weighted_average(results, "recall"),
            "f1": _weighted_average(results, "f1"),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.eval_records)
