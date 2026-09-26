"""
End-to-end check of the live-training endpoints against a REAL running
server (uvicorn), used by .github/workflows/live-training-check.yml with
the server inside a 512 MB / 0.1 CPU container (Render free equivalent).

Triggers a FedAvg run and a FedProx + personalization run over HTTP,
consumes the SSE stream like a browser, timestamps every event on arrival,
and checks every streamed number against known-correct outputs:
  - the Flower (Ray) experiment runs on the same data/seed
    (preflight_results/phase4_fedavg_per_round_per_hospital.csv,
     phase5_fedprox_per_round_per_hospital.csv, phase5_personalized_per_hospital.csv)
  - a direct in-process run of the Ray-free loop (reference JSON)

Usage: python tests/live_training_e2e.py BASE_URL REFERENCE_JSON FLOWER_DIR OUT_JSON
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import csv
import json
import sys
import time

import httpx

METRICS = ("accuracy", "precision", "recall", "f1")
TOL = 1e-9


def wait_healthy(base, timeout=600):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if httpx.get(f"{base}/api/health", timeout=5).status_code == 200:
                return round(time.monotonic() - start, 1)
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise SystemExit("server never became healthy")


def stream_events(base, events_url, headers=None, on_event=None):
    """Reads an SSE stream to the end -> list of (id, type, data, seconds_since_open)."""
    out, block = [], {}
    opened = time.monotonic()
    with httpx.stream("GET", base + events_url, headers=headers or {}, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line == "":
                if "event" in block:
                    ev = (int(block["id"]), block["event"], json.loads(block["data"]), round(time.monotonic() - opened, 2))
                    out.append(ev)
                    if on_event:
                        on_event(ev)
                block = {}
                continue
            if line.startswith(":") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            block[key] = value.strip()
    return out


def load_flower_rounds(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(int(r["round"]), r["hospital"])] = {m: float(r[m]) for m in METRICS}
    return rows


def load_flower_personalized(path):
    with open(path, newline="") as f:
        return {r["hospital"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


def compare(label, streamed, expected, problems):
    """streamed/expected: {key: {metric: value}} -> count of identical keys."""
    same = 0
    for key in sorted(set(streamed) | set(expected), key=str):
        a, b = streamed.get(key), expected.get(key)
        if a is None or b is None:
            problems.append(f"{label}: {key} missing ({'streamed' if a is None else 'expected'})")
            continue
        diffs = {m: abs(a[m] - b[m]) for m in METRICS}
        if max(diffs.values()) <= TOL:
            same += 1
        else:
            problems.append(f"{label}: {key} differs {diffs}")
    return same, len(set(streamed) | set(expected))


def run_one(base, body, problems, probe_busy=False):
    r = httpx.post(f"{base}/api/train", json=body, timeout=120)
    if r.status_code != 202:
        raise SystemExit(f"POST /api/train -> {r.status_code} {r.text}")
    run = r.json()
    busy = {}

    def on_event(ev):
        if probe_busy and ev[1] == "round" and ev[2]["round"] == 1 and not busy:
            second = httpx.post(f"{base}/api/train", json=body, timeout=30)
            busy["status"] = second.status_code
            busy["detail"] = second.json().get("detail")

    events = stream_events(base, run["events_url"], on_event=on_event)
    if probe_busy:
        if busy.get("status") != 409 or busy["detail"].get("active_run_id") != run["run_id"]:
            problems.append(f"second POST during a run should be 409 for {run['run_id']}, got {busy}")
    return run, events, busy


def main():
    base, ref_path, flower_dir, out_path = sys.argv[1:5]
    reference = json.load(open(ref_path))
    problems = []
    cold_start = wait_healthy(base)

    # --- FedAvg, 20 rounds ---
    run, events, busy = run_one(base, {"algorithm": "fedavg", "rounds": 20}, problems, probe_busy=True)
    types = [e[1] for e in events]
    if types != ["start"] + ["round"] * 20 + ["done"]:
        problems.append(f"fedavg event sequence wrong: {types}")
    if [e[0] for e in events] != list(range(1, len(events) + 1)):
        problems.append("fedavg event ids not sequential")
    rounds = [e for e in events if e[1] == "round"]
    streamed = {(e[2]["round"], h["hospital"]): {m: h[m] for m in METRICS} for e in rounds for h in e[2]["hospitals"]}
    fedavg_vs_flower = compare("fedavg vs Flower", streamed, load_flower_rounds(f"{flower_dir}/phase4_fedavg_per_round_per_hospital.csv"), problems)
    ref = {(e["round"], h["hospital"]): {m: h[m] for m in METRICS} for e in reference["fedavg"] for h in e["hospitals"]}
    fedavg_vs_direct = compare("fedavg vs direct Ray-free", streamed, ref, problems)
    # Streaming check: each round must reach the client when the server
    # produced it, not all at once at the end. Compare client arrival times
    # with the server's own elapsed_seconds stamps, relative to round 1.
    arrival = [e[3] for e in rounds]
    produced = [e[2]["elapsed_seconds"] for e in rounds]
    lag_drift = max(abs((a - arrival[0]) - (p - produced[0])) for a, p in zip(arrival, produced))
    spread = produced[-1] - produced[0]
    streamed_live = arrival == sorted(arrival) and spread >= 1.0 and lag_drift <= 1.0
    if not streamed_live:
        problems.append(f"round events not delivered as produced: arrival={arrival} produced={produced} "
                        f"drift={lag_drift:.2f}s spread={spread:.2f}s")

    # Resume after the fact: Last-Event-ID 5 -> ids 6..end, identical payloads.
    resumed = stream_events(base, run["events_url"], headers={"Last-Event-ID": "5"})
    if [e[0] for e in resumed] != list(range(6, len(events) + 1)) or [e[2] for e in resumed] != [e[2] for e in events[5:]]:
        problems.append("resume with Last-Event-ID 5 did not replay ids 6.. identically")

    # --- FedProx + personalization ---
    run2, events2, _ = run_one(base, {"algorithm": "fedprox", "rounds": 20, "personalize": True}, problems)
    types2 = [e[1] for e in events2]
    if types2 != ["start"] + ["round"] * 20 + ["personalized", "done"]:
        problems.append(f"fedprox event sequence wrong: {types2}")
    rounds2 = [e for e in events2 if e[1] == "round"]
    streamed2 = {(e[2]["round"], h["hospital"]): {m: h[m] for m in METRICS} for e in rounds2 for h in e[2]["hospitals"]}
    fedprox_vs_flower = compare("fedprox vs Flower", streamed2, load_flower_rounds(f"{flower_dir}/phase5_fedprox_per_round_per_hospital.csv"), problems)
    ref2 = {(e["round"], h["hospital"]): {m: h[m] for m in METRICS} for e in reference["fedprox"] for h in e["hospitals"]}
    fedprox_vs_direct = compare("fedprox vs direct Ray-free", streamed2, ref2, problems)
    pers = next(e[2] for e in events2 if e[1] == "personalized")
    pers_streamed = {h["hospital"]: {m: h[m] for m in METRICS} for h in pers["hospitals"]}
    pers_vs_flower = compare("personalized vs Phase 5", pers_streamed, load_flower_personalized(f"{flower_dir}/phase5_personalized_per_hospital.csv"), problems)

    summary = {
        "cold_start_seconds_to_healthy": cold_start,
        "fedavg": {
            "run_seconds": events[-1][2]["elapsed_seconds"],
            "first_round_arrived_after_s": arrival[0],
            "last_round_arrived_after_s": arrival[-1],
            "training_seconds_round1_to_20": round(spread, 2),
            "max_delivery_drift_s": round(lag_drift, 2),
            "final_global_accuracy": events[-1][2]["final_global_accuracy"],
            "matches_flower": f"{fedavg_vs_flower[0]}/{fedavg_vs_flower[1]}",
            "matches_direct_rayfree": f"{fedavg_vs_direct[0]}/{fedavg_vs_direct[1]}",
            "second_post_during_run": busy.get("status"),
        },
        "fedprox_personalized": {
            "run_seconds": events2[-1][2]["elapsed_seconds"],
            "final_global_accuracy_personalized": events2[-1][2]["final_global_accuracy"],
            "matches_flower": f"{fedprox_vs_flower[0]}/{fedprox_vs_flower[1]}",
            "matches_direct_rayfree": f"{fedprox_vs_direct[0]}/{fedprox_vs_direct[1]}",
            "personalized_matches_phase5": f"{pers_vs_flower[0]}/{pers_vs_flower[1]}",
        },
        "streamed_incrementally": streamed_live,
        "problems": problems,
    }
    json.dump({"summary": summary, "fedavg_events": events, "fedprox_events": events2}, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # record crashes so the workflow can report them
        json.dump({"summary": {"crashed": f"{type(e).__name__}: {e}", "problems": [f"crashed: {type(e).__name__}: {e}"]}},
                  open(sys.argv[4], "w"))
        raise
