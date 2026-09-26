"""
Tests for the live-training SSE endpoints (src/backend/live_training.py).

These use a stand-in trainer, so they check the HTTP/streaming contract --
event order and format, ids and resume, the one-run lock, error handling --
without torch. The real training loop is verified separately against the
Flower results (see .github/workflows/live-training-check.yml).

Run with pytest, or directly: python tests/test_live_training.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from src.backend import live_training  # noqa: E402
from src.backend.main import app  # noqa: E402

client = TestClient(app)


def fake_trainer(delay=0.0, fail_at=None, gate=None):
    def trainer(req):
        def events():
            yield {"type": "meta", "hospitals": [{"hospital": "a", "n_patients": 10, "n_train": 7, "n_test": 3}],
                   "n_features": 13}
            if gate is not None:
                gate.wait(5)
            for r in range(1, req.rounds + 1):
                if fail_at == r:
                    raise RuntimeError("boom")
                time.sleep(delay)
                yield {"type": "round", "round": r,
                       "hospitals": [{"hospital": "a", "n_test": 3, "accuracy": r / 100}],
                       "global_accuracy": r / 100}
            if req.personalize:
                yield {"type": "personalized", "hospitals": [], "global_accuracy": 0.5}
        return events()
    return trainer


def parse_sse(text):
    """-> list of (id, event, data) for every event block in an SSE body."""
    out = []
    for block in text.strip().split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if line.startswith(":") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key] = value.strip()
        if "event" in fields:
            out.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return out


def wait_idle(timeout=5):
    end = time.time() + timeout
    while client.get("/api/train/status").json()["busy"] and time.time() < end:
        time.sleep(0.02)


def teardown():
    wait_idle()
    live_training.set_trainer(None)


def test_full_stream_in_order():
    live_training.set_trainer(fake_trainer())
    try:
        r = client.post("/api/train", json={"algorithm": "fedprox", "rounds": 5, "personalize": True})
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]
        body = client.get(r.json()["events_url"]).text
        events = parse_sse(body)
        assert body.startswith("retry: 3000")
        assert [e[1] for e in events] == ["start"] + ["round"] * 5 + ["personalized", "done"]
        assert [e[0] for e in events] == list(range(1, 9))
        assert events[0][2]["config"] == {"algorithm": "fedprox", "rounds": 5, "personalize": True, "seed": 42}
        assert events[0][2]["hospitals"][0]["hospital"] == "a" and events[0][2]["n_features"] == 13
        assert [e[2]["round"] for e in events[1:6]] == [1, 2, 3, 4, 5]
        assert events[-1][2]["final_global_accuracy"] == 0.5
        snap = client.get(f"/api/train/{run_id}").json()
        assert snap["status"] == "done" and len(snap["events"]) == 8
    finally:
        teardown()


def test_second_run_rejected_while_busy():
    gate = threading.Event()
    live_training.set_trainer(fake_trainer(gate=gate))
    try:
        first = client.post("/api/train", json={"rounds": 2})
        assert first.status_code == 202
        second = client.post("/api/train", json={"rounds": 2})
        assert second.status_code == 409
        assert second.json()["detail"]["active_run_id"] == first.json()["run_id"]
        assert client.get("/api/train/status").json() == {"busy": True, "active_run_id": first.json()["run_id"]}
        gate.set()
        wait_idle()
        assert client.get("/api/train/status").json()["busy"] is False
        assert client.post("/api/train", json={"rounds": 1}).status_code == 202   # free again
    finally:
        gate.set()
        teardown()


def test_resume_with_last_event_id():
    live_training.set_trainer(fake_trainer())
    try:
        url = client.post("/api/train", json={"rounds": 4}).json()["events_url"]
        full = parse_sse(client.get(url).text)
        resumed = parse_sse(client.get(url, headers={"Last-Event-ID": "3"}).text)
        assert [e[0] for e in resumed] == [4, 5, 6]
        assert resumed == full[3:]
        # reconnecting after "done" returns immediately with nothing new
        assert parse_sse(client.get(url, headers={"Last-Event-ID": str(len(full))}).text) == []
    finally:
        teardown()


def test_training_error_is_streamed_and_releases_lock():
    live_training.set_trainer(fake_trainer(fail_at=3))
    try:
        r = client.post("/api/train", json={"rounds": 5})
        events = parse_sse(client.get(r.json()["events_url"]).text)
        assert [e[1] for e in events] == ["start", "round", "round", "error"]
        assert "boom" in events[-1][2]["detail"]
        wait_idle()
        assert client.get(f"/api/train/{r.json()['run_id']}").json()["status"] == "error"
        assert client.get("/api/train/status").json()["busy"] is False
    finally:
        teardown()


def test_validation_and_unknown_run():
    assert client.post("/api/train", json={"rounds": 0}).status_code == 422
    assert client.post("/api/train", json={"rounds": 21}).status_code == 422
    assert client.post("/api/train", json={"algorithm": "sgd"}).status_code == 422
    assert client.get("/api/train/nope/events").status_code == 404
    assert client.get("/api/train/nope").status_code == 404


def test_without_torch_returns_503_and_stays_free():
    """Only meaningful where torch is missing (e.g. the project's local
    Windows machine); skipped elsewhere."""
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    live_training.set_trainer(None)
    r = client.post("/api/train", json={"rounds": 1})
    assert r.status_code == 503, r.text
    assert "torch" in r.json()["detail"]
    assert client.get("/api/train/status").json()["busy"] is False


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
