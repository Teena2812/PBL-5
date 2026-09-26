"""
Live federated training over Server-Sent Events.

  POST /api/train                  start one run -> 202 {"run_id", "events_url"}
                                   (409 if a run is already in progress)
  GET  /api/train/status           {"busy": bool, "active_run_id": str | None}
  GET  /api/train/{run_id}         JSON snapshot: status + every event so far
  GET  /api/train/{run_id}/events  text/event-stream: start, round x N,
                                   [personalized], then done | error

Training uses the Ray-free loop (src/federated/rayfree_runner.py), which
fits Render's free instance (512 MB / 0.1 CPU) where Flower's Ray engine
does not, and reproduces the Flower runs exactly. Only one run executes at
a time: on 0.1 CPU two concurrent runs would each take twice as long.

Events are numbered (SSE "id:") and kept for the life of the run, so a
browser that reconnects with Last-Event-ID resumes where it left off
instead of starting a second training run.

torch is imported only when a run starts, so this router (and the rest of
the backend) still loads where torch is unavailable; starting a run there
returns 503.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/train", tags=["live-training"])

SPLIT_SEED = 42          # same local train/test split as every experiment phase
PROXIMAL_MU = 0.1        # same as Phase 5
MAX_KEPT_RUNS = 20
HEARTBEAT_SECONDS = 15
POLL_SECONDS = 0.2


class TrainRequest(BaseModel):
    algorithm: Literal["fedavg", "fedprox"] = "fedavg"
    rounds: int = Field(20, ge=1, le=20, description="Federated rounds (the experiments use 20)")
    personalize: bool = Field(False, description="After the last round, fine-tune locally at each hospital")
    seed: int = Field(42, ge=0, le=100_000, description="Model-initialization seed (experiments: 42, 1, 7, 123, 2024)")


@dataclass
class Run:
    run_id: str
    config: dict
    events: list[dict] = field(default_factory=list)
    status: str = "running"   # running | done | error


# A trainer takes the request and returns an iterator of events. The first
# may be {"type": "meta", ...} (merged into the "start" event); the rest are
# round/personalized events. The trainer itself must return quickly (it runs
# inside the POST request) -- slow work such as loading data belongs inside
# the iterator, which runs on the background thread. Replaceable in tests.
Trainer = Callable[[TrainRequest], Iterator[dict]]

_runs: dict[str, Run] = {}
_active_run_id: str | None = None
_state_lock = threading.Lock()
_partition_cache: dict = {}


def _live_partitions():
    """The 4 real sites with the experiments' local split, built once."""
    if "partitions" not in _partition_cache:
        from src.data.multisite import create_site_partitions, load_multisite
        from src.data.partition import add_local_train_test_split

        x, y, df_clean = load_multisite()
        partitions = add_local_train_test_split(create_site_partitions(x, y, df_clean), test_size=0.25, seed=SPLIT_SEED)
        _partition_cache["partitions"] = partitions
        _partition_cache["n_features"] = x.shape[1]
    return _partition_cache["partitions"], _partition_cache["n_features"]


def default_trainer(req: TrainRequest) -> Iterator[dict]:
    try:
        import torch

        from src.federated.rayfree_runner import iter_federated_training
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Live training needs torch in this environment ({e}).") from e

    def events() -> Iterator[dict]:
        torch.set_num_threads(1)  # one core's worth is all a free instance has
        partitions, n_features = _live_partitions()   # first run only: loads the 4 sites
        yield {
            "type": "meta",
            "hospitals": [
                {"hospital": p.name, "n_patients": len(p), "n_train": len(p.x_train), "n_test": len(p.x_test)}
                for p in partitions
            ],
            "n_features": n_features,
        }
        yield from iter_federated_training(
            partitions,
            n_features,
            model_init_seed=req.seed,
            n_rounds=req.rounds,
            proximal_mu=PROXIMAL_MU if req.algorithm == "fedprox" else 0.0,
            personalize=req.personalize,
        )

    return events()


_trainer: Trainer = default_trainer


def set_trainer(trainer: Trainer | None) -> None:
    """Swap the trainer (tests); None restores the real one."""
    global _trainer
    _trainer = trainer or default_trainer


def _append(run: Run, event: dict) -> None:
    with _state_lock:
        run.events.append(event)


def _execute(run: Run, events: Iterator[dict]) -> None:
    global _active_run_id
    started = time.perf_counter()
    try:
        last = None
        sent_start = False
        for event in events:
            if event["type"] == "meta":
                meta = {k: v for k, v in event.items() if k != "type"}
                _append(run, {"type": "start", "config": run.config, **meta})
                sent_start = True
                continue
            if not sent_start:
                _append(run, {"type": "start", "config": run.config})
                sent_start = True
            event = {**event, "elapsed_seconds": round(time.perf_counter() - started, 2)}
            _append(run, event)
            last = event
        if not sent_start:
            _append(run, {"type": "start", "config": run.config})
        _append(run, {
            "type": "done",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "final_global_accuracy": last["global_accuracy"] if last else None,
        })
        run.status = "done"
    except Exception as e:  # noqa: BLE001 -- surface any training failure to the client
        _append(run, {"type": "error", "detail": f"{type(e).__name__}: {e}"})
        run.status = "error"
    finally:
        with _state_lock:
            if _active_run_id == run.run_id:
                _active_run_id = None


@router.post("", status_code=202)
def start_training(req: TrainRequest) -> dict:
    global _active_run_id
    with _state_lock:
        if _active_run_id is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A training run is already in progress.", "active_run_id": _active_run_id},
            )
        run = Run(run_id=uuid.uuid4().hex[:12], config=req.model_dump())
        _active_run_id = run.run_id
        _runs[run.run_id] = run
        while len(_runs) > MAX_KEPT_RUNS:  # forget the oldest finished runs
            oldest = next(iter(_runs))
            if oldest == run.run_id:
                break
            _runs.pop(oldest)

    try:
        events = _trainer(req)
    except Exception:
        with _state_lock:
            _active_run_id = None
            _runs.pop(run.run_id, None)
        raise

    threading.Thread(target=_execute, args=(run, events), daemon=True).start()
    return {"run_id": run.run_id, "events_url": f"/api/train/{run.run_id}/events"}


@router.get("/status")
def training_status() -> dict:
    return {"busy": _active_run_id is not None, "active_run_id": _active_run_id}


def _get_run(run_id: str) -> Run:
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id!r}")
    return run


@router.get("/{run_id}")
def run_snapshot(run_id: str) -> dict:
    run = _get_run(run_id)
    with _state_lock:
        return {"run_id": run.run_id, "status": run.status, "config": run.config, "events": list(run.events)}


def _sse(event_id: int, event: dict) -> str:
    return f"id: {event_id}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"


@router.get("/{run_id}/events")
async def run_events(run_id: str, request: Request) -> StreamingResponse:
    run = _get_run(run_id)
    last_id = request.headers.get("last-event-id") or request.query_params.get("last_event_id") or "0"
    try:
        next_index = max(int(last_id), 0)   # ids are 1-based, so resume at that index
    except ValueError:
        next_index = 0

    async def stream():
        nonlocal next_index
        last_beat = time.monotonic()
        yield "retry: 3000\n\n"
        while True:
            with _state_lock:
                pending = run.events[next_index:]
                finished = run.status != "running"
            if not pending and finished:
                return   # client already has every event (e.g. reconnected after "done")
            for event in pending:
                next_index += 1
                yield _sse(next_index, event)
                if event["type"] in ("done", "error"):
                    return
            if await request.is_disconnected():
                return
            if time.monotonic() - last_beat >= HEARTBEAT_SECONDS:
                yield ": keep-alive\n\n"
                last_beat = time.monotonic()
            await asyncio.sleep(POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
