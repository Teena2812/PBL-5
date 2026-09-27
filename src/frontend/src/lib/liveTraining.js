// Client for the live-training backend (src/backend/live_training.py):
// POST /api/train starts a run, GET /api/train/{id}/events streams it as
// Server-Sent Events. The backend runs the Ray-free loop, which reproduces
// the recorded Flower runs exactly (verified in CI).

const STATIC_BUILD = import.meta.env.VITE_STATIC_DATA === "true";

// Where the live backend lives. Static builds (GitHub Pages) only get one if
// VITE_LIVE_API_URL is set at build time; local dev falls back to the
// FastAPI dev server.
export const LIVE_API_URL =
  import.meta.env.VITE_LIVE_API_URL || (STATIC_BUILD ? null : "http://localhost:8000/api");

/**
 * Checks the backend is reachable, waiting through a cold start (a free
 * Render instance takes ~1 minute to wake). Resolves {ok, busy, waitedSeconds}
 * or {ok: false, reason}. onWaiting(seconds) is called while waiting.
 */
export async function wakeBackend({ timeoutSeconds = 120, onWaiting } = {}) {
  if (!LIVE_API_URL) return { ok: false, reason: "not-configured" };
  const started = Date.now();
  while ((Date.now() - started) / 1000 < timeoutSeconds) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 10000);
      const r = await fetch(`${LIVE_API_URL}/train/status`, { signal: controller.signal });
      clearTimeout(timer);
      if (r.ok) {
        const s = await r.json();
        return { ok: true, busy: s.busy, waitedSeconds: Math.round((Date.now() - started) / 1000) };
      }
    } catch {
      // not up yet (or waking) -- keep trying
    }
    onWaiting?.(Math.round((Date.now() - started) / 1000));
    await new Promise((res) => setTimeout(res, 3000));
  }
  return { ok: false, reason: "unreachable" };
}

/**
 * Starts one run and streams its events. Returns a handle with close().
 * onEvent receives parsed event objects (start, round, personalized, done,
 * error); onFailure receives {status, detail} if the run can't start.
 */
export async function startLiveRun(config, { onEvent, onFailure }) {
  let response;
  try {
    response = await fetch(`${LIVE_API_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
  } catch (e) {
    onFailure?.({ status: 0, detail: `Could not reach the live-training server (${e.message}).` });
    return { close() {} };
  }
  if (response.status !== 202) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : body.detail?.message || detail;
    } catch {
      /* keep the status text */
    }
    onFailure?.({ status: response.status, detail });
    return { close() {} };
  }

  const { events_url: eventsUrl } = await response.json();
  // events_url is "/api/train/{id}/events"; LIVE_API_URL already ends in /api.
  const source = new EventSource(`${LIVE_API_URL}${eventsUrl.replace(/^\/api/, "")}`);
  let finished = false;
  for (const type of ["start", "round", "personalized", "done", "error"]) {
    source.addEventListener(type, (e) => {
      const event = JSON.parse(e.data);
      onEvent?.(event);
      if (type === "done" || type === "error") {
        finished = true;
        source.close();
      }
    });
  }
  // EventSource reconnects on its own (resuming via Last-Event-ID); only
  // give up if the connection is closed for good before the run finished.
  source.onerror = () => {
    if (!finished && source.readyState === EventSource.CLOSED) {
      onFailure?.({ status: 0, detail: "The live stream was closed before the run finished." });
    }
  };
  return {
    close() {
      finished = true;
      source.close();
    },
  };
}

/**
 * Specificity and balanced accuracy for one hospital row of a live run,
 * using that site's test-set class counts (hospitals.json) -- the same
 * derivation the static dashboard data uses.
 */
export function withBalancedAccuracy(row, site) {
  const tp = Math.round(row.recall * site.test_disease);
  const tn = Math.round(row.accuracy * site.n_test) - tp;
  const specificity = site.test_no_disease ? tn / site.test_no_disease : NaN;
  return { ...row, specificity, balanced_accuracy: (row.recall + specificity) / 2 };
}

/** Test-size-weighted overall value of a metric across hospital rows. */
export function weighted(rows, key) {
  const total = rows.reduce((s, r) => s + r.n_test, 0);
  return rows.reduce((s, r) => s + r[key] * r.n_test, 0) / total;
}
