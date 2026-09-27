import { useEffect, useRef, useState } from "react";
import TrainingReplay from "./TrainingReplay";
import { EXPERIMENT_LABELS } from "../constants/experiments";
import { LIVE_API_URL, startLiveRun, wakeBackend } from "../lib/liveTraining";
import "./TradeoffExplorer.css";

const RECORDED_SEED = 42; // the recorded training_curves.json runs use model-init seed 42

/**
 * "Run it live": streams one real training run from the backend into the
 * replay visuals, then (for the recorded seed) checks every round and
 * hospital against the recorded run.
 */
export default function LiveReplay({ curves, hospitals, summary }) {
  const [algorithm, setAlgorithm] = useState("fedavg");
  const [seed, setSeed] = useState(RECORDED_SEED);
  const [rounds, setRounds] = useState([]);
  const [status, setStatus] = useState({ state: "idle" });
  const handle = useRef(null);

  useEffect(() => () => handle.current?.close(), []);

  if (!LIVE_API_URL) {
    return (
      <div className="card">
        <p className="muted-note" style={{ margin: 0 }}>
          Live training isn&apos;t connected on this deployment yet (no backend URL configured). The recorded runs are
          available under <em>Recorded run</em>.
        </p>
      </div>
    );
  }

  const compareWithRecorded = (liveRounds, algo) => {
    const rec = new Map(curves[algo].per_hospital_by_round.map((r) => [`${r.round}|${r.hospital}`, r.accuracy]));
    let same = 0;
    let total = 0;
    for (const e of liveRounds) {
      for (const h of e.hospitals) {
        total += 1;
        if (Math.abs(h.accuracy - (rec.get(`${e.round}|${h.hospital}`) ?? NaN)) < 1e-4) same += 1;
      }
    }
    return { same, total };
  };

  const run = async () => {
    handle.current?.close();
    setRounds([]);
    setStatus({ state: "waking", seconds: 0 });
    const woke = await wakeBackend({ onWaiting: (s) => setStatus({ state: "waking", seconds: s }) });
    if (!woke.ok) {
      setStatus({ state: "error", detail: "The live-training server didn't respond. Try again in a minute." });
      return;
    }
    const algo = algorithm;
    const runSeed = Number(seed);
    const collected = [];
    setStatus({ state: "running" });
    handle.current = await startLiveRun(
      { algorithm: algo, seed: runSeed, rounds: 20, personalize: false },
      {
        onEvent: (e) => {
          if (e.type === "round") {
            collected.push(e);
            setRounds([...collected]);
          } else if (e.type === "done") {
            const check = runSeed === RECORDED_SEED ? compareWithRecorded(collected, algo) : null;
            setStatus({ state: "done", seconds: e.elapsed_seconds, check, algo, runSeed });
          } else if (e.type === "error") {
            setStatus({ state: "error", detail: e.detail });
          }
        },
        onFailure: ({ status: code, detail }) =>
          setStatus({ state: "error", detail: code === 409 ? "Another live run is in progress -- try again when it finishes." : detail }),
      }
    );
  };

  const busy = status.state === "waking" || status.state === "running";
  return (
    <>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="live-controls">
          <div className="segmented" role="radiogroup" aria-label="Algorithm">
            {["fedavg", "fedprox"].map((a) => (
              <button
                key={a}
                type="button"
                role="radio"
                aria-checked={algorithm === a}
                className={"segmented-option" + (algorithm === a ? " active" : "")}
                onClick={() => setAlgorithm(a)}
                disabled={busy}
              >
                {EXPERIMENT_LABELS[a]}
              </button>
            ))}
          </div>
          <label className="live-seed">
            Seed
            <input type="number" min={0} max={100000} value={seed} onChange={(e) => setSeed(e.target.value)} disabled={busy} />
          </label>
          <button type="button" className="button-primary" onClick={run} disabled={busy}>
            {busy ? "Running…" : rounds.length ? "↻ Run again" : "▶ Train live"}
          </button>
        </div>
        <p className="live-status" aria-live="polite">
          {status.state === "idle" &&
            "Trains for real on the server: 4 hospitals, 20 rounds, model weights only -- no patient record leaves a hospital."}
          {status.state === "waking" && `Waking up the server… ${status.seconds}s (a sleeping free server takes about a minute)`}
          {status.state === "running" && `Training… ${rounds.length} / 20 rounds received`}
          {status.state === "done" &&
            `Done in ${status.seconds}s. ` +
              (status.check
                ? status.check.same === status.check.total
                  ? `✓ Identical to the recorded ${EXPERIMENT_LABELS[status.algo]} run: ${status.check.same}/${status.check.total} round-by-hospital accuracies match.`
                  : `${status.check.same}/${status.check.total} round-by-hospital accuracies match the recorded run.`
                : `No recorded run exists for seed ${status.runSeed}; this one is new.`)}
          {status.state === "error" && <span className="live-error">{status.detail}</span>}
        </p>
      </div>
      {rounds.length > 0 && (
        <TrainingReplay
          curves={curves}
          hospitals={hospitals}
          summary={summary}
          live={{ algorithm: status.algo ?? algorithm, rounds, totalRounds: 20 }}
        />
      )}
    </>
  );
}
