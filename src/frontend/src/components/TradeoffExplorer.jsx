import { useMemo, useRef, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { EXPERIMENT_COLORS, EXPERIMENT_LABELS, EXPERIMENT_ORDER, fmtPct, hospitalLabel } from "../constants/experiments";
import { LIVE_API_URL, startLiveRun, wakeBackend, weighted, withBalancedAccuracy } from "../lib/liveTraining";
import "./TradeoffExplorer.css";

const METRICS = [
  { id: "accuracy", label: "Accuracy" },
  { id: "balanced_accuracy", label: "Balanced accuracy" },
];

function summarize(rows, metric) {
  const values = rows.map((r) => ({ hospital: r.hospital, value: r[metric] }));
  const worst = values.reduce((a, b) => (b.value < a.value ? b : a));
  const best = values.reduce((a, b) => (b.value > a.value ? b : a));
  return { overall: weighted(rows, metric), worst: worst.value, worstHospital: worst.hospital, gap: best.value - worst.value };
}

/**
 * Overall performance vs the worst-served hospital's performance, per
 * training setting. Recorded points: the seed-42 single runs of all five
 * settings (experiment_comparison.json). Live points: runs the viewer
 * starts on the backend, computed from the streamed per-hospital results.
 */
export default function TradeoffExplorer({ comparison, hospitals }) {
  const [metric, setMetric] = useState("accuracy");
  const [liveRuns, setLiveRuns] = useState([]);
  const sites = Object.fromEntries(hospitals.map((h) => [h.hospital, h]));

  const recorded = useMemo(
    () =>
      EXPERIMENT_ORDER.filter((k) => comparison.per_hospital[k]).map((k) => ({
        key: k,
        label: EXPERIMENT_LABELS[k],
        rows: comparison.per_hospital[k],
        ...summarize(comparison.per_hospital[k], metric),
      })),
    [comparison, metric]
  );
  const live = liveRuns.map((r) => ({ ...r, ...summarize(r.rows, metric) }));
  const all = [...recorded, ...live];
  // Separate, zoomed axes (5-point grid) so close points stay readable.
  const axis = (values, pad) => {
    const lo = Math.max(0, Math.floor((Math.min(...values) - pad) * 20) / 20);
    const hi = Math.min(1, Math.ceil((Math.max(...values) + pad) * 20) / 20);
    const ticks = [];
    for (let t = Math.round(lo * 20); t <= Math.round(hi * 20); t++) ticks.push(t / 20);
    return { lo, hi, ticks };
  };
  const xAxis = axis(all.map((p) => p.overall), 0.02);
  const yAxis = axis([...all.map((p) => p.worst), ...(metric === "balanced_accuracy" ? [0.5] : [])], 0.02);
  const chance = metric === "balanced_accuracy";

  return (
    <div className="tradeoff">
      <div className="card">
        <div className="tradeoff-head">
          <div>
            <h3>Fairness vs accuracy</h3>
            <p className="muted-note" style={{ marginBottom: 0 }}>
              Right = better overall; up = better for the hospital each setting serves worst. Recorded points: the
              model-init seed 42 run of each setting.
            </p>
          </div>
          <div className="segmented" role="radiogroup" aria-label="Metric">
            {METRICS.map((m) => (
              <button
                key={m.id}
                type="button"
                role="radio"
                aria-checked={metric === m.id}
                className={"segmented-option" + (metric === m.id ? " active" : "")}
                onClick={() => setMetric(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={340}>
          <ScatterChart margin={{ top: 16, right: 24, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              dataKey="overall"
              domain={[xAxis.lo, xAxis.hi]}
              ticks={xAxis.ticks}
              tickFormatter={(v) => fmtPct(v, 0)}
              tick={{ fontSize: 11 }}
              name="Overall"
              label={{ value: `Overall ${metric === "accuracy" ? "accuracy" : "balanced accuracy"}`, position: "insideBottom", offset: -4, fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="worst"
              domain={[yAxis.lo, yAxis.hi]}
              ticks={yAxis.ticks}
              tickFormatter={(v) => fmtPct(v, 0)}
              tick={{ fontSize: 11 }}
              name="Worst hospital"
              label={{ value: "Worst hospital", angle: -90, position: "insideLeft", offset: 12, fontSize: 11 }}
            />
            {chance && <ReferenceLine y={0.5} stroke="var(--color-text-muted)" strokeDasharray="4 4" label={{ value: "chance", fontSize: 10, position: "insideTopLeft" }} />}
            <Tooltip content={<PointTooltip metric={metric} />} />
            <Scatter data={recorded} shape={<Dot />}>
              {recorded.map((p) => (
                <Cell key={p.key} fill={EXPERIMENT_COLORS[p.key]} />
              ))}
            </Scatter>
            {live.length > 0 && (
              <Scatter data={live} shape={<Dot hollow />}>
                {live.map((p) => (
                  <Cell key={p.id} fill={EXPERIMENT_COLORS[p.algorithm]} />
                ))}
              </Scatter>
            )}
          </ScatterChart>
        </ResponsiveContainer>

        <table className="tradeoff-table">
          <thead>
            <tr>
              <th>Setting</th>
              <th>Overall</th>
              <th>Worst hospital</th>
              <th>Gap best–worst</th>
            </tr>
          </thead>
          <tbody>
            {all.map((p) => (
              <tr key={p.key ?? p.id}>
                <td>
                  <span
                    className={"tradeoff-swatch" + (p.id ? " hollow" : "")}
                    style={{ background: p.id ? undefined : EXPERIMENT_COLORS[p.key], borderColor: EXPERIMENT_COLORS[p.key ?? p.algorithm] }}
                  />
                  {p.label}
                </td>
                <td>{fmtPct(p.overall)}</td>
                <td>
                  {fmtPct(p.worst)} <span className="muted-inline">({hospitalLabel(p.worstHospital)})</span>
                </td>
                <td>{(p.gap * 100).toFixed(1)} pp</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted-note" style={{ marginTop: 10, marginBottom: 0 }}>
          With balanced accuracy the picture changes: in every recorded run, Switzerland&apos;s model recognises none
          of its 2 healthy test patients, so its balanced accuracy is at or below chance. Per-hospital results for
          every setting were recorded for seed 42 only; the multi-seed runs saved overall accuracy.
        </p>
      </div>

      <LiveRunPanel
        sites={sites}
        recorded={recorded}
        onResult={(run) => setLiveRuns((runs) => [...runs, run])}
      />
    </div>
  );
}

function LiveRunPanel({ sites, recorded, onResult }) {
  const [algorithm, setAlgorithm] = useState("fedavg");
  const [seed, setSeed] = useState(42);
  const [status, setStatus] = useState({ state: "idle" });
  const handle = useRef(null);
  const count = useRef(0);

  if (!LIVE_API_URL) {
    return (
      <div className="card live-panel">
        <h3>Add a live run</h3>
        <p className="muted-note" style={{ margin: 0 }}>
          Live training isn&apos;t connected on this deployment yet (no backend URL configured). The recorded points
          above come from the confirmed experiment runs.
        </p>
      </div>
    );
  }

  const run = async () => {
    setStatus({ state: "waking", seconds: 0 });
    const woke = await wakeBackend({ onWaiting: (s) => setStatus({ state: "waking", seconds: s }) });
    if (!woke.ok) {
      setStatus({ state: "error", detail: "The live-training server didn't respond. Try again in a minute." });
      return;
    }
    setStatus({ state: "running", round: 0 });
    let last = null;
    handle.current = await startLiveRun(
      { algorithm, seed: Number(seed), rounds: 20, personalize: false },
      {
        onEvent: (e) => {
          if (e.type === "round") {
            last = e;
            setStatus({ state: "running", round: e.round });
          } else if (e.type === "done") {
            const rows = last.hospitals.map((r) => withBalancedAccuracy(r, sites[r.hospital]));
            count.current += 1;
            const label = `Live ${EXPERIMENT_LABELS[algorithm]}, seed ${seed}`;
            const ref = recorded.find((p) => p.key === algorithm);
            const matchesRecorded =
              Number(seed) === 42 && ref &&
              ref.rows.every((r) => {
                const l = rows.find((x) => x.hospital === r.hospital);
                return l && Math.abs(l.accuracy - r.accuracy) < 1e-4;
              });
            onResult({ id: `live-${count.current}`, algorithm, seed: Number(seed), label, shortLabel: `live #${count.current}`, rows });
            setStatus({ state: "done", seconds: e.elapsed_seconds, matchesRecorded, label });
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
    <div className="card live-panel">
      <h3>Add a live run</h3>
      <p className="muted-note">
        Trains FedAvg or FedProx for real on the server (20 rounds, the 4 hospitals, no patient data leaves a
        hospital) and adds the result as a hollow point.
      </p>
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
          {busy ? "Running…" : "▶ Train live"}
        </button>
      </div>
      <p className="live-status" aria-live="polite">
        {status.state === "waking" && `Waking up the server… ${status.seconds}s (a sleeping free server takes about a minute)`}
        {status.state === "running" && `Training… round ${status.round} / 20`}
        {status.state === "done" &&
          `${status.label}: done in ${status.seconds}s.${status.matchesRecorded ? " ✓ Identical to the recorded seed-42 run at every hospital." : ""}`}
        {status.state === "error" && <span className="live-error">{status.detail}</span>}
      </p>
    </div>
  );
}

// Point with its label drawn alongside (LabelList doesn't render for a
// Scatter with a custom shape in Recharts 3). Labels alternate sides by
// setting order so near-identical points (FedAvg / FedProx) stay legible.
const LABEL_LEFT = new Set(["fedprox", "centralized"]);

function Dot({ cx, cy, fill, hollow, payload }) {
  if (cx == null || cy == null) return null;
  const left = hollow || LABEL_LEFT.has(payload?.key);
  const text = hollow ? payload?.shortLabel : payload?.label;
  return (
    <g>
      {hollow ? (
        <circle cx={cx} cy={cy} r={7} fill="var(--color-surface)" stroke={fill} strokeWidth={3} />
      ) : (
        <circle cx={cx} cy={cy} r={8} fill={fill} stroke="#ffffff" strokeWidth={2} />
      )}
      {text && (
        <text
          x={left ? cx - 12 : cx + 12}
          y={cy + (hollow ? 16 : 4)}
          textAnchor={left ? "end" : "start"}
          fontSize={hollow ? 10 : 11}
          fill={hollow ? "var(--color-text-muted)" : "var(--color-text)"}
        >
          {text}
        </text>
      )}
    </g>
  );
}

function PointTooltip({ active, payload, metric }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{p.label}</strong>
      <div>Overall: {fmtPct(p.overall)}</div>
      <div>
        Worst hospital: {fmtPct(p.worst)} ({hospitalLabel(p.worstHospital)})
      </div>
      {p.rows.map((r) => (
        <div key={r.hospital} className="muted-inline">
          {hospitalLabel(r.hospital)}: {fmtPct(r[metric])}
        </div>
      ))}
    </div>
  );
}
