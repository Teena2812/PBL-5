import { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";
import { HOSPITAL_COLORS, fmtPct, hospitalLabel } from "../constants/experiments";
import "./TrainingReplay.css";

const ALGORITHMS = [
  { id: "fedavg", label: "FedAvg" },
  { id: "fedprox", label: "FedProx (μ = 0.1)" },
];
const STEP_MS = 1300;
const GLOBAL_COLOR = "var(--color-text)";

// Diagram geometry (SVG user units).
const SERVER = { x: 260, y: 58 };
const HOSPITAL_Y = 262;
const HOSPITAL_X = [60, 160, 260, 360, 460];

/** Parameter count of HeartDiseaseNet from its layer sizes (22 -> 16 -> 8 -> 1). */
function countParameters(nFeatures, hiddenSizes) {
  const sizes = [nFeatures, ...hiddenSizes, 1];
  let total = 0;
  for (let i = 1; i < sizes.length; i++) total += sizes[i - 1] * sizes[i] + sizes[i];
  return total;
}

/**
 * Round-by-round replay of the real federated training runs
 * (training_curves.json). Per-hospital values are the shared global
 * model's accuracy on each hospital's own local test set after that
 * round's aggregation; "global" weights them by test-set size.
 */
export default function TrainingReplay({ curves, hospitals, summary, autoPlay = false, compact = false }) {
  const [algorithm, setAlgorithm] = useState("fedavg");
  const [round, setRound] = useState(1);
  const [playing, setPlaying] = useState(autoPlay);
  const [focus, setFocus] = useState(null);

  const hospitalIds = hospitals.map((h) => h.hospital).sort();
  const nTrain = Object.fromEntries(hospitals.map((h) => [h.hospital, h.n_train]));
  const nParams = countParameters(summary.dataset.n_features, summary.model_architecture.hidden_sizes);

  const { rows, nRounds, minTest, yMin } = useMemo(() => {
    const run = curves[algorithm];
    const byRound = new Map(run.global_by_round.map((g) => [g.round, { round: g.round, global: g.global_weighted_accuracy }]));
    for (const r of run.per_hospital_by_round) byRound.get(r.round)[r.hospital] = r.accuracy;
    const sorted = [...byRound.values()].sort((a, b) => a.round - b.round);
    // Zoom the axis to just below the lowest value in either run, so the
    // axis doesn't jump when switching algorithms.
    const lowest = Math.min(
      ...["fedavg", "fedprox"].flatMap((a) => curves[a].per_hospital_by_round.map((r) => r.accuracy))
    );
    return {
      rows: sorted,
      nRounds: sorted.length,
      minTest: Math.min(...run.per_hospital_by_round.map((r) => r.n_test)),
      yMin: Math.max(0, Math.floor(lowest * 10) / 10),
    };
  }, [curves, algorithm]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = setInterval(() => {
      setRound((r) => {
        if (r >= nRounds) {
          setPlaying(false);
          return r;
        }
        return r + 1;
      });
    }, STEP_MS);
    return () => clearInterval(timer);
  }, [playing, nRounds]);

  const togglePlay = () => {
    if (!playing && round >= nRounds) setRound(1);
    setPlaying((p) => !p);
  };

  const yTicks = [];
  for (let t = Math.round(yMin * 10); t <= 10; t++) yTicks.push(t / 10);
  // Mark only each line's newest point, so the head of every line is
  // visible even at round 1 when a line is a single point.
  const headDot = (color) =>
    function HeadDot({ cx, cy, index }) {
      if (index !== round - 1 || cx == null || cy == null) return <g key={index} />;
      return <circle key={index} cx={cx} cy={cy} r={4.5} fill={color} stroke="#ffffff" strokeWidth={1.5} />;
    };

  const current = rows[round - 1];
  const first = rows[0];
  const visible = rows.slice(0, round);
  const weightsExchanged = round * hospitalIds.length * 2 * nParams;

  return (
    <div className={"replay" + (compact ? " replay-compact" : "")}>
      <div className="replay-controls">
        {!compact && (
          <div className="segmented replay-algos" role="radiogroup" aria-label="Training algorithm">
            {ALGORITHMS.map((a) => (
              <button
                key={a.id}
                type="button"
                role="radio"
                aria-checked={a.id === algorithm}
                className={"segmented-option" + (a.id === algorithm ? " active" : "")}
                onClick={() => setAlgorithm(a.id)}
              >
                {a.label}
              </button>
            ))}
          </div>
        )}
        <button type="button" className="button-primary replay-play" onClick={togglePlay} aria-pressed={playing}>
          {playing ? "❚❚ Pause" : round >= nRounds ? "↻ Replay" : "▶ Play"}
        </button>
        <label className="replay-scrubber">
          <span>
            Round <strong>{round}</strong> / {nRounds}
          </span>
          <input
            type="range"
            min={1}
            max={nRounds}
            step={1}
            value={round}
            onChange={(e) => {
              setPlaying(false);
              setRound(Number(e.target.value));
            }}
            aria-label="Training round"
          />
        </label>
      </div>

      {compact ? (
        <p className="replay-inline-stats">
          Shared model <strong>{fmtPct(current.global)}</strong> &middot;{" "}
          <strong>{weightsExchanged.toLocaleString("en-US")}</strong> weights exchanged ({round} × {hospitalIds.length}{" "}
          hospitals × {nParams}, each way)
        </p>
      ) : (
        <div className="grid grid-stats replay-stats">
          <div className="replay-stat">
            <span className="replay-stat-label">Shared model accuracy</span>
            <span className="replay-stat-value">{fmtPct(current.global)}</span>
            <span className="replay-stat-sub">
              {round === 1
                ? "after the first round"
                : `${current.global >= first.global ? "+" : "−"}${Math.abs((current.global - first.global) * 100).toFixed(1)} pts since round 1`}
            </span>
          </div>
          <div className="replay-stat">
            <span className="replay-stat-label">Model weights exchanged</span>
            <span className="replay-stat-value">{weightsExchanged.toLocaleString("en-US")}</span>
            <span className="replay-stat-sub">
              {round} round{round === 1 ? "" : "s"} × {hospitalIds.length} hospitals × {nParams} weights, each way
            </span>
          </div>
          <div className="replay-stat">
            <span className="replay-stat-label">Patient records shared</span>
            <span className="replay-stat-value">0</span>
            <span className="replay-stat-sub">every record stays at its hospital</span>
          </div>
        </div>
      )}

      <div className="replay-layout">
        <div className="card">
          <h3>One training round</h3>
          <NetworkDiagram
            key={`${algorithm}-${round}`}
            hospitalIds={hospitalIds}
            accuracy={current}
            focus={focus}
            animate={round > 0}
          />
          {!compact && (
            <ol className="replay-steps">
              <li>
                <span className="replay-dot replay-dot-down" /> Server sends the shared model: <strong>{nParams} weights</strong>{" "}
                to each hospital
              </li>
              <li>
                Each hospital trains {summary.training.local_epochs} epochs on its own patients &mdash; the records never
                leave
              </li>
              <li>
                <span className="replay-dot replay-dot-up" /> Each sends back <strong>{nParams} updated weights</strong>{" "}
                and one number, its training-set size
              </li>
              <li>Server averages the weights, weighted by those sizes &rarr; next round&apos;s shared model</li>
            </ol>
          )}
        </div>

        <div className="card">
          <h3>Accuracy round by round</h3>
          <p className="muted-note">
            The shared model, scored on each hospital&apos;s own held-out patients after every round &middot; axis
            starts at {fmtPct(yMin, 0)}
          </p>
          <ResponsiveContainer width="100%" height={compact ? 190 : 280}>
            <LineChart data={visible} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                type="number"
                dataKey="round"
                domain={[1, nRounds]}
                ticks={[1, 5, 10, 15, 20].filter((t) => t <= nRounds)}
                tick={{ fontSize: 11 }}
                allowDataOverflow
              />
              <YAxis
                domain={[yMin, 1]}
                ticks={yTicks}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 11 }}
              />
              <Tooltip content={<RoundTooltip hospitalIds={hospitalIds} />} />
              <ReferenceLine x={round} stroke="var(--color-border)" />
              {hospitalIds.map((id) => (
                <Line
                  key={id}
                  dataKey={id}
                  type="linear"
                  stroke={HOSPITAL_COLORS[id]}
                  strokeWidth={2}
                  strokeOpacity={focus && focus !== id ? 0.15 : 1}
                  dot={headDot(HOSPITAL_COLORS[id])}
                  activeDot={{ r: 4 }}
                  isAnimationActive={false}
                />
              ))}
              <Line
                dataKey="global"
                type="linear"
                stroke={GLOBAL_COLOR}
                strokeWidth={3}
                strokeDasharray="6 3"
                strokeOpacity={focus && focus !== "global" ? 0.2 : 1}
                dot={headDot("#1a202c")}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>

          {compact ? (
            <div className="replay-chips">
              {[...hospitalIds, "global"].map((id) => (
                <span key={id} className="replay-chip">
                  <span
                    className={"replay-swatch" + (id === "global" ? " dashed" : "")}
                    style={{ background: id === "global" ? undefined : HOSPITAL_COLORS[id] }}
                  />
                  {id === "global" ? "Overall" : `H${hospitalIds.indexOf(id) + 1}`} <strong>{fmtPct(current[id], 0)}</strong>
                </span>
              ))}
            </div>
          ) : (
            <table className="replay-legend">
              <thead>
                <tr>
                  <th>Line</th>
                  <th>Round {round}</th>
                  <th>Trains on</th>
                </tr>
              </thead>
              <tbody>
                {[...hospitalIds, "global"].map((id) => (
                  <tr
                    key={id}
                    onMouseEnter={() => setFocus(id)}
                    onMouseLeave={() => setFocus(null)}
                    className={focus === id ? "focused" : undefined}
                  >
                    <td>
                      <span
                        className={"replay-swatch" + (id === "global" ? " dashed" : "")}
                        style={{ background: id === "global" ? undefined : HOSPITAL_COLORS[id] }}
                      />
                      {id === "global" ? "Overall (test-size weighted)" : hospitalLabel(id)}
                    </td>
                    <td>
                      <strong>{fmtPct(current[id])}</strong>
                    </td>
                    <td>{id === "global" ? "—" : `${nTrain[id]} patients`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted-note" style={{ marginTop: 10, marginBottom: 0 }}>
            Each hospital&apos;s test set is small (as few as {minTest} patients), so its line moves in steps of up to{" "}
            {(100 / minTest).toFixed(1)} points. {algorithm === "fedprox" && "FedProx's personalized models are then fine-tuned locally after round 20 — that step isn't shown here (see the Worst-served hospital tab)."}
          </p>
        </div>
      </div>
    </div>
  );
}

function NetworkDiagram({ hospitalIds, accuracy, focus, animate }) {
  return (
    <svg className="replay-diagram" viewBox="0 0 520 330" role="img" aria-label="Server exchanging model weights with 5 hospitals">
      {hospitalIds.map((id, i) => (
        <line
          key={`link-${id}`}
          x1={SERVER.x}
          y1={SERVER.y + 26}
          x2={HOSPITAL_X[i]}
          y2={HOSPITAL_Y - 30}
          className="replay-link"
        />
      ))}

      {animate &&
        hospitalIds.map((id, i) => {
          const dx = HOSPITAL_X[i] - SERVER.x;
          const dy = HOSPITAL_Y - 30 - (SERVER.y + 26);
          return (
            <g key={`packets-${id}`}>
              <circle
                className="replay-packet replay-packet-down"
                cx={SERVER.x}
                cy={SERVER.y + 26}
                r={6}
                style={{ "--dx": `${dx}px`, "--dy": `${dy}px` }}
              />
              <circle
                className="replay-packet replay-packet-up"
                cx={HOSPITAL_X[i]}
                cy={HOSPITAL_Y - 30}
                r={6}
                fill={HOSPITAL_COLORS[id]}
                style={{ "--dx": `${-dx}px`, "--dy": `${-dy}px` }}
              />
            </g>
          );
        })}

      <g>
        <rect x={SERVER.x - 90} y={SERVER.y - 26} width={180} height={52} rx={10} className="replay-server" />
        <text x={SERVER.x} y={SERVER.y - 4} textAnchor="middle" className="replay-node-title">
          Server
        </text>
        <text x={SERVER.x} y={SERVER.y + 14} textAnchor="middle" className="replay-node-sub">
          averages weights
        </text>
      </g>

      {hospitalIds.map((id, i) => (
        <g key={id} opacity={focus && focus !== id && focus !== "global" ? 0.3 : 1}>
          <rect
            x={HOSPITAL_X[i] - 44}
            y={HOSPITAL_Y - 30}
            width={88}
            height={60}
            rx={10}
            className="replay-hospital"
            style={{ stroke: HOSPITAL_COLORS[id] }}
          />
          <text x={HOSPITAL_X[i]} y={HOSPITAL_Y - 8} textAnchor="middle" className="replay-node-title">
            H{i + 1}
          </text>
          <text x={HOSPITAL_X[i]} y={HOSPITAL_Y + 16} textAnchor="middle" className="replay-node-value">
            {fmtPct(accuracy[id], 0)}
          </text>
          <text x={HOSPITAL_X[i]} y={HOSPITAL_Y + 50} textAnchor="middle" className="replay-node-sub">
            🔒 data stays
          </text>
        </g>
      ))}
    </svg>
  );
}

function RoundTooltip({ active, payload, hospitalIds }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>Round {row.round}</strong>
      {hospitalIds.map((id) => (
        <div key={id}>
          <span className="replay-swatch" style={{ background: HOSPITAL_COLORS[id] }} />
          {hospitalLabel(id)}: {fmtPct(row[id])}
        </div>
      ))}
      <div>
        <span className="replay-swatch dashed" />
        Overall: {fmtPct(row.global)}
      </div>
    </div>
  );
}
