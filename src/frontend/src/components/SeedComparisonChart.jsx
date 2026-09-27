import { EXPERIMENT_COLORS, EXPERIMENT_LABELS, fmtPct } from "../constants/experiments";
import "./WorstServedChart.css";

/**
 * One row per model-init seed: training alone (Local NN) -> federated
 * (FedAvg), with the pooled-data result (Centralized NN) marked. Dots on a
 * labelled, zoomed axis (dots, not bars, so zooming stays honest). Reuses
 * the dumbbell styles from WorstServedChart.
 *
 * seeds: phase4_multiseed_results rows {seed, local_nn_accuracy,
 * fedavg_nn_accuracy, centralized_nn_accuracy}.
 */
export default function SeedComparisonChart({ seeds }) {
  const values = seeds.flatMap((r) => [r.local_nn_accuracy, r.fedavg_nn_accuracy, r.centralized_nn_accuracy]);
  const lo = Math.floor(Math.min(...values) * 50) / 50 - 0.02;
  const hi = Math.min(1, Math.ceil(Math.max(...values) * 50) / 50 + 0.02);
  const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const ticks = [];
  for (let t = Math.ceil(lo * 50); t <= Math.floor(hi * 50); t++) ticks.push(t / 50);

  const marker = (key, v) => (
    <span
      key={key}
      className={"dumbbell-dot" + (key === "centralized" ? " dumbbell-dot-ring" : "")}
      style={{ left: pos(v), background: key === "centralized" ? "var(--color-surface)" : EXPERIMENT_COLORS[key], borderColor: EXPERIMENT_COLORS[key] }}
      title={`${EXPERIMENT_LABELS[key]} ${fmtPct(v)}`}
    />
  );

  return (
    <div className="dumbbell" role="table" aria-label="Overall accuracy per seed: training alone, federated, pooled">
      <div className="dumbbell-legend" aria-hidden="true">
        <span>
          <span className="dumbbell-key" style={{ background: EXPERIMENT_COLORS.local }} /> Each hospital alone
        </span>
        <span>
          <span className="dumbbell-key" style={{ background: EXPERIMENT_COLORS.fedavg }} /> Federated (FedAvg)
        </span>
        <span>
          <span className="dumbbell-key dumbbell-key-ring" style={{ borderColor: EXPERIMENT_COLORS.centralized }} /> All data
          pooled (not private)
        </span>
        <span className="dumbbell-axis-note">
          overall accuracy &middot; axis {fmtPct(lo, 0)}–{fmtPct(hi, 0)}
        </span>
      </div>

      {seeds.map((r) => {
        const from = Math.min(r.local_nn_accuracy, r.fedavg_nn_accuracy);
        const to = Math.max(r.local_nn_accuracy, r.fedavg_nn_accuracy);
        const gain = (r.fedavg_nn_accuracy - r.local_nn_accuracy) * 100;
        return (
          <div className="dumbbell-row" role="row" key={r.seed}>
            <span className="dumbbell-label" role="rowheader">
              <strong>Seed {r.seed}</strong>
              <span>model init</span>
            </span>
            <span className="dumbbell-track" role="cell">
              {ticks.map((t) => (
                <span key={t} className="dumbbell-grid" style={{ left: pos(t) }} />
              ))}
              <span className="dumbbell-bar" style={{ left: pos(from), width: `calc(${pos(to)} - ${pos(from)})` }} />
              {marker("centralized", r.centralized_nn_accuracy)}
              {marker("local", r.local_nn_accuracy)}
              {marker("fedavg", r.fedavg_nn_accuracy)}
            </span>
            <span className="dumbbell-values" role="cell">
              {fmtPct(r.local_nn_accuracy)} → <strong>{fmtPct(r.fedavg_nn_accuracy)}</strong>
              <span className="dumbbell-sub"> pooled {fmtPct(r.centralized_nn_accuracy)}</span>
            </span>
            <span role="cell">
              <span className={`badge ${gain > 0 ? "badge-low" : gain < 0 ? "badge-high" : "badge-neutral"}`}>
                {gain > 0 ? "▲ +" : gain < 0 ? "▼ −" : "= "}
                {Math.abs(gain).toFixed(1)} pp
              </span>
            </span>
          </div>
        );
      })}

      <div className="dumbbell-row dumbbell-ticks" aria-hidden="true">
        <span />
        <span className="dumbbell-track dumbbell-track-bare">
          {ticks.map((t) => (
            <span key={t} className="dumbbell-tick" style={{ left: pos(t) }}>
              {fmtPct(t, 0)}
            </span>
          ))}
        </span>
        <span />
        <span />
      </div>
    </div>
  );
}
