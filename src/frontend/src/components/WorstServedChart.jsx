import { EXPERIMENT_COLORS, fmtPct, hospitalLabel } from "../constants/experiments";
import "./WorstServedChart.css";

function fmtPts(delta) {
  const pts = delta * 100;
  return `${pts >= 0 ? "+" : "−"}${Math.abs(pts).toFixed(1)} pp`;
}

/* Before/after per seed: FedAvg dot -> Personalized dot for the worst-served
   hospital. Dots (not bars), so a zoomed, labelled axis is honest. */
export default function WorstServedChart({ seeds }) {
  const values = seeds.flatMap((r) => [r.fedavg_accuracy, r.personalized_accuracy]);
  const lo = Math.floor(Math.min(...values) * 20) / 20 - 0.05;
  const hi = Math.min(1, Math.ceil(Math.max(...values) * 20) / 20 + 0.05);
  const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const ticks = [];
  for (let t = Math.round(lo * 20); t <= Math.round(hi * 20); t++) ticks.push(t / 20);

  return (
    <div className="dumbbell" role="table" aria-label="Worst-served hospital accuracy per seed, before and after personalization">
      <div className="dumbbell-legend" aria-hidden="true">
        <span>
          <span className="dumbbell-key" style={{ background: EXPERIMENT_COLORS.fedavg }} /> FedAvg (shared model)
        </span>
        <span>
          <span className="dumbbell-key" style={{ background: EXPERIMENT_COLORS.personalized }} /> Personalized
          (FedProx + local fine-tuning)
        </span>
        <span className="dumbbell-axis-note">
          axis {fmtPct(lo, 0)}–{fmtPct(hi, 0)}
        </span>
      </div>

      {seeds.map((r) => {
        const from = Math.min(r.fedavg_accuracy, r.personalized_accuracy);
        const to = Math.max(r.fedavg_accuracy, r.personalized_accuracy);
        return (
          <div className="dumbbell-row" role="row" key={r.seed}>
            <span className="dumbbell-label" role="rowheader">
              <strong>Seed {r.seed}</strong>
              <span>{hospitalLabel(r.worst_hospital)}</span>
            </span>
            <span className="dumbbell-track" role="cell">
              {ticks.map((t) => (
                <span key={t} className="dumbbell-grid" style={{ left: pos(t) }} />
              ))}
              <span className="dumbbell-bar" style={{ left: pos(from), width: `calc(${pos(to)} - ${pos(from)})` }} />
              <span
                className="dumbbell-dot"
                style={{ left: pos(r.fedavg_accuracy), background: EXPERIMENT_COLORS.fedavg }}
                title={`FedAvg ${fmtPct(r.fedavg_accuracy)}`}
              />
              <span
                className="dumbbell-dot"
                style={{ left: pos(r.personalized_accuracy), background: EXPERIMENT_COLORS.personalized }}
                title={`Personalized ${fmtPct(r.personalized_accuracy)}`}
              />
            </span>
            <span className="dumbbell-values" role="cell">
              {fmtPct(r.fedavg_accuracy)} → <strong>{fmtPct(r.personalized_accuracy)}</strong>
            </span>
            <span role="cell">
              <span className={`badge ${r.delta > 0 ? "badge-low" : r.delta < 0 ? "badge-high" : "badge-neutral"}`}>
                {r.delta > 0 ? `▲ ${fmtPts(r.delta)}` : r.delta < 0 ? `▼ ${fmtPts(r.delta)}` : "No change"}
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

