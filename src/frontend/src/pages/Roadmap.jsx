import {
  getDashboardSummary,
  getEquityAnalysis,
  getExplainability,
  getHospitals,
  getSamplePatients,
} from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import { MULTISEED_NAME_TO_KEY, fmtPct } from "../constants/experiments";
import { featureLabel } from "../constants/features";
import { PHASES } from "../constants/roadmap";
import "./Roadmap.css";

async function getRoadmapData() {
  const [summary, hospitals, equity, explain, samples] = await Promise.all([
    getDashboardSummary(),
    getHospitals(),
    getEquityAnalysis(),
    getExplainability(),
    getSamplePatients(),
  ]);
  return { summary, hospitals: hospitals.hospitals, equity, explain, samples: samples.samples };
}

// One headline outcome per phase, computed from the exported results.
// Phases 1 and 3 have no exported dashboard data, so they show none rather
// than a number from somewhere else.
function phaseResults({ summary, hospitals, equity, explain, samples }) {
  const rates = hospitals.map((h) => h.disease_rate);
  const multiseed = Object.fromEntries(
    equity.phase5_multiseed_summary.map((r) => [MULTISEED_NAME_TO_KEY[r.experiment], r])
  );
  const improved = equity.per_seed.filter((r) => r.delta > 0).length;
  const correct = samples.filter((s) => s.predicted_label === s.actual_label).length;

  return {
    1: `Dataset chosen: ${summary.dataset.name}`,
    2: `${summary.dataset.n_patients} patients → ${summary.n_hospitals} hospitals, disease rate ${fmtPct(Math.min(...rates))}–${fmtPct(Math.max(...rates))}`,
    3: null,
    4: `FedAvg ${fmtPct(multiseed.fedavg.mean_accuracy)} ± ${fmtPct(multiseed.fedavg.std_accuracy)} over ${multiseed.fedavg.n_seeds} seeds vs Local ${fmtPct(multiseed.local.mean_accuracy)}`,
    5: `Worst-served hospital improved in ${improved}/${equity.per_seed.length} seeds`,
    6: `Top driver: ${featureLabel(explain.rf_global_importance[0].feature)}`,
    7: samples.length
      ? `Live dashboard; ${samples.length} sample patients scored by the saved models, ${correct}/${samples.length} correct`
      : "Live dashboard over the exported results",
    8: null,
  };
}

export default function Roadmap() {
  const { data, loading, error } = useApiData(getRoadmapData, []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;

  const results = phaseResults(data);
  const nDone = PHASES.filter((p) => p.done).length;
  const next = PHASES.find((p) => !p.done);

  return (
    <div>
      <div className="page-header">
        <h1>Roadmap</h1>
        <p>The 8-phase project plan &middot; {nDone} of {PHASES.length} phases complete</p>
      </div>

      <div className="card roadmap-progress">
        <div className="roadmap-progress-bar" aria-hidden="true">
          {PHASES.map((p) => (
            <span key={p.n} className={"roadmap-progress-seg" + (p.done ? " done" : "")} />
          ))}
        </div>
        <div className="roadmap-progress-text">
          <strong>
            {nDone}/{PHASES.length} complete
          </strong>
          {next && (
            <span>
              {" "}
              &middot; Next: Phase {next.n} &mdash; {next.title}
            </span>
          )}
        </div>
      </div>

      <div className="card">
        <ol className="timeline">
          {PHASES.map((p) => (
            <li key={p.n} className={"timeline-item" + (p.done ? " done" : " pending")}>
              <span className="timeline-marker" aria-hidden="true">
                {p.done ? "✓" : p.n}
              </span>
              <div className="timeline-body">
                <div className="timeline-title">
                  <span className="timeline-phase">Phase {p.n}</span>
                  {p.title}
                  <span className={`badge ${p.done ? "badge-low" : "badge-neutral"}`}>
                    {p.done ? "Done" : "Upcoming"}
                  </span>
                </div>
                {results[p.n] && <div className="timeline-result">{results[p.n]}</div>}
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
