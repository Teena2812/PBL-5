import { Link } from "react-router-dom";
import {
  getDashboardSummary,
  getEquityAnalysis,
  getHospitals,
  getModelWeights,
  getSamplePatients,
} from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import StatCard from "../components/StatCard";
import PrivacyBanner from "../components/PrivacyBanner";
import WorstServedChart from "../components/WorstServedChart";
import {
  EXPERIMENT_COLORS,
  EXPERIMENT_LABELS,
  EXPERIMENT_ORDER,
  MULTISEED_NAME_TO_KEY,
  fmtPct,
  fmtProb,
  hospitalLabel,
} from "../constants/experiments";
import { predict, verifyAgainstBackend } from "../lib/inference";
import "./Overview.css";

async function getOverviewData() {
  const [summary, hospitals, equity, samples, weights] = await Promise.all([
    getDashboardSummary(),
    getHospitals(),
    getEquityAnalysis(),
    getSamplePatients(),
    getModelWeights(),
  ]);
  return { summary, hospitals: hospitals.hospitals, equity, samples: samples.samples, weights };
}

function fmtPts(delta) {
  const pts = delta * 100;
  return `${pts >= 0 ? "+" : "−"}${Math.abs(pts).toFixed(1)} pp`;
}

export default function Overview() {
  const { data, loading, error } = useApiData(getOverviewData, []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;

  const { summary, hospitals, equity } = data;
  const seeds = equity.per_seed;
  const improved = seeds.filter((r) => r.delta > 0);
  const worse = seeds.filter((r) => r.delta < 0);
  const flat = seeds.filter((r) => r.delta === 0);
  const mean = (list) => list.reduce((sum, r) => sum + r.delta, 0) / list.length;

  const sizes = hospitals.map((h) => h.n_patients);
  const rates = hospitals.map((h) => h.disease_rate);
  const personalized = hospitals.map((h) => h.personalized_accuracy);

  return (
    <div>
      <div className="page-header">
        <h1>Overview</h1>
        <p>Personalized federated learning for heart-disease risk &middot; the headline result first, then the setup behind it</p>
      </div>

      <section className="hero card" aria-labelledby="hero-title">
        <span className="hero-eyebrow">Headline finding</span>
        <h2 id="hero-title" className="hero-title">
          Personalization improved the worst-served hospital in{" "}
          <span className="hero-number">
            {improved.length}/{seeds.length}
          </span>{" "}
          seeds &mdash; by <span className="hero-number">{fmtPts(mean(improved))}</span> on average in those{" "}
          {improved.length}
        </h2>
        <p className="hero-sub">
          {fmtPts(mean(seeds))} averaged across all {seeds.length} seeds
          {flat.length > 0 && `, including ${flat.length === 1 ? "the one" : `the ${flat.length}`} with no change`}.{" "}
          {worse.length === 0 ? "It never made that hospital worse." : `It made that hospital worse in ${worse.length}.`}{" "}
          &ldquo;Worst-served&rdquo; = the hospital the shared FedAvg model scored lowest, re-identified in each seed.
        </p>

        <WorstServedChart seeds={seeds} />

        <div className="hero-footer">
          <Link to="/experiments">Full equity analysis →</Link>
          <span className="muted-note" style={{ margin: 0 }}>
            Why this metric: an overall accuracy weights hospitals by patient count, so it can hide how the
            worst-served hospital fares.
          </span>
        </div>
      </section>

      <KnownLimitation data={data} />

      <h3 className="section-heading">The setup behind this result</h3>
      <PrivacyBanner text={summary.privacy_statement} />

      <div className="grid grid-stats" style={{ marginTop: 16 }}>
        <StatCard
          label="Simulated hospitals"
          value={summary.n_hospitals}
          sublabel={`non-IID split of one public dataset`}
          accent="primary"
        />
        <StatCard
          label="Patients in total"
          value={summary.dataset.n_patients}
          sublabel={`${Math.min(...sizes)}–${Math.max(...sizes)} per hospital`}
          accent="teal"
        />
        <StatCard
          label="Disease rate by hospital"
          value={`${fmtPct(Math.min(...rates))}–${fmtPct(Math.max(...rates))}`}
          sublabel="why one shared model struggles"
          accent="warning"
        />
        <StatCard
          label="Federated rounds"
          value={summary.training.n_rounds}
          sublabel={`${summary.training.local_epochs} local epochs each`}
          accent="primary"
        />
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <OverallAccuracy equity={equity} />

        <div className="card">
          <h3>Per-hospital personalized accuracy</h3>
          <p className="muted-note">
            Ranges from {fmtPct(Math.min(...personalized))} to {fmtPct(Math.max(...personalized))} &middot; each
            hospital&apos;s own small test set, so one patient moves these by several points
          </p>
          <table>
            <thead>
              <tr>
                <th>Hospital</th>
                <th>Patients</th>
                <th>Test set</th>
                <th>Disease rate</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {hospitals.map((h) => (
                <tr key={h.hospital}>
                  <td>{hospitalLabel(h.hospital)}</td>
                  <td>{h.n_patients}</td>
                  <td>{h.n_test}</td>
                  <td>{fmtPct(h.disease_rate)}</td>
                  <td>{fmtPct(h.personalized_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Model</h3>
        <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--color-text-muted)" }}>
          {summary.model_architecture.class} &middot; hidden layers {summary.model_architecture.hidden_sizes.join(" → ")}{" "}
          &middot; FedProx proximal &mu; = {summary.training.proximal_mu} &middot; {summary.training.fine_tune_epochs}{" "}
          local fine-tuning epochs per hospital &middot; identical architecture in every setting compared
        </p>
      </div>
    </div>
  );
}

/* Secondary callout: the one missed sample patient, re-scored live by every
   hospital's in-browser model. Hidden if there's no miss or the in-browser
   models fail their backend check. */
function KnownLimitation({ data }) {
  const { samples, weights, hospitals } = data;
  const missed = samples.filter((s) => s.predicted_label !== s.actual_label);
  if (missed.length !== 1 || !verifyAgainstBackend(weights).ok) return null;

  const s = missed[0];
  const others = hospitals
    .map((h) => h.hospital)
    .filter((id) => id !== s.hospital_id)
    .map((id) => predict(s.patient, id, weights));
  const own = hospitals.find((h) => h.hospital === s.hospital_id);
  const lowestRate = Math.min(...hospitals.map((h) => h.disease_rate));
  const ownIsLowest = own.disease_rate === lowestRate;

  return (
    <aside className="limitation" aria-label="Known limitation">
      <div className="limitation-head">
        <span className="limitation-label">Known limitation</span>
        <span className="badge badge-neutral">Single-patient observation</span>
      </div>
      <p>
        Of the {samples.length} sample patients, {hospitalLabel(s.hospital_id)}&apos;s model misses one: a patient who had
        heart disease, scored {fmtProb(s.predicted_probability)} (just under the 50% threshold). The other{" "}
        {others.length} hospitals&apos; models score the same patient {fmtProb(Math.min(...others))}–
        {fmtProb(Math.max(...others))}.{" "}
        {ownIsLowest
          ? `${hospitalLabel(s.hospital_id)} has the lowest disease rate of the ${hospitals.length} (${fmtPct(own.disease_rate)}), so personalizing to it may pull risk estimates down for patients unlike its usual case mix.`
          : `${hospitalLabel(s.hospital_id)}'s disease rate is ${fmtPct(own.disease_rate)}.`}{" "}
        This is one patient, not a measured effect &mdash; but it shows personalization can cut both ways.
      </p>
      <Link to={`/explainability?view=try&sample=${s.id}`}>See this patient across all 5 models →</Link>
    </aside>
  );
}

/* Overall accuracy is a close race; say so instead of a flat bar chart. */
function OverallAccuracy({ equity }) {
  const rows = equity.phase5_multiseed_summary
    .map((r) => ({ ...r, key: MULTISEED_NAME_TO_KEY[r.experiment] }))
    .filter((r) => r.key)
    .sort((a, b) => EXPERIMENT_ORDER.indexOf(a.key) - EXPERIMENT_ORDER.indexOf(b.key));
  const means = rows.map((r) => r.mean_accuracy);
  const spread = Math.max(...means) - Math.min(...means);
  const nSeeds = rows[0]?.n_seeds;

  return (
    <div className="card">
      <h3>Overall accuracy: a close race</h3>
      <p className="muted-note">
        All {rows.length} settings land within {(spread * 100).toFixed(1)} points of each other (mean over {nSeeds}{" "}
        seeds) &mdash; which is why the headline uses the worst-served hospital instead
      </p>
      <table>
        <thead>
          <tr>
            <th>Setting</th>
            <th>Mean ± std</th>
            <th>Range</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>
                <span className="setting-swatch" style={{ background: EXPERIMENT_COLORS[r.key] }} />
                {EXPERIMENT_LABELS[r.key]}
              </td>
              <td>
                {fmtPct(r.mean_accuracy)} <span className="muted-inline">± {fmtPct(r.std_accuracy)}</span>
              </td>
              <td className="muted-inline">
                {fmtPct(r.min_accuracy)}–{fmtPct(r.max_accuracy)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Link to="/experiments?view=multiseed" className="card-link">
        See the 5-seed comparison →
      </Link>
    </div>
  );
}
