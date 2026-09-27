import { Link } from "react-router-dom";
import { getDashboardSummary, getEquityAnalysis, getExperimentComparison, getHospitals } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import StatCard from "../components/StatCard";
import PrivacyBanner from "../components/PrivacyBanner";
import SeedComparisonChart from "../components/SeedComparisonChart";
import {
  EXPERIMENT_COLORS,
  EXPERIMENT_LABELS,
  EXPERIMENT_ORDER,
  MULTISEED_NAME_TO_KEY,
  fmtPct,
  hospitalLabel,
  hospitalsWhereAloneWins,
} from "../constants/experiments";
import "./Overview.css";

async function getOverviewData() {
  const [summary, hospitals, equity, comparison] = await Promise.all([
    getDashboardSummary(),
    getHospitals(),
    getEquityAnalysis(),
    getExperimentComparison(),
  ]);
  return { summary, hospitals: hospitals.hospitals, equity, comparison };
}

export default function Overview() {
  const { data, loading, error } = useApiData(getOverviewData, []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;

  const { summary, hospitals, equity, comparison } = data;
  const aloneWins = hospitalsWhereAloneWins(comparison);
  const h = equity.headline;
  const sizes = hospitals.map((x) => x.n_patients);
  const rates = hospitals.map((x) => x.disease_rate);

  return (
    <div>
      <div className="page-header">
        <h1>Overview</h1>
        <p>
          Federated learning across {summary.n_hospitals} real hospitals&apos; heart-disease data &middot; the
          headline result first, then the setup behind it
        </p>
      </div>

      <section className="hero card" aria-labelledby="hero-title">
        <span className="hero-eyebrow">Headline finding</span>
        {/* Same sentence as equity_analysis.json's headline.text, with the numbers emphasised. */}
        <h2 id="hero-title" className="hero-title" data-headline={h.text}>
          Federated training beats hospitals training alone in{" "}
          <span className="hero-number">
            {h.seeds_fedavg_beats_local}/{h.n_seeds} seeds
          </span>{" "}
          (<span className="hero-number">{fmtPct(h.fedavg_mean_accuracy)}</span> vs{" "}
          {fmtPct(h.local_mean_accuracy)} overall accuracy, <span className="hero-number">+{h.delta_pp}pp</span>{" "}
          average) and matches or beats full data pooling in{" "}
          <span className="hero-number">
            {h.seeds_fedavg_matches_or_beats_centralized}/{h.n_seeds} seeds
          </span>{" "}
          &mdash; without sharing patient records.
        </h2>
        <p className="hero-sub">
          Same model, same data, five different random starting points (model-initialization seeds). Pooling every
          hospital&apos;s records in one place &mdash; which privacy rules forbid &mdash; averaged{" "}
          {fmtPct(h.centralized_mean_accuracy)}.
        </p>

        <SeedComparisonChart seeds={h.per_seed} />

        <div className="hero-footer">
          <Link to="/experiments">Full comparison →</Link>
          <span className="muted-note" style={{ margin: 0 }}>
            Overall accuracy = correct predictions across all {hospitals.reduce((s, x) => s + x.n_test, 0)} held-out
            test patients of the four hospitals.
            {aloneWins.length > 0 &&
              ` Not at every hospital: in the seed-42 run, ${aloneWins
                .map((x) => `${hospitalLabel(x.hospital)}'s own model scores ${fmtPct(x.local)} vs federated ${fmtPct(x.fedavg)}`)
                .join("; ")} (see below for why that accuracy says little).`}
          </span>
        </div>
      </section>

      <NaiveAccuracyStory hospitals={hospitals} />
      <PersonalizationResult equity={equity} comparison={comparison} />

      <h3 className="section-heading">The setup behind this result</h3>
      <PrivacyBanner text={summary.privacy_statement} />

      <div className="grid grid-stats" style={{ marginTop: 16 }}>
        <StatCard label="Hospitals" value={summary.n_hospitals} sublabel={summary.dataset.sites} accent="primary" />
        <StatCard
          label="Patients in total"
          value={summary.dataset.n_patients}
          sublabel={`${Math.min(...sizes)}–${Math.max(...sizes)} per hospital`}
          accent="teal"
        />
        <StatCard
          label="Disease rate by hospital"
          value={`${fmtPct(Math.min(...rates))}–${fmtPct(Math.max(...rates))}`}
          sublabel="real differences between the sites"
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
          <h3>Per-hospital personalized model</h3>
          <p className="muted-note">
            Accuracy alone can mislead on small, imbalanced test sets &mdash; compare it with always predicting the
            majority class, and with balanced accuracy and AUC (0.5 = chance)
          </p>
          <table>
            <thead>
              <tr>
                <th>Hospital</th>
                <th>Test (disease : healthy)</th>
                <th>Accuracy</th>
                <th>Majority-class</th>
                <th>Balanced acc.</th>
                <th>AUC</th>
              </tr>
            </thead>
            <tbody>
              {hospitals.map((x) => (
                <tr key={x.hospital}>
                  <td>{hospitalLabel(x.hospital)}</td>
                  <td>
                    {x.test_disease} : {x.test_no_disease}
                  </td>
                  <td>{fmtPct(x.personalized_accuracy)}</td>
                  <td className="muted-inline">{fmtPct(x.majority_class_accuracy)}</td>
                  <td>{fmtPct(x.personalized_balanced_accuracy)}</td>
                  <td>{x.personalized_auc.toFixed(3)}</td>
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
          &middot; {summary.dataset.n_clinical_features} clinical features ({summary.dataset.n_features} model inputs
          after one-hot encoding) &middot; FedProx proximal &mu; ={" "}
          {summary.training.proximal_mu} &middot; {summary.training.fine_tune_epochs} local fine-tuning epochs per
          hospital &middot; identical architecture in every setting compared
        </p>
      </div>
    </div>
  );
}

/* Secondary story: Switzerland's near-perfect-looking accuracy is class
   imbalance, not skill. Every number from hospitals.json. */
function NaiveAccuracyStory({ hospitals }) {
  const s = hospitals.find((x) => x.hospital === "switzerland");
  if (!s) return null;
  return (
    <aside className="story card" aria-labelledby="story-title">
      <span className="story-eyebrow">What naive accuracy reporting misses</span>
      <h3 id="story-title" className="story-title">
        {hospitalLabel(s.hospital)} looks nearly perfect: {fmtPct(s.personalized_accuracy)} accuracy,{" "}
        {fmtPct(s.personalized_recall, 0)} recall. Its AUC is {s.personalized_auc.toFixed(3)} &mdash; close to chance.
      </h3>
      <div className="story-stats">
        <div>
          <span className="story-value">{fmtPct(s.personalized_accuracy)}</span>
          <span className="story-label">accuracy</span>
        </div>
        <div>
          <span className="story-value">{fmtPct(s.majority_class_accuracy)}</span>
          <span className="story-label">always saying &ldquo;disease&rdquo;</span>
        </div>
        <div>
          <span className="story-value story-bad">{s.personalized_auc.toFixed(3)}</span>
          <span className="story-label">AUC (0.5 = chance)</span>
        </div>
        <div>
          <span className="story-value story-bad">{fmtPct(s.personalized_specificity, 0)}</span>
          <span className="story-label">
            healthy patients recognised ({Math.round(s.personalized_specificity * s.test_no_disease)} of{" "}
            {s.test_no_disease})
          </span>
        </div>
      </div>
      <p>
        Its test set holds {s.test_disease} patients with heart disease and {s.test_no_disease} without. Predicting
        &ldquo;disease&rdquo; for everyone scores {fmtPct(s.majority_class_accuracy)} &mdash; the same as the model.
        The high accuracy reflects the class imbalance, not the model telling patients apart: balanced accuracy is{" "}
        {fmtPct(s.personalized_balanced_accuracy)}. That is why this dashboard reports AUC and balanced accuracy next
        to accuracy.
      </p>
      <Link to="/experiments?view=per-hospital">See every model at every hospital →</Link>
    </aside>
  );
}

/* Personalization's honest, mixed result. */
function PersonalizationResult({ equity, comparison }) {
  const pers = equity.phase5_multiseed_summary.find((r) => MULTISEED_NAME_TO_KEY[r.experiment] === "personalized");
  const fedavg = equity.phase5_multiseed_summary.find((r) => MULTISEED_NAME_TO_KEY[r.experiment] === "fedavg");
  const ws = equity.worst_served;
  const worstCounts = ws.per_seed.reduce(
    (acc, r) => ({ ...acc, [r.worst_hospital]: (acc[r.worst_hospital] || 0) + 1 }),
    {}
  );
  const [worstHospital, worstN] = Object.entries(worstCounts).sort((a, b) => b[1] - a[1])[0];

  // Seed-42 single run: extra correct test patients per hospital, personalized vs FedAvg.
  const correct = (rows) => Object.fromEntries(rows.map((r) => [r.hospital, Math.round(r.accuracy * r.n_test)]));
  const p = correct(comparison.per_hospital.personalized);
  const f = correct(comparison.per_hospital.fedavg);
  const perHospital = Object.keys(p).map((k) => ({ hospital: k, delta: p[k] - f[k] }));
  const net = perHospital.reduce((s, x) => s + x.delta, 0);
  const swiss = perHospital.find((x) => x.hospital === "switzerland");
  const others = perHospital.filter((x) => x.hospital !== "switzerland" && x.delta !== 0);

  return (
    <aside className="limitation" aria-label="Personalization result">
      <div className="limitation-head">
        <span className="limitation-label">Personalization: a mixed result</span>
      </div>
      <p>
        Fine-tuning the federated model at each hospital raises the 5-seed overall accuracy from{" "}
        {fmtPct(fedavg.mean_accuracy)} to {fmtPct(pers.mean_accuracy)}. But in the seed-42 run,{" "}
        {swiss && net > 0
          ? `it classifies a net ${net} more test patients correctly than FedAvg, and Switzerland alone contributes +${swiss.delta}${
              others.length ? ` (${others.map((x) => `${hospitalLabel(x.hospital)} ${x.delta > 0 ? "+" : ""}${x.delta}`).join(", ")})` : ""
            } — and Switzerland's personalized model predicts "disease" for everyone`
          : "the gain is spread unevenly across hospitals"}
        . At the hospital federated training serves worst ({hospitalLabel(worstHospital)}, in {worstN}/
        {ws.per_seed.length} seeds), personalization helped in {ws.seeds_improved}, changed nothing in {ws.seeds_flat}{" "}
        and hurt in {ws.seeds_worse} (mean {ws.mean_delta_pp > 0 ? "+" : ""}
        {ws.mean_delta_pp} pp) &mdash; one or two patients either way.
      </p>
      <Link to="/experiments?view=worst-served">See the per-seed breakdown →</Link>
    </aside>
  );
}

/* Overall accuracy of all five settings. */
function OverallAccuracy({ equity }) {
  const rows = equity.phase5_multiseed_summary
    .map((r) => ({ ...r, key: MULTISEED_NAME_TO_KEY[r.experiment] }))
    .filter((r) => r.key)
    .sort((a, b) => EXPERIMENT_ORDER.indexOf(a.key) - EXPERIMENT_ORDER.indexOf(b.key));
  const nSeeds = rows[0]?.n_seeds;

  return (
    <div className="card">
      <h3>Overall accuracy, all five settings</h3>
      <p className="muted-note">Mean ± std over {nSeeds} seeds &middot; identical model in every setting</p>
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
