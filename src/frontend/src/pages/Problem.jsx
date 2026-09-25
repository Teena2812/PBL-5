import { Link } from "react-router-dom";
import { getDashboardSummary, getEquityAnalysis, getExplainability, getHospitals } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import { fmtPct } from "../constants/experiments";
import { featureLabel } from "../constants/features";
import "./Problem.css";

async function getProblemData() {
  const [summary, hospitals, equity, explain] = await Promise.all([
    getDashboardSummary(),
    getHospitals(),
    getEquityAnalysis(),
    getExplainability(),
  ]);
  return { summary, hospitals: hospitals.hospitals, equity, explain };
}

export default function Problem() {
  const { data, loading, error } = useApiData(getProblemData, []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;

  const { summary, hospitals, equity, explain } = data;
  const sizes = hospitals.map((h) => h.n_patients);
  const rates = hospitals.map((h) => h.disease_rate);
  const smallest = hospitals.reduce((a, b) => (b.n_patients < a.n_patients ? b : a));
  const improved = equity.per_seed.filter((r) => r.delta > 0).length;
  const topDriver = featureLabel(explain.rf_global_importance[0].feature);

  // Each barrier follows from the one before it; the evidence line ties it
  // to this project's own data.
  const barriers = [
    {
      title: "Privacy laws",
      text: "HIPAA (US) and GDPR (EU) restrict moving patient records between institutions.",
      evidence: `${summary.dataset.n_patients} patients, but no hospital may send its records to another`,
    },
    {
      title: "Data silos",
      text: "Each hospital can only learn from its own patients, so small sites end up with weak, overfitted models.",
      evidence: `Hospitals hold ${Math.min(...sizes)}–${Math.max(...sizes)} patients; the smallest trains on just ${smallest.n_train}`,
    },
    {
      title: "Non-IID heterogeneity",
      text: "Patient mix differs by site, so one shared model averages conflicting signals and fits no hospital well.",
      evidence: `Disease rate ranges ${fmtPct(Math.min(...rates))}–${fmtPct(Math.max(...rates))} across the ${hospitals.length} hospitals`,
    },
    {
      title: "Clinician mistrust",
      text: "A risk score with no reason behind it can't be checked, so clinicians won't act on it.",
      evidence: `Every prediction here is explained; top driver overall: ${topDriver}`,
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h1>Problem</h1>
        <p>Why hospitals can&apos;t simply train one heart-disease model together &mdash; four barriers, each feeding the next</p>
      </div>

      <div className="card">
        <h3>The barrier chain</h3>
        <p className="muted-note">Read left to right: each barrier is a consequence of the one before it</p>

        <ol className="flow" aria-label="Problem flowchart">
          {barriers.map((b, i) => (
            <li key={b.title} className="flow-step">
              <div className="flow-node">
                <span className="flow-index">{i + 1}</span>
                <h4>{b.title}</h4>
                <p>{b.text}</p>
                <div className="flow-evidence">
                  <span className="flow-evidence-label">In this project</span>
                  {b.evidence}
                </div>
              </div>
              {i < barriers.length - 1 && <span className="flow-arrow" aria-hidden="true" />}
            </li>
          ))}
        </ol>

        <div className="flow-responses">
          <div className="flow-response" style={{ gridColumn: "1 / span 2" }}>
            <span className="flow-response-label">Answered by &middot; Federated learning</span>
            <p>Hospitals train together by sharing model weights only &mdash; raw records never leave a site.</p>
            <Link to="/hospitals">See the {hospitals.length} simulated hospitals →</Link>
          </div>
          <div className="flow-response" style={{ gridColumn: "3" }}>
            <span className="flow-response-label">Answered by &middot; Personalization</span>
            <p>
              FedProx + local fine-tuning adapts the shared model to each hospital. The worst-served hospital improved
              in {improved}/{equity.per_seed.length} seeds.
            </p>
            <Link to="/experiments">See the equity result →</Link>
          </div>
          <div className="flow-response" style={{ gridColumn: "4" }}>
            <span className="flow-response-label">Answered by &middot; Explainability</span>
            <p>SHAP shows which clinical features pushed each individual prediction up or down.</p>
            <Link to="/explainability">See explained predictions →</Link>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>The project&apos;s question</h3>
        <p style={{ margin: 0, fontSize: "0.92rem", lineHeight: 1.6 }}>
          Can {hospitals.length} hospitals with very different patient populations collaboratively train a heart-disease
          risk model that stays private, serves each hospital well &mdash; especially the worst-off one &mdash; and
          explains every prediction? This is a research simulation: non-IID partitions of the public{" "}
          {summary.dataset.name} dataset, not a real multi-hospital deployment.
        </p>
      </div>
    </div>
  );
}
