import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from "recharts";
import { getExplainability, getHospitals, getModelWeights, getSamplePatients } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import Tabs from "../components/Tabs";
import PredictionResult from "../components/PredictionResult";
import ShapChart from "../components/ShapChart";
import WhatIfExplorer from "../components/WhatIfExplorer";
import { verifyAgainstBackend } from "../lib/inference";
import { hospitalLabel, fmtProb } from "../constants/experiments";
import { PATIENT_FIELDS, featureLabel, normalizeFeature, patientFieldText } from "../constants/features";
import { useMediaQuery } from "../hooks/useMediaQuery";
import "./Explainability.css";

const TABS = [
  { id: "samples", label: "Sample patients" },
  { id: "try", label: "Try a prediction" },
  { id: "global", label: "Global importance" },
];

export default function Explainability() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requested = searchParams.get("view");
  const active = TABS.some((t) => t.id === requested) ? requested : "samples";
  const setActive = (id) => setSearchParams(id === "samples" ? {} : { view: id }, { replace: true });
  const openInExplorer = (sampleId) => setSearchParams({ view: "try", sample: sampleId });

  return (
    <div>
      <div className="page-header">
        <h1>Explainability</h1>
        <p>Why each hospital&apos;s personalized model predicts what it does &middot; SHAP feature contributions</p>
      </div>

      <Tabs tabs={TABS} active={active} onChange={setActive} label="Explainability views" />

      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {active === "samples" && <SamplesView onExplore={openInExplorer} />}
        {active === "try" && (
          <TryPredictionView key={searchParams.get("sample") ?? "default"} initialSampleId={searchParams.get("sample")} />
        )}
        {active === "global" && <GlobalView />}
      </div>

      <p className="disclaimer">
        Research prototype, not a clinical decision-making tool. Models were trained on small simulated partitions of
        the public UCI Heart Disease dataset and are not validated for patient care.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sample patients: precomputed predictions + SHAP (Colab export)      */
/* ------------------------------------------------------------------ */

function SamplesView({ onExplore }) {
  const samples = useApiData(getSamplePatients, []);
  const [selectedId, setSelectedId] = useState(null);

  if (samples.loading) return <Loading />;
  if (samples.error) return <ErrorState error={samples.error} />;

  const list = samples.data.samples;
  if (list.length === 0) return <SamplesMissing />;
  const selected = list.find((s) => s.id === selectedId) ?? list[0];

  return (
    <div className="samples-layout">
      <div className="sample-list" role="listbox" aria-label="Sample patients">
        {list.map((s) => (
          <button
            key={s.id}
            type="button"
            role="option"
            aria-selected={s.id === selected.id}
            className={"sample-item" + (s.id === selected.id ? " active" : "")}
            onClick={() => setSelectedId(s.id)}
          >
            <span className="sample-item-title">{hospitalLabel(s.hospital_id)} patient</span>
            <span className="sample-item-meta">
              {s.patient.age} yrs &middot; {s.patient.sex === 1 ? "male" : "female"} &middot;{" "}
              {fmtProb(s.predicted_probability, 0)} risk
            </span>
          </button>
        ))}
        <p className="muted-note" style={{ margin: "8px 4px 0" }}>
          One real held-out test patient per hospital, scored by that hospital&apos;s own model.
        </p>
      </div>

      <div>
        <div className="card">
          <PredictionResult result={selected} rawPatient={selected.patient} actualLabel={selected.actual_label} />
          <div className="explore-link">
            <button type="button" className="button-secondary" onClick={() => onExplore(selected.id)}>
              Explore this patient in the what-if explorer →
            </button>
            <span className="muted-note" style={{ margin: 0 }}>
              Change their values and watch the risk update live, for all 5 hospital models.
            </span>
          </div>
        </div>
        <div className="card">
          <h3>Patient record</h3>
          <PatientTable patient={selected.patient} />
        </div>
      </div>
    </div>
  );
}

function SamplesMissing() {
  return (
    <div className="info-box">
      <strong>Sample patients haven&apos;t been generated yet.</strong>
      <div style={{ marginTop: 6 }}>
        They need torch + SHAP, so they&apos;re precomputed on Colab: run section 11 of{" "}
        <code>notebooks/phase6_colab.ipynb</code> (<code>experiments/export_sample_patients.py</code>) and copy the
        downloaded <code>sample_patients.json</code> into <code>experiments/results/dashboard_data/</code>.
      </div>
      <div style={{ marginTop: 6 }}>
        Meanwhile, the <em>Global importance</em> tab shows the Phase 6 SHAP results, including one example patient.
      </div>
    </div>
  );
}

function PatientTable({ patient }) {
  return (
    <table className="patient-table">
      <tbody>
        {PATIENT_FIELDS.map((field) => (
          <tr key={field.key}>
            <th scope="row">{field.label}</th>
            <td>{patientFieldText(field, patient[field.key])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ------------------------------------------------------------------ */
/* Try a prediction: live POST /api/predict                            */
/* ------------------------------------------------------------------ */

async function getWhatIfContext() {
  const [weights, hospitals, samples] = await Promise.all([getModelWeights(), getHospitals(), getSamplePatients()]);
  return {
    weights,
    hospitals: hospitals.hospitals,
    samples: samples.samples,
    // Guard against the in-browser models drifting from the real ones.
    verification: verifyAgainstBackend(weights),
  };
}

function TryPredictionView({ initialSampleId }) {
  const context = useApiData(getWhatIfContext, []);
  if (context.loading) return <Loading />;
  if (context.error) return <ErrorState error={context.error} />;

  const { weights, hospitals, samples, verification } = context.data;
  if (!verification.ok) {
    return (
      <div className="error-box">
        <strong>In-browser predictions are switched off.</strong>
        <div style={{ marginTop: 6 }}>
          The exported model weights no longer reproduce the real backend&apos;s outputs (largest difference{" "}
          {verification.maxDiff.toExponential(2)}, allowed 5e-5). Re-run{" "}
          <code>experiments/export_model_weights.py</code> rather than trusting drifted predictions.
        </div>
      </div>
    );
  }

  return (
    <WhatIfExplorer
      weights={weights}
      hospitals={hospitals}
      samples={samples}
      verification={verification}
      initialSampleId={initialSampleId}
    />
  );
}

/* ------------------------------------------------------------------ */
/* Global importance: Phase 6 SHAP summaries                           */
/* ------------------------------------------------------------------ */

const TOP_N = 10;

function GlobalView() {
  const explain = useApiData(getExplainability, []);
  if (explain.loading) return <Loading />;
  if (explain.error) return <ErrorState error={explain.error} />;

  const d = explain.data;
  const rf = d.rf_global_importance.slice(0, TOP_N);
  const nn = d.nn_hospital_1_importance.slice(0, TOP_N);
  const sameTop = normalizeFeature(rf[0].feature) === normalizeFeature(nn[0].feature);

  return (
    <>
      <div className="headline-callout">
        <span className="headline-callout-label">What this shows</span>
        <span>
          Mean |SHAP| is how far a feature moves the predicted disease probability on average, in either direction.{" "}
          {sameTop
            ? `${featureLabel(rf[0].feature)} is the top feature for both the pooled Random Forest and Hospital 1's personalized model.`
            : `The pooled Random Forest and Hospital 1's personalized model rank different top features (${featureLabel(rf[0].feature)} vs ${featureLabel(nn[0].feature)}).`}
        </span>
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Random Forest (centralized)</h3>
          <p className="muted-note">TreeExplainer on the pooled test set &middot; top {TOP_N} of {d.rf_global_importance.length} features &middot; percentage points</p>
          <ImportanceChart rows={rf} />
        </div>
        <div className="card">
          <h3>Hospital 1 personalized model</h3>
          <p className="muted-note">KernelExplainer on Hospital 1&apos;s test set &middot; top {TOP_N} of {d.nn_hospital_1_importance.length} features &middot; percentage points</p>
          <ImportanceChart rows={nn} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Example patient (Phase 6, Random Forest)</h3>
        <p className="muted-note">
          One pooled-test-set patient&apos;s top {d.example_patient.length} contributions. Only the model&apos;s encoded
          inputs were saved for this patient, so numeric values are shown as standard deviations from the dataset mean.
        </p>
        <ShapChart contributions={d.example_patient} />
      </div>
    </>
  );
}

function ImportanceChart({ rows }) {
  const narrow = useMediaQuery("(max-width: 768px)");
  const data = rows.map((r) => ({ label: featureLabel(r.feature), value: r.mean_abs_shap }));
  return (
    <ResponsiveContainer width="100%" height={rows.length * 30 + 30}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, left: 4, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => (v * 100).toFixed(0)} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="label" width={narrow ? 130 : 220} tick={{ fontSize: narrow ? 10 : 12 }} interval={0} />
        <Tooltip
          cursor={{ fill: "rgba(148, 163, 184, 0.12)" }}
          formatter={(v) => [`${(v * 100).toFixed(1)} pts`, "Mean |SHAP|"]}
        />
        <Bar dataKey="value" fill="var(--color-primary)" radius={[0, 4, 4, 0]} barSize={16}>
          <LabelList
            dataKey="value"
            position="right"
            formatter={(v) => (v * 100).toFixed(1)}
            style={{ fontSize: 11, fill: "var(--color-text)" }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
