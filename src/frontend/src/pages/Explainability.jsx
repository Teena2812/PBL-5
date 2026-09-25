import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from "recharts";
import { getExplainability, getHealth, getHospitals, getSamplePatients, predict } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import Tabs from "../components/Tabs";
import PredictionResult from "../components/PredictionResult";
import ShapChart from "../components/ShapChart";
import { hospitalLabel, fmtPct } from "../constants/experiments";
import {
  PATIENT_FIELDS,
  DEFAULT_PATIENT,
  featureLabel,
  normalizeFeature,
  patientFieldText,
} from "../constants/features";
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

  return (
    <div>
      <div className="page-header">
        <h1>Explainability</h1>
        <p>Why each hospital&apos;s personalized model predicts what it does &middot; SHAP feature contributions</p>
      </div>

      <Tabs tabs={TABS} active={active} onChange={setActive} label="Explainability views" />

      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {active === "samples" && <SamplesView />}
        {active === "try" && <TryPredictionView />}
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

function SamplesView() {
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
              {fmtPct(s.predicted_probability, 0)} risk
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

async function getTryPredictionContext() {
  const [health, hospitals, samples] = await Promise.all([
    getHealth(),
    getHospitals(),
    // Samples are optional here (only used to prefill the form).
    getSamplePatients().catch(() => ({ samples: [] })),
  ]);
  return { health, hospitals: hospitals.hospitals, samples: samples.samples };
}

function TryPredictionView() {
  const context = useApiData(getTryPredictionContext, []);
  const [form, setForm] = useState({ ...DEFAULT_PATIENT, hospital_id: "hospital_1" });
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  if (context.loading) return <Loading />;
  if (context.error) return <ErrorState error={context.error} />;

  const { health, hospitals, samples } = context.data;
  const available = health.live_prediction_available;

  const setField = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const loadPreset = (value) => {
    if (value === "default") {
      setForm((f) => ({ ...DEFAULT_PATIENT, hospital_id: f.hospital_id }));
      return;
    }
    const sample = samples.find((s) => s.id === value);
    if (sample) setForm({ ...sample.patient, hospital_id: sample.hospital_id });
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setSubmitError(null);
    try {
      const payload = Object.fromEntries(Object.entries(form).map(([k, v]) => [k, k === "hospital_id" ? v : Number(v)]));
      setResult({ response: await predict(payload), patient: payload });
    } catch (err) {
      setResult(null);
      setSubmitError(err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      {!available && (
        <div className="warning-box" style={{ marginBottom: 20 }}>
          <strong>Live prediction isn&apos;t available on this backend.</strong>
          <div style={{ marginTop: 6 }}>
            {health.live_prediction_note} Run the backend where torch and SHAP are installed (e.g. Colab, section 10
            of the notebook) to use this form. The <em>Sample patients</em> tab shows predictions made with the same
            code.
          </div>
        </div>
      )}

      <div className="try-layout">
        <form className="card try-form" onSubmit={onSubmit}>
          <h3>Patient</h3>
          <div className="form-row">
            <label htmlFor="preset">Start from</label>
            <select id="preset" defaultValue="default" onChange={(e) => loadPreset(e.target.value)}>
              <option value="default">Example patient (API docs)</option>
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {hospitalLabel(s.hospital_id)} sample patient
                </option>
              ))}
            </select>
          </div>

          <div className="form-grid">
            {PATIENT_FIELDS.map((field) => (
              <div className="form-row" key={field.key}>
                <label htmlFor={`f-${field.key}`}>
                  {field.label}
                  {field.unit && <span className="form-unit"> ({field.unit})</span>}
                </label>
                {field.type === "select" ? (
                  <select
                    id={`f-${field.key}`}
                    value={form[field.key]}
                    onChange={(e) => setField(field.key, Number(e.target.value))}
                  >
                    {Object.entries(field.options).map(([value, text]) => (
                      <option key={value} value={value}>
                        {text}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={`f-${field.key}`}
                    type="number"
                    required
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={form[field.key]}
                    onChange={(e) => setField(field.key, e.target.value)}
                  />
                )}
              </div>
            ))}
          </div>

          <div className="form-row" style={{ marginTop: 8 }}>
            <label htmlFor="f-hospital">Hospital model</label>
            <select id="f-hospital" value={form.hospital_id} onChange={(e) => setField("hospital_id", e.target.value)}>
              {hospitals.map((h) => (
                <option key={h.hospital} value={h.hospital}>
                  {hospitalLabel(h.hospital)} personalized ({fmtPct(h.personalized_accuracy)} test accuracy)
                </option>
              ))}
            </select>
          </div>

          <button type="submit" className="button-primary" disabled={!available || submitting}>
            {submitting ? "Predicting…" : "Predict risk"}
          </button>
        </form>

        <div className="card">
          {submitError ? (
            <PredictErrorState error={submitError} />
          ) : result ? (
            <PredictionResult result={result.response} rawPatient={result.patient} />
          ) : (
            <p className="muted-note" style={{ margin: 0 }}>
              {available
                ? "Fill in the patient record and pick a hospital's model, then press Predict risk. The model only runs inference — nothing is retrained."
                : "The prediction and its SHAP explanation will appear here once a backend with torch is running."}
            </p>
          )}
        </div>
      </div>
    </>
  );
}

function PredictErrorState({ error }) {
  const detail = error?.response?.data?.detail;
  // FastAPI validation errors (422) come back as a list of field errors.
  if (Array.isArray(detail)) {
    return (
      <div className="error-box">
        <strong>Some inputs are out of range.</strong>
        <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
          {detail.map((d, i) => (
            <li key={i}>
              {d.loc?.slice(1).join(".")}: {d.msg}
            </li>
          ))}
        </ul>
      </div>
    );
  }
  if (typeof detail === "string") {
    return (
      <div className="error-box">
        <strong>Prediction failed.</strong>
        <div style={{ marginTop: 6 }}>{detail}</div>
      </div>
    );
  }
  return <ErrorState error={error} />;
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
  const data = rows.map((r) => ({ label: featureLabel(r.feature), value: r.mean_abs_shap }));
  return (
    <ResponsiveContainer width="100%" height={rows.length * 30 + 30}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, left: 4, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => (v * 100).toFixed(0)} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="label" width={220} tick={{ fontSize: 12 }} interval={0} />
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
