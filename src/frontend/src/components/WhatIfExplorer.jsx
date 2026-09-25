import { useMemo, useState } from "react";
import ShapChart from "./ShapChart";
import { PATIENT_FIELDS, DEFAULT_PATIENT } from "../constants/features";
import { fmtPct, fmtProb, hospitalLabel } from "../constants/experiments";
import { deviationBreakdown, predict, riskLevel } from "../lib/inference";
import "./WhatIfExplorer.css";

const RISK_BADGE = { LOW: "badge-low", MODERATE: "badge-moderate", HIGH: "badge-high" };
const THRESHOLD = 0.5;
const TOP_FIELDS = 8;

/**
 * Live what-if explorer: every change re-scores the patient in the browser
 * with the exported personalized models (lib/inference.js).
 *
 * weights: model_weights.json; hospitals: hospitals.json rows (for disease
 * rates); samples: precomputed sample patients (for "start from");
 * initialSampleId: optional sample to load first.
 */
export default function WhatIfExplorer({ weights, hospitals, samples, verification, initialSampleId }) {
  const initialSample = samples.find((s) => s.id === initialSampleId);
  const [patient, setPatient] = useState(initialSample ? { ...initialSample.patient } : { ...DEFAULT_PATIENT });
  const [hospitalId, setHospitalId] = useState(initialSample?.hospital_id ?? "hospital_1");
  const [preset, setPreset] = useState(initialSample ? initialSample.id : "default");

  const hospitalIds = Object.keys(weights.hospitals).sort();
  const ranges = weights.preprocessing.observed_range;
  const diseaseRate = Object.fromEntries(hospitals.map((h) => [h.hospital, h.disease_rate]));

  const result = useMemo(() => deviationBreakdown(patient, hospitalId, weights), [patient, hospitalId, weights]);
  const acrossHospitals = useMemo(
    () => hospitalIds.map((id) => ({ id, probability: predict(patient, id, weights) })),
    [patient, weights, hospitalIds]
  );

  const setField = (key, value) => {
    setPatient((p) => ({ ...p, [key]: value }));
    setPreset("custom");
  };

  const loadPreset = (value) => {
    setPreset(value);
    if (value === "default") setPatient({ ...DEFAULT_PATIENT });
    const sample = samples.find((s) => s.id === value);
    if (sample) {
      setPatient({ ...sample.patient });
      setHospitalId(sample.hospital_id);
    }
  };

  const probs = acrossHospitals.map((h) => h.probability);
  const minP = Math.min(...probs);
  const maxP = Math.max(...probs);
  const trainSizes = hospitalIds.map((id) => weights.hospitals[id].n_train);
  const rates = hospitalIds.map((id) => diseaseRate[id]);
  const selected = weights.hospitals[hospitalId];
  const risk = riskLevel(result.probability);

  return (
    <>
      <div className="verify-strip">
        <span className="verify-strip-icon" aria-hidden="true">✓</span>
        <span>
          <strong>Running in your browser</strong> &mdash; the {hospitalIds.length} saved personalized models (
          {weights.architecture.n_parameters} parameters each), no server. Checked on load against the real backend:{" "}
          {verification.cases.length}/{verification.cases.length} reference patients match (max difference{" "}
          {verification.maxDiff.toExponential(1)}).
        </span>
      </div>

      <div className="whatif-layout">
        <div className="card whatif-inputs">
          <h3>Patient</h3>
          <div className="form-row">
            <label htmlFor="wi-preset">Start from</label>
            <select id="wi-preset" value={preset} onChange={(e) => loadPreset(e.target.value)}>
              <option value="default">Example patient (API docs)</option>
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {hospitalLabel(s.hospital_id)} sample patient
                </option>
              ))}
              {preset === "custom" && <option value="custom">Edited patient</option>}
            </select>
          </div>

          <div className="whatif-fields">
            {PATIENT_FIELDS.map((field) =>
              ranges[field.key] && field.key !== "ca" ? (
                <SliderField
                  key={field.key}
                  field={field}
                  range={ranges[field.key]}
                  value={patient[field.key]}
                  onChange={(v) => setField(field.key, v)}
                />
              ) : (
                <div className="form-row" key={field.key}>
                  <label htmlFor={`wi-${field.key}`}>{field.label}</label>
                  <select
                    id={`wi-${field.key}`}
                    value={patient[field.key]}
                    onChange={(e) => setField(field.key, Number(e.target.value))}
                  >
                    {Object.entries(field.options).map(([value, text]) => (
                      <option key={value} value={value}>
                        {text}
                      </option>
                    ))}
                  </select>
                </div>
              )
            )}
          </div>
          <p className="muted-note" style={{ margin: "12px 0 0" }}>
            Slider ranges are the values seen across the dataset&apos;s {weights.preprocessing.n_patients_total} patients,
            so the models are never asked about values they never saw.
          </p>
        </div>

        <div className="whatif-results">
          <div className="card">
            <div className="segmented" role="radiogroup" aria-label="Hospital model">
              {hospitalIds.map((id) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={id === hospitalId}
                  className={"segmented-option" + (id === hospitalId ? " active" : "")}
                  onClick={() => setHospitalId(id)}
                >
                  {hospitalLabel(id)}
                </button>
              ))}
            </div>

            <div className="whatif-risk">
              <div>
                <div className="stat-card-label">{hospitalLabel(hospitalId)} model &middot; predicted risk</div>
                <div className="prediction-probability" aria-live="polite">
                  {fmtProb(result.probability)}
                </div>
                <div className="prediction-meter" aria-hidden="true">
                  <div className="prediction-meter-track">
                    <div className="prediction-meter-fill" style={{ width: `${result.probability * 100}%` }} />
                  </div>
                  <div className="prediction-meter-threshold" style={{ left: `${THRESHOLD * 100}%` }} />
                  <div className="prediction-meter-threshold-label" style={{ left: `${THRESHOLD * 100}%` }}>
                    {fmtPct(THRESHOLD, 0)} threshold
                  </div>
                </div>
              </div>
              <dl className="prediction-facts">
                <div>
                  <dt>Risk level</dt>
                  <dd>
                    <span className={`badge ${RISK_BADGE[risk]}`}>{risk}</span>
                  </dd>
                </div>
                <div>
                  <dt>Prediction</dt>
                  <dd>{result.probability >= THRESHOLD ? "Heart disease" : "No heart disease"}</dd>
                </div>
                <div>
                  <dt>This model</dt>
                  <dd>
                    Fine-tuned on {selected.n_train} patients &middot; {fmtPct(selected.test_accuracy)} test accuracy
                  </dd>
                </div>
              </dl>
            </div>
          </div>

          <div className="card">
            <div className="whatif-heading">
              <h3>What moves this prediction</h3>
              <span className="badge badge-moderate">Approximation &middot; not SHAP</span>
            </div>
            <p className="muted-note">
              Each bar: how much the risk changes if only that one value were replaced by the average across{" "}
              {hospitalLabel(hospitalId)}&apos;s {selected.baseline.n_patients} patients (that all-average profile
              itself scores {fmtProb(result.baselineProbability)}). A quick one-at-a-time estimate &mdash; values
              interact, so the bars don&apos;t add up exactly. Real SHAP explanations are on the{" "}
              <em>Sample patients</em> tab.
            </p>
            <ShapChart
              contributions={result.contributions.slice(0, TOP_FIELDS).map((c) => ({
                feature: c.feature,
                feature_value: null,
                shap_value: c.delta,
              }))}
              rawPatient={patient}
              legend={["Lowers risk vs. hospital average", "Raises risk vs. hospital average"]}
            />
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Same patient, 5 hospitals</h3>
        <p className="muted-note">
          The inputs above, scored by each hospital&apos;s own personalized model. Click a row to switch the model used
          above.
        </p>
        <div className="across-list">
          {acrossHospitals.map((h) => (
            <button
              key={h.id}
              type="button"
              className={"across-row" + (h.id === hospitalId ? " active" : "")}
              onClick={() => setHospitalId(h.id)}
              aria-pressed={h.id === hospitalId}
            >
              <span className="across-label">
                <strong>{hospitalLabel(h.id)}</strong>
                <span>
                  {weights.hospitals[h.id].baseline.n_patients} patients &middot; {fmtPct(diseaseRate[h.id])} disease
                  rate
                </span>
              </span>
              <span className="across-track" aria-hidden="true">
                <span className="across-fill" style={{ width: `${h.probability * 100}%` }} />
                <span className="across-threshold" style={{ left: `${THRESHOLD * 100}%` }} />
              </span>
              <span className="across-value">{fmtProb(h.probability)}</span>
              <span className={`badge ${RISK_BADGE[riskLevel(h.probability)]}`}>{riskLevel(h.probability)}</span>
            </button>
          ))}
        </div>
        <p className="across-summary">
          The same inputs score between <strong>{fmtProb(minP)}</strong> and <strong>{fmtProb(maxP)}</strong> depending
          on which hospital&apos;s model is asked. Each model was fine-tuned on its own {Math.min(...trainSizes)}–
          {Math.max(...trainSizes)} training patients, and those hospitals&apos; disease rates range from{" "}
          {fmtPct(Math.min(...rates))} to {fmtPct(Math.max(...rates))}. That is personalization at work &mdash; but with
          this few patients per hospital, part of the spread is also noise, not only real differences between
          populations.
        </p>
      </div>
    </>
  );
}

function SliderField({ field, range, value, onChange }) {
  const [min, max] = range;
  const id = `wi-${field.key}`;
  const decimals = field.step < 1 ? 1 : 0;
  return (
    <div className="form-row slider-row">
      <label htmlFor={id}>
        {field.label}
        <span className="slider-value">
          {Number(value).toFixed(decimals)} {field.unit}
        </span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={field.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="slider-bounds">
        <span>{min.toFixed(decimals)}</span>
        <span>{max.toFixed(decimals)}</span>
      </span>
    </div>
  );
}
