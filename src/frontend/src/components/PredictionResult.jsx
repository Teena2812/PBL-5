import ShapChart from "./ShapChart";
import "./PredictionResult.css";
import { fmtPct, fmtProb, hospitalLabel } from "../constants/experiments";

const RISK_BADGE = { LOW: "badge-low", MODERATE: "badge-moderate", HIGH: "badge-high" };

// Matches the backend's predicted_label rule (probability >= 0.5).
const THRESHOLD = 0.5;
// Within this distance of the threshold a prediction is a coin-flip call.
const BORDERLINE_MARGIN = 0.05;

/**
 * One prediction as returned by POST /api/predict (or precomputed with the
 * same function): risk summary + SHAP chart. actualLabel is only known for
 * sample patients drawn from a hospital's test set.
 */
export default function PredictionResult({ result, rawPatient, actualLabel }) {
  const predictedDisease = result.predicted_label === 1;
  const hasActual = actualLabel === 0 || actualLabel === 1;
  const correct = hasActual && actualLabel === result.predicted_label;
  const shapTotal = result.explanation.reduce((sum, c) => sum + c.shap_value, 0);
  const distance = result.predicted_probability - THRESHOLD;
  const borderline = Math.abs(distance) < BORDERLINE_MARGIN;

  return (
    <>
      <div className="prediction-summary">
        <div>
          <div className="stat-card-label">Predicted disease probability</div>
          <div className="prediction-probability">{fmtProb(result.predicted_probability)}</div>
          <div className="prediction-meter" aria-hidden="true">
            <div className="prediction-meter-track">
              <div className="prediction-meter-fill" style={{ width: `${result.predicted_probability * 100}%` }} />
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
              <span className={`badge ${RISK_BADGE[result.risk_level] || "badge-neutral"}`}>{result.risk_level}</span>
            </dd>
          </div>
          <div>
            <dt>Prediction</dt>
            <dd>{predictedDisease ? "Heart disease" : "No heart disease"}</dd>
          </div>
          {hasActual && (
            <div>
              <dt>Actual outcome</dt>
              <dd>
                {actualLabel === 1 ? "Heart disease" : "No heart disease"}{" "}
                <span className={`badge ${correct ? "badge-low" : "badge-high"}`}>
                  {correct ? "✓ correct" : "✗ missed"}
                </span>
              </dd>
            </div>
          )}
          <div>
            <dt>Model</dt>
            <dd>
              {hospitalLabel(result.hospital_id)} personalized &middot; {fmtPct(result.model_test_accuracy)} test
              accuracy
            </dd>
          </div>
        </dl>
      </div>

      {borderline && (
        <div className="borderline-note">
          <strong>Borderline call.</strong> {fmtPct(result.predicted_probability)} is{" "}
          {Math.abs(distance * 100).toFixed(1)} pts {distance < 0 ? "below" : "above"} the {fmtPct(THRESHOLD, 0)}{" "}
          decision threshold, so the model was nearly undecided
          {hasActual && !correct ? " — this miss is a near-coin-flip, not a confident error." : "."}
        </div>
      )}

      <h4 style={{ margin: "20px 0 4px 0" }}>Why: top {result.explanation.length} feature contributions (SHAP)</h4>
      <p className="muted-note">
        Starts from the model&apos;s average output over this hospital&apos;s training patients (
        {fmtPct(result.base_value)}); these features together move it by {shapTotal >= 0 ? "+" : "−"}
        {Math.abs(shapTotal * 100).toFixed(1)} pts. The remaining features account for the rest.
      </p>
      <ShapChart contributions={result.explanation} rawPatient={rawPatient} />
    </>
  );
}
