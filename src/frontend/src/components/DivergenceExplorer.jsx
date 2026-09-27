import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { HOSPITAL_COLORS, fmtPct, fmtProb, hospitalLabel } from "../constants/experiments";
import { fieldsForModel, patientFieldText } from "../constants/features";
import { predict } from "../lib/inference";
import "./DivergenceExplorer.css";

const THRESHOLD = 0.5;
const PAGE = 12;

/**
 * Where the hospitals' personalized models disagree, on real held-out test
 * patients. Every probability is computed live in the browser by the
 * exported models (verified against torch on load), not read from a file.
 *
 * weights: model_weights.json; testPatients: test_patients.json patients;
 * hospitals: hospitals.json rows.
 */
export default function DivergenceExplorer({ weights, testPatients, hospitals }) {
  const hospitalIds = Object.keys(weights.hospitals).sort();
  const fields = fieldsForModel(weights.preprocessing);
  const [site, setSite] = useState("all");
  const [onlySplit, setOnlySplit] = useState(true);
  const [shown, setShown] = useState(PAGE);
  const [selectedId, setSelectedId] = useState(null);

  const scored = useMemo(
    () =>
      testPatients.map((t) => {
        const probs = Object.fromEntries(hospitalIds.map((h) => [h, predict(t.patient, h, weights)]));
        const values = Object.values(probs);
        const nDisease = values.filter((p) => p >= THRESHOLD).length;
        return {
          ...t,
          probs,
          spread: Math.max(...values) - Math.min(...values),
          nDisease,
          split: nDisease > 0 && nDisease < values.length,
        };
      }),
    [testPatients, weights, hospitalIds]
  );

  const splitPatients = scored.filter((t) => t.split);
  // For each model: share of all test patients it calls diseased, and how
  // often it is on the minority side of a split verdict.
  const tendency = hospitalIds.map((h) => {
    const calls = scored.filter((t) => t.probs[h] >= THRESHOLD).length;
    const minority = splitPatients.filter((t) => {
      const says = t.probs[h] >= THRESHOLD;
      const sameSide = hospitalIds.filter((o) => (t.probs[o] >= THRESHOLD) === says).length;
      return sameSide < hospitalIds.length - sameSide;
    }).length;
    return {
      hospital: h,
      diseaseCallRate: calls / scored.length,
      ownDiseaseRate: hospitals.find((x) => x.hospital === h)?.disease_rate,
      minority,
    };
  });
  const mostMinority = tendency.reduce((a, b) => (b.minority > a.minority ? b : a));

  const list = scored
    .filter((t) => (site === "all" || t.site === site) && (!onlySplit || t.split))
    .sort((a, b) => b.spread - a.spread);
  const selected = scored.find((t) => t.id === selectedId) ?? list[0];

  return (
    <div className="divergence">
      <div className="headline-callout">
        <span className="headline-callout-label">Where the models disagree</span>
        <span>
          The four hospitals&apos; personalized models give <strong>conflicting verdicts</strong> (some say disease,
          some don&apos;t) for {splitPatients.length} of {scored.length} held-out test patients (
          {fmtPct(splitPatients.length / scored.length)}). The model most often on the minority side is{" "}
          {hospitalLabel(mostMinority.hospital)}&apos;s ({mostMinority.minority} of {splitPatients.length}).
        </span>
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Each model&apos;s tendency</h3>
          <p className="muted-note">
            Share of all {scored.length} test patients each model calls diseased, next to its own hospital&apos;s
            disease rate &mdash; a model fine-tuned on a mostly-diseased hospital leans towards &ldquo;disease&rdquo;
          </p>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Calls diseased</th>
                <th>Its hospital&apos;s rate</th>
                <th>In minority</th>
              </tr>
            </thead>
            <tbody>
              {tendency.map((t) => (
                <tr key={t.hospital}>
                  <td>
                    <span className="div-swatch" style={{ background: HOSPITAL_COLORS[t.hospital] }} />
                    {hospitalLabel(t.hospital)}
                  </td>
                  <td>
                    <strong>{fmtPct(t.diseaseCallRate)}</strong>
                  </td>
                  <td className="muted-inline">{t.ownDiseaseRate != null ? fmtPct(t.ownDiseaseRate) : "—"}</td>
                  <td>
                    {t.minority} / {splitPatients.length}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Conflicting verdicts by hospital</h3>
          <p className="muted-note">Test patients of each hospital on whom the four models split</p>
          <table>
            <thead>
              <tr>
                <th>Patients from</th>
                <th>Split</th>
                <th>Share</th>
              </tr>
            </thead>
            <tbody>
              {hospitalIds.map((h) => {
                const own = scored.filter((t) => t.site === h);
                const s = own.filter((t) => t.split).length;
                return (
                  <tr key={h}>
                    <td>{hospitalLabel(h)}</td>
                    <td>
                      {s} / {own.length}
                    </td>
                    <td>{fmtPct(s / own.length)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="div-layout">
        <div className="card">
          <div className="div-filters">
            <label>
              Patients from{" "}
              <select value={site} onChange={(e) => { setSite(e.target.value); setShown(PAGE); setSelectedId(null); }}>
                <option value="all">all hospitals</option>
                {hospitalIds.map((h) => (
                  <option key={h} value={h}>
                    {hospitalLabel(h)}
                  </option>
                ))}
              </select>
            </label>
            <label className="div-check">
              <input type="checkbox" checked={onlySplit} onChange={(e) => { setOnlySplit(e.target.checked); setShown(PAGE); setSelectedId(null); }} />
              only conflicting verdicts
            </label>
            <span className="muted-inline">{list.length} patients, largest disagreement first</span>
          </div>

          <ul className="div-list" role="listbox" aria-label="Test patients">
            {list.slice(0, shown).map((t) => (
              <li key={t.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={selected?.id === t.id}
                  className={"div-row" + (selected?.id === t.id ? " active" : "")}
                  onClick={() => setSelectedId(t.id)}
                >
                  <span className="div-row-label">
                    <strong>{hospitalLabel(t.site)} patient #{t.test_row + 1}</strong>
                    <span>
                      {t.patient.age} yrs &middot; {t.actual_label === 1 ? "had disease" : "no disease"}
                    </span>
                  </span>
                  <span className="div-mini" aria-hidden="true">
                    <span className="div-mini-threshold" />
                    {hospitalIds.map((h) => (
                      <span
                        key={h}
                        className="div-mini-dot"
                        style={{ left: `${t.probs[h] * 100}%`, background: HOSPITAL_COLORS[h] }}
                      />
                    ))}
                  </span>
                  <span className="div-row-spread">{(t.spread * 100).toFixed(0)} pts apart</span>
                </button>
              </li>
            ))}
          </ul>
          {shown < list.length && (
            <button type="button" className="button-secondary" onClick={() => setShown((n) => n + PAGE)}>
              Show {Math.min(PAGE, list.length - shown)} more
            </button>
          )}
          {list.length === 0 && <p className="muted-note">No patients match these filters.</p>}
        </div>

        {selected && (
          <div className="card div-detail">
            <h3>
              {hospitalLabel(selected.site)} patient #{selected.test_row + 1}
            </h3>
            <p className="muted-note">
              Actually {selected.actual_label === 1 ? "had heart disease" : "had no heart disease"} &middot; held-out
              test patient of {hospitalLabel(selected.site)}
            </p>
            <div className="div-bars">
              {hospitalIds.map((h) => {
                const p = selected.probs[h];
                const right = (p >= THRESHOLD ? 1 : 0) === selected.actual_label;
                return (
                  <div key={h} className="div-bar-row">
                    <span className="div-bar-label">
                      {hospitalLabel(h)} model{h === selected.site && <span className="div-own"> (own)</span>}
                    </span>
                    <span className="div-bar-track">
                      <span className="div-bar-fill" style={{ width: `${p * 100}%`, background: HOSPITAL_COLORS[h] }} />
                      <span className="div-bar-threshold" />
                    </span>
                    <span className="div-bar-value">{fmtProb(p)}</span>
                    <span className={`badge ${right ? "badge-low" : "badge-high"}`}>{right ? "✓ right" : "✗ wrong"}</span>
                  </div>
                );
              })}
            </div>
            <table className="patient-table" style={{ marginTop: 14 }}>
              <tbody>
                {fields.map((f) => (
                  <tr key={f.key}>
                    <th scope="row">{f.label}</th>
                    <td>{patientFieldText(f, selected.patient[f.key])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Link to={`/explainability?view=try&patient=${selected.id}`} className="card-link">
              Open this patient in the what-if explorer →
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
