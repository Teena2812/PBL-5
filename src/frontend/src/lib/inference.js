// In-browser inference for the 5 personalized HeartDiseaseNet models, using
// the weights exported by experiments/export_model_weights.py. Mirrors the
// backend exactly: encode like src/data/load_dataset.py:transform_new_patient,
// then Linear -> ReLU -> Linear -> ReLU -> Linear -> sigmoid.

export const CLINICAL_FIELDS = [
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal",
];

/** Raw clinical record -> encoded feature vector in the model's column order. */
export function encodePatient(raw, weights) {
  const pre = weights.preprocessing;
  const row = {};
  for (const f of pre.numeric_features) row[f] = (Number(raw[f]) - pre.means[f]) / pre.stds[f];
  for (const f of pre.binary_features) row[f] = Number(raw[f]);
  for (const f of pre.categorical_features) {
    const value = Number(raw[f]);
    const categories = pre.categorical_categories[f];
    if (!categories.includes(value)) {
      throw new Error(`'${f}'=${value} was never seen during training (valid values: ${categories.join(", ")})`);
    }
    for (const c of categories) row[`${f}_${c}`] = value === c ? 1 : 0;
  }
  return weights.feature_order.map((name) => row[name]);
}

/** Forward pass for one hospital's model; returns the disease probability. */
export function forwardProbability(layers, x) {
  let h = x;
  layers.forEach((layer, i) => {
    const out = layer.bias.map((b, j) => {
      let sum = b;
      const w = layer.weight[j];
      for (let k = 0; k < h.length; k++) sum += w[k] * h[k];
      return sum;
    });
    h = i < layers.length - 1 ? out.map((v) => Math.max(v, 0)) : out;
  });
  return 1 / (1 + Math.exp(-h[0]));
}

export function predict(raw, hospitalId, weights) {
  return forwardProbability(weights.hospitals[hospitalId].layers, encodePatient(raw, weights));
}

// Encoded column indices belonging to each clinical field (one-hot fields
// own several columns, e.g. cp -> cp_1..cp_4).
function fieldColumns(weights) {
  const map = {};
  weights.feature_order.forEach((name, i) => {
    const field = CLINICAL_FIELDS.find((f) => name === f || name.startsWith(`${f}_`));
    (map[field] ??= []).push(i);
  });
  return map;
}

/**
 * Approximate per-field breakdown (NOT SHAP): for each clinical field, how
 * much the predicted probability changes if only that field is swapped for
 * this hospital's average patient. Positive = this patient's value raises
 * risk relative to the hospital average. Fields interact, so these do not
 * sum exactly to (risk - average patient's risk).
 */
export function deviationBreakdown(raw, hospitalId, weights) {
  const hospital = weights.hospitals[hospitalId];
  const x = encodePatient(raw, weights);
  const p = forwardProbability(hospital.layers, x);
  const baseline = hospital.baseline.mean_encoded;
  const columns = fieldColumns(weights);

  const contributions = CLINICAL_FIELDS.map((field) => {
    const swapped = x.slice();
    for (const i of columns[field]) swapped[i] = baseline[i];
    return { feature: field, delta: p - forwardProbability(hospital.layers, swapped) };
  }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  return { probability: p, baselineProbability: forwardProbability(hospital.layers, baseline), contributions };
}

/**
 * Re-scores the exported verification cases (real predict_and_explain()
 * outputs, saved at 4 dp) and reports whether every one matches within that
 * rounding. The UI refuses to show in-browser predictions if this fails.
 */
export function verifyAgainstBackend(weights) {
  const cases = weights.verification.cases.map((c) => {
    const probability = predict(c.patient, c.hospital_id, weights);
    return { ...c, probability, diff: Math.abs(probability - c.expected_probability) };
  });
  const maxDiff = Math.max(...cases.map((c) => c.diff));
  return { ok: maxDiff <= 5e-5, maxDiff, cases };
}

/** Risk band, same cut-offs as the backend's _risk_level(). */
export function riskLevel(p) {
  if (p >= 0.66) return "HIGH";
  if (p >= 0.33) return "MODERATE";
  return "LOW";
}
