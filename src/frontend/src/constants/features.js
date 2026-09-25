// Human-readable names for the 13 UCI clinical inputs and the 22 encoded
// model features SHAP explains (6 standardized numerics, 3 binaries, and
// one-hot columns for cp / restecg / slope / thal).

export const CATEGORY_LABELS = {
  cp: { 1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic" },
  restecg: { 0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy" },
  slope: { 1: "Upsloping", 2: "Flat", 3: "Downsloping" },
  // Only the three thal codes the model was trained on (preprocessing.json).
  thal: { 3: "Normal", 6: "Fixed defect", 7: "Reversible defect" },
};

const CATEGORY_PREFIX = {
  cp: "Chest pain",
  restecg: "Resting ECG",
  slope: "ST slope",
  thal: "Thallium test",
};

const BASE_LABELS = {
  age: "Age",
  sex: "Sex",
  trestbps: "Resting blood pressure",
  chol: "Cholesterol",
  fbs: "Fasting blood sugar > 120",
  thalach: "Max heart rate",
  exang: "Exercise-induced angina",
  oldpeak: "ST depression (oldpeak)",
  ca: "Major vessels (fluoroscopy)",
  // Whole categorical fields (used by the in-browser per-field breakdown).
  cp: "Chest pain type",
  restecg: "Resting ECG",
  slope: "ST slope",
  thal: "Thallium test",
};

const UNITS = { age: "yrs", trestbps: "mm Hg", chol: "mg/dl", thalach: "bpm" };

// Phase 6's RF export names some one-hot columns "cp_4.0"; the model
// manifest uses "cp_4". Normalize to the latter.
export function normalizeFeature(name) {
  return name.replace(/\.0$/, "");
}

function splitOneHot(feature) {
  const match = /^(cp|restecg|slope|thal)_(\d+)$/.exec(feature);
  return match ? { base: match[1], code: Number(match[2]) } : null;
}

// Shorter category names for chart axis labels, where space is tight.
const SHORT_CATEGORY_LABELS = { restecg: { 1: "ST-T abnormality", 2: "LV hypertrophy" } };

export function featureLabel(name) {
  const feature = normalizeFeature(name);
  const oneHot = splitOneHot(feature);
  if (oneHot) {
    const category =
      SHORT_CATEGORY_LABELS[oneHot.base]?.[oneHot.code] ??
      CATEGORY_LABELS[oneHot.base]?.[oneHot.code] ??
      `type ${oneHot.code}`;
    return `${CATEGORY_PREFIX[oneHot.base]}: ${category}`;
  }
  return BASE_LABELS[feature] ?? feature;
}

/**
 * The patient's value for an encoded feature, in clinical terms when the raw
 * record is known (sample / live patients), otherwise from the encoded value
 * (numerics are standardized there, so shown as SDs from the mean).
 */
export function featureValueText(name, encodedValue, rawPatient) {
  const feature = normalizeFeature(name);
  const oneHot = splitOneHot(feature);
  if (oneHot) {
    const isSet = rawPatient ? rawPatient[oneHot.base] === oneHot.code : encodedValue === 1;
    return isSet ? "yes" : "no";
  }
  if (CATEGORY_LABELS[feature] && rawPatient) {
    const code = rawPatient[feature];
    return SHORT_CATEGORY_LABELS[feature]?.[code] ?? CATEGORY_LABELS[feature][code] ?? String(code);
  }
  if (["sex", "fbs", "exang"].includes(feature)) {
    const value = rawPatient ? rawPatient[feature] : encodedValue;
    if (feature === "sex") return value === 1 ? "male" : "female";
    return value === 1 ? "yes" : "no";
  }
  if (rawPatient) {
    const value = feature === "oldpeak" ? Number(rawPatient[feature]).toFixed(1) : String(rawPatient[feature]);
    return UNITS[feature] ? `${value} ${UNITS[feature]}` : value;
  }
  const sign = encodedValue >= 0 ? "+" : "−";
  return `${sign}${Math.abs(encodedValue).toFixed(2)} SD`;
}

// Form definition for the 13 clinical inputs. Numeric bounds mirror the
// backend's PatientInput sanity ranges (src/backend/schemas.py).
export const PATIENT_FIELDS = [
  { key: "age", label: "Age", type: "number", min: 1, max: 120, step: 1, unit: "years" },
  { key: "sex", label: "Sex", type: "select", options: { 1: "Male", 0: "Female" } },
  { key: "cp", label: "Chest pain type", type: "select", options: CATEGORY_LABELS.cp },
  { key: "trestbps", label: "Resting blood pressure", type: "number", min: 50, max: 300, step: 1, unit: "mm Hg" },
  { key: "chol", label: "Serum cholesterol", type: "number", min: 50, max: 700, step: 1, unit: "mg/dl" },
  { key: "fbs", label: "Fasting blood sugar > 120 mg/dl", type: "select", options: { 0: "No", 1: "Yes" } },
  { key: "restecg", label: "Resting ECG", type: "select", options: CATEGORY_LABELS.restecg },
  { key: "thalach", label: "Max heart rate achieved", type: "number", min: 50, max: 250, step: 1, unit: "bpm" },
  { key: "exang", label: "Exercise-induced angina", type: "select", options: { 0: "No", 1: "Yes" } },
  { key: "oldpeak", label: "ST depression (oldpeak)", type: "number", min: 0, max: 10, step: 0.1, unit: "mm" },
  { key: "slope", label: "Peak exercise ST slope", type: "select", options: CATEGORY_LABELS.slope },
  { key: "ca", label: "Major vessels colored (fluoroscopy)", type: "select", options: { 0: "0", 1: "1", 2: "2", 3: "3" } },
  { key: "thal", label: "Thallium stress test", type: "select", options: CATEGORY_LABELS.thal },
];

// Same record as the backend's PatientInput example (schemas.py).
export const DEFAULT_PATIENT = {
  age: 63, sex: 1, cp: 4, trestbps: 145, chol: 233, fbs: 1, restecg: 0,
  thalach: 150, exang: 0, oldpeak: 2.3, slope: 1, ca: 0, thal: 6,
};

export function patientFieldText(field, value) {
  if (field.type === "select") return field.options[value] ?? String(value);
  // Show fractional fields (oldpeak) at their input precision, e.g. 4.0.
  const text = field.step < 1 ? Number(value).toFixed(1) : String(value);
  return field.unit ? `${text} ${field.unit}` : text;
}
