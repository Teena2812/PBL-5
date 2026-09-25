import axios from "axios";

// Vite exposes env vars prefixed with VITE_ via import.meta.env.
// Defaults to the local FastAPI dev server (uvicorn default port 8000).
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

// Static builds (GitHub Pages) have no backend: every dashboard endpoint is
// just one of the committed JSON files in experiments/results/dashboard_data/,
// copied into the build's data/ folder by scripts/copy-dashboard-data.mjs.
const STATIC_DATA = import.meta.env.VITE_STATIC_DATA === "true";
const STATIC_FILES = {
  "/dashboard/summary": "dashboard_summary.json",
  "/hospitals": "hospitals.json",
  "/experiments/comparison": "experiment_comparison.json",
  "/experiments/training-curves": "training_curves.json",
  "/equity": "equity_analysis.json",
  "/explainability": "explainability.json",
  "/models/weights": "model_weights.json",
  "/explainability/samples": "sample_patients.json",
};

export const api = STATIC_DATA
  ? axios.create({ baseURL: `${import.meta.env.BASE_URL}data`, timeout: 15000 })
  : axios.create({ baseURL: BASE_URL, timeout: 15000 });

async function getData(endpoint) {
  const { data } = await api.get(STATIC_DATA ? `/${STATIC_FILES[endpoint]}` : endpoint);
  return data;
}

export async function getHealth() {
  if (STATIC_DATA) return { status: "ok", live_prediction_available: false, static_build: true };
  const { data } = await api.get("/health");
  return data;
}

export async function getDashboardSummary() {
  return getData("/dashboard/summary");
}

export async function getHospitals() {
  return getData("/hospitals");
}

export async function getExperimentComparison() {
  return getData("/experiments/comparison");
}

export async function getTrainingCurves() {
  return getData("/experiments/training-curves");
}

export async function getEquityAnalysis() {
  return getData("/equity");
}

export async function getExplainability() {
  return getData("/explainability");
}

/** The 5 personalized models' weights + preprocessing, for in-browser inference. */
export async function getModelWeights() {
  return getData("/models/weights");
}

/** Precomputed sample-patient predictions (empty list until generated on Colab). */
export async function getSamplePatients() {
  const data = await getData("/explainability/samples");
  // The backend adds generated: true/false; the raw file doesn't have it.
  return { generated: data.samples.length > 0, ...data };
}

/**
 * POST /api/predict. Live inference + single-instance SHAP explanation
 * only -- the backend never retrains anything here.
 */
export async function predict(patient) {
  // Not used by the dashboard (predictions run in the browser); the backend
  // endpoint only exists when the FastAPI server is running.
  if (STATIC_DATA) throw new Error("POST /api/predict needs the FastAPI backend; this is a static build.");
  const { data } = await api.post("/predict", patient);
  return data;
}
