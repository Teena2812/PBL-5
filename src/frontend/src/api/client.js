import axios from "axios";

// Vite exposes env vars prefixed with VITE_ via import.meta.env.
// Defaults to the local FastAPI dev server (uvicorn default port 8000).
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
});

export async function getHealth() {
  const { data } = await api.get("/health");
  return data;
}

export async function getDashboardSummary() {
  const { data } = await api.get("/dashboard/summary");
  return data;
}

export async function getHospitals() {
  const { data } = await api.get("/hospitals");
  return data;
}

export async function getExperimentComparison() {
  const { data } = await api.get("/experiments/comparison");
  return data;
}

export async function getTrainingCurves() {
  const { data } = await api.get("/experiments/training-curves");
  return data;
}

export async function getEquityAnalysis() {
  const { data } = await api.get("/equity");
  return data;
}

export async function getExplainability() {
  const { data } = await api.get("/explainability");
  return data;
}

/** The 5 personalized models' weights + preprocessing, for in-browser inference. */
export async function getModelWeights() {
  const { data } = await api.get("/models/weights");
  return data;
}

/** Precomputed sample-patient predictions (empty list until generated on Colab). */
export async function getSamplePatients() {
  const { data } = await api.get("/explainability/samples");
  return data;
}

/**
 * POST /api/predict. Live inference + single-instance SHAP explanation
 * only -- the backend never retrains anything here.
 */
export async function predict(patient) {
  const { data } = await api.post("/predict", patient);
  return data;
}
