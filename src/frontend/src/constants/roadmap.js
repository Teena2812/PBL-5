// The 8-phase plan from docs/proposal/ and the README's "Project status"
// checklist. Keep `done` in sync with that checklist.
export const PHASES = [
  { n: 1, title: "Literature survey, dataset & gap statement", done: true },
  { n: 2, title: "Dataset preprocessing; non-IID hospital splits", done: true },
  { n: 3, title: "Baselines: Local ML + Centralized ML", done: true },
  { n: 4, title: "Federated learning (FedAvg via Flower)", done: true },
  { n: 5, title: "Personalization (FedProx + fine-tuning) vs FedAvg", done: true },
  { n: 6, title: "SHAP explainability integration", done: true },
  { n: 7, title: "FastAPI backend + React dashboard", done: true },
  { n: 8, title: "Results compilation, research paper draft & presentation", done: false },
];
