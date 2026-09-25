// Copies the committed dashboard JSON (experiments/results/dashboard_data/)
// into public/data/ so a static build serves exactly the same files the
// FastAPI backend would. Run before `vite build` for static deployments.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "..", "..", "experiments", "results", "dashboard_data");
const target = join(here, "..", "public", "data");

const REQUIRED = [
  "dashboard_summary.json",
  "hospitals.json",
  "experiment_comparison.json",
  "training_curves.json",
  "equity_analysis.json",
  "explainability.json",
  "model_weights.json",
  "sample_patients.json",
];

mkdirSync(target, { recursive: true });
const available = new Set(readdirSync(source));
const missing = REQUIRED.filter((f) => !available.has(f));
if (missing.length) {
  console.error(`Missing dashboard data: ${missing.join(", ")} (in ${source})`);
  process.exit(1);
}
for (const f of REQUIRED) copyFileSync(join(source, f), join(target, f));
console.log(`Copied ${REQUIRED.length} dashboard data files to public/data/`);
