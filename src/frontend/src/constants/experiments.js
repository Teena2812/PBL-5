// Shared identity for the five training settings, so each one keeps the
// same name and color on every screen.
//
// Colors were run through the dataviz palette validator in this display
// order (local, centralized, fedavg, fedprox, personalized): all adjacent
// pairs clear the colorblind (CVD dE >= 9.1) and normal-vision (dE >= 16.3)
// floors. Yellow and aqua sit below 3:1 contrast on white, so charts using
// them must also show direct value labels or a table.

export const EXPERIMENT_ORDER = ["local", "centralized", "fedavg", "fedprox", "personalized"];

export const EXPERIMENT_LABELS = {
  local: "Local ML",
  centralized: "Centralized",
  fedavg: "FedAvg",
  fedprox: "FedProx",
  personalized: "Personalized",
};

export const EXPERIMENT_COLORS = {
  local: "#eda100",
  centralized: "#4a3aa7",
  fedavg: "#2a78d6",
  fedprox: "#1baf7a",
  personalized: "#eb6834",
};

// Experiment names as written by the Phase 4/5 multi-seed CSVs.
export const MULTISEED_NAME_TO_KEY = {
  "Local NN": "local",
  "Centralized NN": "centralized",
  "FedAvg NN": "fedavg",
  "FedProx NN": "fedprox",
  "Personalized (FedProx+FT)": "personalized",
};

export function fmtPct(x, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

// A model probability as a percentage, without rounding a near-certain
// output (e.g. 0.9998) up to a misleading "100%" or down to "0%".
export function fmtProb(p, digits = 1) {
  const floor = 10 ** -(digits + 2);
  if (p > 1 - floor) return `>${fmtPct(1 - floor, digits)}`;
  if (p < floor) return `<${fmtPct(floor, digits)}`;
  return fmtPct(p, digits);
}

export function hospitalLabel(id) {
  return id.replace("hospital_", "Hospital ");
}
