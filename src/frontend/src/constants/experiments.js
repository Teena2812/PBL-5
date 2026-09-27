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

// One color per real site. Validated as a set of 4 on all pairs (lines
// cross in the replay chart): colorblind dE >= 9.2, normal-vision dE >= 16.3.
export const HOSPITAL_COLORS = {
  cleveland: "#2a78d6",
  hungary: "#eb6834",
  switzerland: "#1baf7a",
  va_long_beach: "#4a3aa7",
};

// Short display names for the 4 UCI Heart Disease sites (hospitals.json
// carries the full institution names).
export const HOSPITAL_NAMES = {
  cleveland: "Cleveland",
  hungary: "Hungary",
  switzerland: "Switzerland",
  va_long_beach: "VA Long Beach",
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
  return HOSPITAL_NAMES[id] ?? id;
}

/**
 * Hospitals whose own local model beats FedAvg on accuracy in the seed-42
 * run (the only run with per-hospital results). The headline is about
 * overall accuracy; these are the exceptions every headline mention must
 * not paper over.
 */
export function hospitalsWhereAloneWins(comparison) {
  const fedavg = new Map(comparison.per_hospital.fedavg.map((r) => [r.hospital, r.accuracy]));
  return comparison.per_hospital.local
    .filter((r) => r.accuracy > (fedavg.get(r.hospital) ?? Infinity))
    .map((r) => ({ hospital: r.hospital, local: r.accuracy, fedavg: fedavg.get(r.hospital) }));
}
