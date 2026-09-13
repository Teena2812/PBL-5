# Personalized Federated Learning for Explainable and Privacy-Preserving Disease Risk Prediction in Heterogeneous Healthcare Systems

> **Research prototype, not a deployed medical tool.** This project simulates
> multiple hospitals by partitioning one public dataset (UCI Heart Disease,
> Cleveland) into non-IID splits. No real multi-hospital data or deployment
> is involved. Results are experimental and for academic/research purposes
> (PBL-3, Sharda University, Dept. of CSE (AI/ML), CSP-391).

## What this is

Hospitals cannot legally pool patient data, and a single global model trained
via standard Federated Learning (FedAvg) often underperforms for individual
hospitals whose patient populations differ (non-IID data). This project
studies whether combining **personalization (FedProx)** and
**explainability (SHAP)** on top of FedAvg improves both accuracy and
clinical interpretability, compared to FL alone, on simulated heterogeneous
hospital data.

Full problem framing, literature survey (48 papers), and proposal report are
in [`docs/proposal/`](docs/proposal/).

## Architecture

```
Layer 1 — Federated Learning: FedAvg via Flower across simulated hospital clients.
           Only model weight updates are transmitted — never raw patient data.
Layer 2 — Personalization: FedProx (proximal term) + local fine-tuning per hospital.
Layer 3 — Explainability: SHAP applied to every prediction (ranked feature contributions).
```

## Project status

Following the 8-phase roadmap in `docs/proposal/`:

- [x] Phase 1 — Literature survey, dataset & gap statement
- [x] **Phase 2 — Dataset preprocessing; non-IID hospital splits** (this commit)
- [ ] Phase 3 — Baseline: Local ML + Centralized ML
- [ ] Phase 4 — Federated Learning (FedAvg via Flower)
- [ ] Phase 5 — Personalization (FedProx) vs FedAvg comparison
- [ ] Phase 6 — SHAP explainability integration
- [ ] Phase 7 — FastAPI backend + React dashboard
- [ ] Phase 8 — Results compilation, research paper draft

## Dataset

UCI Heart Disease dataset, Cleveland processed subset (auto-downloaded to
`data/raw/` on first run). 303 patients, 14 clinical features
(age, sex, chest pain type, resting BP, cholesterol, etc.). The multi-class
severity target (0-4) is binarized to 0 = no disease, 1 = disease present,
following the standard convention for this dataset. 6 rows with missing
`ca`/`thal` values are dropped, leaving 297 patients.

## Non-IID hospital simulation strategy

> **Methodology statement (use this exact framing in the final report's
> Methodology section):** Hospital heterogeneity is simulated using
> **Dirichlet label-skew partitioning** — each simulated hospital's
> distribution over the binary disease label is drawn from a
> `Dirichlet(alpha)` distribution, not by splitting on a demographic or
> clinical feature (e.g. age band or chest-pain type). This is a deliberate
> refinement of the splitting strategy loosely described in the original
> proposal docs (`docs/proposal/`), which mentioned feature-based skew
> (age band, cp-type) as illustrative examples before the implementation
> phase settled on the literature-standard method below. Any reference to
> "age-band" or "feature-based" splitting elsewhere in `docs/proposal/`
> should be read as superseded by this document.

We use Dirichlet label-skew partitioning (Hsu et al., 2019), the standard
non-IID simulation method in the federated learning literature. Each
simulated hospital's disease-prevalence rate is drawn from a
`Dirichlet(alpha)` distribution per class:

- **Low alpha (e.g. 0.1)** → highly skewed hospitals (some hospitals almost
  entirely healthy patients, others almost entirely diseased) — the
  low-alpha regime commonly used to stress-test personalized FL.
- **Moderate alpha (e.g. 0.5, our default)** → realistic heterogeneity.
- **High alpha (e.g. 5.0)** → close to IID, used as a control/ablation.

This approach was chosen over a fixed feature-based split because it is (1)
the standard, citable method reviewers will recognize, (2) tunable via one
parameter (`alpha`) for ablation experiments without rewriting the splitting
logic, and (3) scales to any number of clients. See
[`src/data/partition.py`](src/data/partition.py) for the implementation and
[`experiments/phase2_partition_report.py`](experiments/phase2_partition_report.py)
for the reproducible report (summary table + chart, `experiments/results/`).

**Seed selection.** Dirichlet label-skew controls class *proportions* per
hospital, not hospital *size* — some seeds produced one hospital holding
55%+ of all patients, which would let that hospital dominate FedAvg's
sample-weighted aggregation and undermine the FedAvg-vs-personalized-FL
comparison later. [`experiments/phase2_seed_search.py`](experiments/phase2_seed_search.py)
searched seeds 1-50 at alpha=0.5, n=5 and selected **seed=8** as the
default: no hospital holds more than 23.9% of patients (vs. worst-case
seeds >55%), while disease rate still ranges from 0.0 to 0.97 across
hospitals against a global rate of 0.46 — balanced enough for meaningful
aggregation, still clearly non-IID.

| hospital | n_patients | disease_rate | mean_age | pct_female | dominant_cp_type |
|---|---|---|---|---|---|
| hospital_1 | 71 | 0.366 | 54.0 | 0.366 | 4 |
| hospital_2 | 62 | 0.000 | 53.3 | 0.452 | 3 |
| hospital_3 | 64 | 0.422 | 53.4 | 0.188 | 4 |
| hospital_4 | 63 | 0.762 | 55.2 | 0.333 | 4 |
| hospital_5 | 37 | 0.973 | 58.5 | 0.243 | 4 |

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

> Note: on this development machine, `pandas>=3.0` failed to import due to a
> Windows Application Control policy blocking one of its DLLs. `pandas` is
> pinned to `2.2.3` in `requirements.txt`, which works correctly.

## Running Phase 2

```bash
venv\Scripts\python experiments\phase2_partition_report.py
```

Downloads the dataset (if needed), preprocesses it, builds 5 non-IID
hospital partitions, and writes a summary CSV + heterogeneity chart to
`experiments/results/`.

## Folder structure

```
PBL-5/
├── docs/proposal/       ← approved planning docs (problem def, concept, lit survey, report)
├── research_papers/     ← literature collection + papers_index.csv
├── src/
│   ├── data/             ← dataset loading, preprocessing, non-IID splitting
│   ├── models/            ← baseline sklearn models + PyTorch FL model (Phase 3-4)
│   ├── federated/          ← Flower client/server/strategy code (Phase 4-5)
│   ├── explainability/       ← SHAP integration (Phase 6)
│   ├── backend/                ← FastAPI app (Phase 7)
│   └── frontend/                 ← React dashboard (Phase 7)
├── experiments/          ← scripts to run comparison experiments + results/plots
├── requirements.txt
└── README.md
```
