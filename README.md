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
- [x] Phase 2 — Dataset preprocessing; non-IID hospital splits
- [x] **Phase 3 — Baseline: Local ML + Centralized ML** (this commit)
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
hospital, not hospital *size* or *minimum class count* — so a given seed can
independently produce two different problems: (a) one hospital holding 55%+
of all patients, which would let that hospital dominate FedAvg's
sample-weighted aggregation, and (b) a hospital with **zero** examples of one
class. (b) is the more serious problem: an earlier pass of this search
(seed=42, then seed=8) optimized only for size balance and both times left a
hospital with disease_rate 0.0 or 1.0 — a *single-class hospital*. That's not
just a cosmetic issue with one baseline number: a single-class hospital
can't fit Logistic Regression at all (its solver requires 2+ classes) and
forces Random Forest into a degenerate majority-class predictor, and going
into Phase 4-5 it would contribute no meaningful gradient signal for the
missing class during federated training — a problem for FedAvg/FedProx
itself, not just for reporting one baseline's accuracy.

[`experiments/phase2_seed_search.py`](experiments/phase2_seed_search.py) was
extended to search seeds 1-500 at alpha=0.5, n=5 under **two constraints
together**: (1) low max hospital share of total patients, and (2) every
hospital has **>= 5 samples of both classes**. Only 38/500 seeds satisfy
both. Among those, **seed=117** was selected: max hospital share 27.6%
(close to the best unconstrained value of 26.6%) with a comfortable
class-count margin (min 8, vs. the 5 required) — reducing the number of
simulated hospitals (to 4 or 3) was not needed since 5 hospitals already
had enough valid seeds.

| hospital | n_patients | disease_rate | mean_age | pct_female | dominant_cp_type |
|---|---|---|---|---|---|
| hospital_1 | 60 | 0.717 | 56.2 | 0.283 | 4 |
| hospital_2 | 29 | 0.690 | 54.4 | 0.310 | 4 |
| hospital_3 | 54 | 0.148 | 53.6 | 0.278 | 3 |
| hospital_4 | 82 | 0.341 | 54.7 | 0.378 | 3 |
| hospital_5 | 72 | 0.528 | 53.7 | 0.333 | 4 |

Every hospital now has both classes present in both its local train and
local test split (verified — see Phase 3 below); no hospital is single-class.

## Phase 3: Local ML vs Centralized ML baselines

Each hospital's local data was split into a local train/test set
(75/25, stratified where possible, seed=42) using
[`add_local_train_test_split`](src/data/partition.py) — every later
experiment (FedAvg, Personalized FL) will reuse this exact split so
per-hospital results stay comparable across phases.

Run: `venv\Scripts\python experiments\phase3_baselines.py` — trains Random
Forest and Logistic Regression baselines, evaluates:
- **Local ML**: each hospital's own model on its own local test set
- **Centralized ML**: one model trained on all hospitals' pooled training
  data, evaluated both overall (pooled test set) and per-hospital

Results (`experiments/results/phase3_*.csv`, `phase3_local_vs_centralized.png`),
with the seed=117 partition (no single-class hospitals):

| model | Local ML (mean/hospital) | Centralized ML (pooled) |
|---|---|---|
| random_forest | 0.833 | 0.842 |
| logistic_regression | 0.819 | 0.855 |

Direction still matches the project hypothesis (Local ≤ Centralized) for
both models, though the gap is smaller than under the earlier single-class
seed — expected, since that seed's inflated "Local ML" accuracy (below) is
gone.

**Caveat for the write-up — read before citing these numbers:** every
hospital now has both classes in its local test set, so there is no more
single-class degeneracy. However, hospital_3 (disease_rate=0.148, only 8
diseased patients total → 2 in its local test set of 14) still gets
precision/recall/F1 = 0 from both models: with only 2 positive test
examples, the model missed both. This is an honest small-sample /
class-imbalance limitation of evaluating on tiny local hospital test sets
(14-21 examples per hospital), not a single-class artifact — report it as
such, and prefer per-hospital F1/AUC breakdowns
(`phase3_local_ml_*.csv`) over accuracy alone when writing the final paper.

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
