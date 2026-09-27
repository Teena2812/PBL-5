# Personalized Federated Learning for Explainable and Privacy-Preserving Disease Risk Prediction in Heterogeneous Healthcare Systems

> **Research prototype, not a deployed medical tool.** It uses the public UCI
> Heart Disease data from **four real institutions** (Cleveland, Hungary,
> Switzerland, VA Long Beach; collected in 1988) as four federated clients --
> **849 patients in total**. Those hospitals never actually ran federated
> training; the federation is simulated on their published records. Results
> are experimental and for academic/research purposes (PBL-3, Sharda
> University, Dept. of CSE (AI/ML), CSP-391).

**Live dashboard:** https://teena2812.github.io/PBL-5/

## Headline result

> **Federated training beats hospitals training alone in 5/5 seeds (82.6% vs
> 78.8% overall accuracy, +3.9pp average) and matches or beats full data
> pooling in 5/5 seeds -- without sharing patient records.**

"Overall accuracy" is the share of all 213 held-out test patients (from all
four hospitals) classified correctly. The same small network is used in every
setting, so the comparison is like for like. Differences of about ±1-2
patients / ±0.5-1 pp between identical runs are run-to-run noise (see the
reproducibility caveat below).

## What this is

Hospitals cannot legally pool patient data, and a single global model trained
with standard Federated Learning (FedAvg) can underperform for hospitals whose
patient populations differ (non-IID data). This project measures, on real
multi-site data, whether hospitals gain from training together without sharing
records, whether **personalization (FedProx + local fine-tuning)** helps the
hospitals the shared model serves worst, and whether **explainability (SHAP)**
gives clinically plausible reasons for each prediction.

Full problem framing, literature survey (48 papers), and proposal report are
in [`docs/proposal/`](docs/proposal/).

## Architecture

```
Layer 1 — Federated Learning: FedAvg via Flower across the 4 hospital clients.
           Only model weight updates are transmitted — never raw patient data.
Layer 2 — Personalization: FedProx (proximal term) + local fine-tuning per hospital.
Layer 3 — Explainability: SHAP feature contributions for individual predictions.
```

Model: `HeartDiseaseNet`, 13 encoded inputs → 16 → 8 → 1 (369 parameters),
identical in every setting compared.

## Project status

Following the 8-phase roadmap in `docs/proposal/`:

- [x] Phase 1 — Literature survey, dataset & gap statement
- [x] Phase 2 — Real 4-site data as federated clients (replaced the earlier simulated split)
- [x] Phase 3 — Baselines: each hospital alone, all data pooled
- [x] Phase 4 — Federated Learning (FedAvg via Flower)
- [x] Phase 5 — Personalization (FedProx + fine-tuning)
- [x] Phase 6 — SHAP explainability
- [x] Phase 7 — FastAPI backend + React dashboard (incl. live training)
- [ ] Phase 8 — Results compilation, research paper draft & presentation

The project originally split Cleveland alone (297 patients) into 5 simulated
hospitals. That setup and all its results are preserved in
[`docs/archive/README-cleveland-only.md`](docs/archive/README-cleveland-only.md);
**none of its numbers apply to the current results.**

## Data: four real UCI Heart Disease sites

**Source.** UCI Machine Learning Repository, *Heart Disease* (Janosi,
Steinbrunn, Pfisterer & Detrano, 1988), CC BY 4.0,
[doi:10.24432/C52P4X](https://doi.org/10.24432/C52P4X). As the data's
authors request, the principal investigators are: Andras Janosi, M.D.
(Hungarian Institute of Cardiology, Budapest); William Steinbrunn, M.D.
(University Hospital, Zurich); Matthias Pfisterer, M.D. (University
Hospital, Basel); Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long
Beach, and Cleveland Clinic Foundation).

**Cleaning ("Option A") -- [`src/data/multisite.py`](src/data/multisite.py),
report: [`experiments/phase2_site_report.py`](experiments/phase2_site_report.py).**
Every figure below was counted from the raw UCI files, not recalled.

- Binary target (0 = no disease, 1-4 = disease).
- Physiologically impossible zeros are treated as missing values recorded
  as 0: cholesterol = 0 for **all 123** Swiss rows and 49 VA rows;
  resting BP = 0 for 1 VA row.
- One exact duplicate row dropped in each of Hungary and VA.
- Only the **8 features every site actually recorded** are used: age, sex,
  chest-pain type, resting BP, resting ECG, max heart rate, exercise
  angina, ST depression. Excluded because they are largely missing:
  `ca` (96-99% missing in Hungary, Switzerland, VA), `thal` (42-90%),
  `slope` (51-65% in Hungary and VA), cholesterol (100% in Switzerland),
  fasting blood sugar (61% in Switzerland).
- Complete cases on those 8 features; **no imputation**.

| Site (client) | Patients | Disease | No disease | Disease rate |
|---|---|---|---|---|
| Cleveland Clinic | 303 | 139 | 164 | 45.9% |
| Hungarian Institute of Cardiology | 291 | 105 | 186 | 36.1% |
| University Hospitals Zurich & Basel | 116 | 108 | 8 | 93.1% |
| VA Long Beach | 139 | 109 | 30 | 78.4% |
| **Total** | **849** | 461 | 388 | 54.3% |

Honest caveats: dropping `ca` and `thal` removes the two strongest
predictors in the Cleveland-only SHAP analysis, so accuracy is expected to
fall; Switzerland has only 8 patients without disease, so its per-site
accuracy is not meaningful on its own (a model that always predicts
"disease" scores 93% there) and is reported with balanced accuracy and
AUC (Results, section 3);
numeric features are standardized with statistics pooled across all 849
patients, as in the earlier phases (a per-site alternative is possible).

> **Reproducibility caveat -- read before comparing numbers.** Runs that use
> Flower's simulation engine (Ray) are **not bit-for-bit reproducible**:
> Ray returns client results in a varying order, which changes the
> floating-point summation order when the server averages the weights, and
> that can flip a borderline test prediction. Measured on this data: two
> identical FedAvg runs (model-init seed 7) differed by **2 test patients
> (0.94 pp)** across the Phase 4 and Phase 5 scripts. Treat differences of
> about **±1-2 patients / ±0.5-1 pp** between identical runs as run-to-run
> noise, not an effect. The Ray-free loop used for live training
> (`src/federated/rayfree_runner.py`) visits clients in a fixed order and
> is deterministic.

**Not used, and why.**
- *Statlog (Heart)* ([doi:10.24432/C57303](https://doi.org/10.24432/C57303))
  is not a fifth site: all 270 of its rows exactly match Cleveland rows.
- The widely shared Kaggle "Heart Failure Prediction" file (918 rows) is
  the same four sites plus Statlog minus 272 duplicates, but it has no
  site column (so it cannot define federated clients), keeps the 172
  cholesterol zeros as real values, and fills missing `slope` and
  Hungarian cholesterol values from no documented source.

**Relation to FLamby.** The FLamby benchmark's Fed-Heart-Disease dataset
(Ogier du Terrail et al., "FLamby: Datasets and Benchmarks for Cross-Silo
Federated Learning in Realistic Healthcare Settings", *Advances in Neural
Information Processing Systems 35*, NeurIPS 2022 Datasets and Benchmarks
Track, [arXiv:2210.04620](https://arxiv.org/abs/2210.04620)) uses the same
four sites as clients. Applying its preprocessing rule to the UCI files
reproduces its published 740 patients exactly (303 / 261 / 46 / 130). That
rule keeps 79 cholesterol = 0 values as real measurements (all 46 of its
Swiss patients and 33 VA patients), concentrated in the two sites with the
highest disease rates, which lets a model partly identify those sites from
a recording artifact. This project deliberately avoids that artifact by
treating those zeros as missing and not using cholesterol, which is why
its counts (849) differ from FLamby's.

## Results (4 real sites)

Protocol, identical everywhere: each site's patients are split 75/25 into
train/test (stratified, split seed 42), giving 636 training and **213 test
patients**; 20 federated rounds × 5 local epochs; FedProx μ = 0.1;
personalization = 10 epochs of local fine-tuning of the FedProx model. Five
model-initialization seeds (42, 1, 7, 123, 2024). Run on Google Colab
([`notebooks/phase6_colab.ipynb`](notebooks/phase6_colab.ipynb)) with
[`experiments/phase3_baselines.py`](experiments/phase3_baselines.py) to
[`phase6_shap_analysis.py`](experiments/phase6_shap_analysis.py); outputs in
`experiments/results/`, logs in `experiments/results/logs/`, and the
dashboard JSON generated from them by
[`export_dashboard_data.py`](experiments/export_dashboard_data.py). Every
number below is read from those files.

### 1. Federated vs alone vs pooled (5 seeds)

| Seed | Each hospital alone | Federated (FedAvg) | All data pooled | FedAvg vs alone |
|---|---|---|---|---|
| 42 | 78.4% | 83.1% | 82.2% | +4.7 pp |
| 1 | 80.3% | 81.7% | 80.8% | +1.4 pp |
| 7 | 79.8% | 82.2% | 82.2% | +2.4 pp |
| 123 | 77.9% | 82.6% | 81.7% | +4.7 pp |
| 2024 | 77.5% | 83.6% | 82.2% | +6.1 pp |
| **Mean** | **78.8%** | **82.6%** | **81.8%** | **+3.9 pp** |

All five settings, 5-seed summary:

| Setting | Mean | Std | Min | Max |
|---|---|---|---|---|
| Each hospital alone (Local NN) | 78.8% | 1.2% | 77.5% | 80.3% |
| FedAvg | 82.6% | 0.7% | 81.7% | 83.6% |
| FedProx (μ = 0.1) | 82.5% | 0.8% | 81.7% | 83.1% |
| Personalized (FedProx + fine-tuning) | 83.8% | 0.5% | 83.1% | 84.5% |
| All data pooled (Centralized NN) | 81.8% | 0.6% | 80.8% | 82.2% |

Why "training alone" and not "every hospital": at Switzerland, the model
trained on Switzerland's data alone scores higher than FedAvg (table 2).
The headline is about overall accuracy, where federated wins in every seed.

### 2. Per hospital (seed 42)

| Hospital | Test patients (disease : none) | Always "disease" | Alone | FedAvg | Pooled | Personalized | Pers. balanced acc. | Pers. AUC |
|---|---|---|---|---|---|---|---|---|
| Cleveland | 76 (35 : 41) | 53.9% | 73.7% | 82.9% | 80.3% | 81.6% | 81.5% | 0.904 |
| Hungary | 73 (26 : 47) | 64.4% | 79.5% | 84.9% | 83.6% | 84.9% | 81.4% | 0.897 |
| Switzerland | 29 (27 : 2) | 93.1% | 89.7% | 82.8% | 86.2% | 93.1% | 50.0% | 0.574 |
| VA Long Beach | 35 (27 : 8) | 77.1% | 77.1% | 80.0% | 80.0% | 82.9% | 66.9% | 0.718 |

"Always disease" is the accuracy of a model that ignores the patient and
predicts disease for everyone. Test class counts are derived from each
result file's accuracy, precision and recall, and checked to agree across
all models. At these test sizes one patient is 1.3-3.4 pp of a hospital's
accuracy.

### 3. What naive accuracy reporting misses: Switzerland

Switzerland's personalized model has the best accuracy of any hospital
(93.1%) and 100% recall -- and an **AUC of 0.574, close to chance**. Its test
set has 27 patients with heart disease and 2 without; the model predicts
"disease" for all 29, recognises 0 of the 2 healthy patients, and so scores
exactly what always saying "disease" scores. Balanced accuracy is 50.0%.
Reporting accuracy alone would make this look like the best result in the
project; the dashboard therefore shows AUC and balanced accuracy next to
every accuracy.

### 4. Personalization and the worst-served hospital

Personalization raises overall accuracy from 82.6% (FedAvg) to 83.8%
(5-seed means). In the seed-42 run it classifies a net 3 more test
patients correctly than FedAvg: Switzerland +3, VA Long Beach +1, Cleveland
-1, Hungary 0. Switzerland's +3 comes from the model that predicts "disease"
for everyone (above), so the overall gain is not evidence of better
discrimination.

At the hospital FedAvg serves worst (VA Long Beach in all 5 seeds),
personalization is mixed:

| Seed | Worst-served hospital | FedAvg | Personalized | Change |
|---|---|---|---|---|
| 42 | VA Long Beach | 80.0% | 82.9% | +2.9 pp |
| 1 | VA Long Beach | 74.3% | 77.1% | +2.9 pp |
| 7 | VA Long Beach | 74.3% | 74.3% | 0.0 pp |
| 123 | VA Long Beach | 77.1% | 74.3% | -2.9 pp |
| 2024 | VA Long Beach | 77.1% | 71.4% | -5.7 pp |

Better in 2 seeds, unchanged in 1, worse in 2; mean -0.57 pp. With 35 test
patients, 1 patient = 2.9 pp, so every change here is 0-2 patients. The
earlier simulated-split finding (personalization reliably helping the
worst-off hospital) **did not carry over** to the real sites.

### 5. Explainability (SHAP)

Top 5 features by mean |SHAP value|:

| Rank | Random Forest, all data pooled (TreeExplainer) | VA Long Beach personalized NN (KernelExplainer) |
|---|---|---|
| 1 | `cp_4` asymptomatic chest pain (0.113) | `cp_4` (0.069) |
| 2 | `oldpeak` exercise ST depression (0.069) | `oldpeak` (0.053) |
| 3 | `exang` exercise-induced angina (0.068) | `thalach` (0.040) |
| 4 | `thalach` max heart rate (0.065) | `exang` (0.036) |
| 5 | `sex` (0.059) | `sex` (0.035) |

The NN analysis targets the worst-served hospital from Phase 5 (VA Long
Beach). Two different model types agree on the same five features, all
standard exercise-test findings in the heart disease literature. `ca` and
`thal`, the strongest predictors in the earlier Cleveland-only analysis, are
not available here (largely missing at three sites). This is a plausibility
check by non-experts, not clinical validation.

### 6. Patients the models split on

The dashboard runs all four hospitals' personalized models on each of the
213 test patients, in the browser. They disagree (at the 0.5 threshold) on
**99 of 213 (46.5%)**. Switzerland's model calls 89.7% of all patients
diseased (the others: 49.8-62.9%) and is the lone or minority vote in 50 of
the 99 splits -- the same class-imbalance effect as in section 3, seen from
another angle.

## Dashboard

`src/frontend/` -- Vite + React, 6 screens plus a story mode. Every number is
read from the exported JSON or computed live in the browser; nothing is
hard-coded.

| Screen | What it shows |
|---|---|
| **Overview** | The headline, per-seed chart (alone → federated, pooled marker), the Switzerland accuracy story, the mixed personalization result, per-hospital table with always-"disease" baseline, balanced accuracy and AUC |
| **Problem** | Barrier chain (privacy laws → data silos → non-IID → clinician mistrust), each with evidence from this data |
| **Hospitals** | Site sizes, disease split, demographics |
| **Experiment Comparison** | Tabs: headline; **training replay** (recorded run, or **run it live** on the server); **fairness vs accuracy** (overall vs worst-hospital metric per setting and seed, plus live runs); per hospital (accuracy / balanced accuracy / AUC); 5-seed mean ± std; worst-served hospital |
| **Explainability** | Tabs: sample patients (real SHAP); **try a prediction** (all 4 models in the browser, hand-off from any test patient); **patients the models split on** (all 4 models on all 213 test patients, live); global SHAP importance |
| **Roadmap** | 8-phase timeline with each phase's result |
| **Present** (`/present`) | 6 full-screen slides: problem, non-IID, training replay, headline, Switzerland, roadmap |

### In-browser inference

The 4 saved personalized models run **directly in the browser** (no torch,
no backend), so predictions and the divergence explorer work on the static
site.

- [`experiments/export_model_weights.py`](experiments/export_model_weights.py)
  reads the `.pt` checkpoints without torch (a small unpickler over the raw
  float32 buffers) and writes `dashboard_data/model_weights.json`: 4 models ×
  369 parameters, the preprocessing constants, each hospital's average
  encoded patient (aggregate means, no individual records) and the slider
  bounds. It refuses to write unless its numpy forward pass reproduces the
  real torch outputs.
- [`src/frontend/src/lib/inference.js`](src/frontend/src/lib/inference.js)
  re-implements the encoding and `HeartDiseaseNet`'s forward pass.
- **Self-check on every load:** 856 reference predictions (852 torch
  predictions = 213 test patients × 4 models, from
  [`export_test_patients.py`](experiments/export_test_patients.py), plus 4
  SHAP sample patients), each within its own rounding tolerance. If any
  fails, the browser prediction features switch off.

The what-if explorer's per-feature bars are an **approximation, not SHAP**
(change in risk if one field is replaced by the hospital average), and are
labelled as such. Real SHAP values are on the Sample patients tab.

### Live training in the dashboard

With a live-training server configured (`VITE_LIVE_API_URL`), the training
replay and the fairness-vs-accuracy tabs can start a **real** 20-round FedAvg
or FedProx run on the 4 sites and draw it round by round as it streams in.
For seed 42 the dashboard compares the live run with the recorded one
(Ray-free runs are deterministic, so all 80 round-by-hospital accuracies
should match exactly). Without a server configured, those panels say so and
the recorded runs remain available.

**Deploying the live server** (free Render instance, measured to fit -- see
the feasibility table below):

1. Render → New → Blueprint → this repository. [`render.yaml`](render.yaml)
   installs CPU-only torch + [`requirements-live.txt`](requirements-live.txt),
   downloads the four UCI site files at build time, and allows CORS from
   `https://teena2812.github.io` (`ALLOWED_ORIGINS`).
2. Check `https://<service>.onrender.com/api/health`.
3. GitHub → Settings → Secrets and variables → Actions → Variables: add
   `LIVE_API_URL` = `https://<service>.onrender.com/api`, then re-run the
   Pages deploy. The free instance sleeps after 15 minutes idle; the first
   request shows a wake-up countdown (about a minute).

## Backend

Single FastAPI app, `src/backend/`:

| File | Role |
|---|---|
| [`main.py`](src/backend/main.py) | App setup, CORS, `/api/health`, `POST /api/predict` |
| [`data_routes.py`](src/backend/data_routes.py) | GET endpoints serving the dashboard JSON as-is (incl. model weights, sample and test patients) |
| [`live_training.py`](src/backend/live_training.py) | Live federated training over SSE (below) |
| [`inference.py`](src/backend/inference.py) | `POST /api/predict`: load checkpoint → forward pass → single-instance SHAP. No training code |
| [`schemas.py`](src/backend/schemas.py) | Pydantic request/response models with clinical sanity bounds |

torch/shap are imported lazily: without them every data endpoint still
works, `/api/health` reports `live_prediction_available: false`, and
`/api/predict` and `/api/train` return `503` instead of crashing.

Run: `venv\Scripts\uvicorn src.backend.main:app --reload` (docs at `/docs`).

### Live training over SSE

[`src/backend/live_training.py`](src/backend/live_training.py) runs one real
federated training on the 4 sites and streams it to the browser with
**Server-Sent Events** (progress only flows server -> browser, and
`EventSource` reconnects on its own):

| Endpoint | Purpose |
|---|---|
| `POST /api/train` | start a run (`algorithm`: fedavg / fedprox, `rounds` <= 20, `personalize`, `seed`) -> `202 {run_id, events_url}`; `409` if one is already running |
| `GET /api/train/{run_id}/events` | SSE stream: `start` (config, per-site sizes), one `round` per round (per-hospital metrics + test-size-weighted accuracy), optional `personalized`, then `done` or `error` |
| `GET /api/train/{run_id}` | JSON snapshot of all events so far |
| `GET /api/train/status` | whether a run is in progress |

It uses the Ray-free loop ([`src/federated/rayfree_runner.py`](src/federated/rayfree_runner.py))
because Flower's Ray engine does not fit a 512 MB free instance (see the
measurements below). One run at a time; events are numbered so a
reconnecting browser resumes with `Last-Event-ID` instead of restarting
training. Deploy dependencies: [`requirements-live.txt`](requirements-live.txt)
(no Ray, no shap). Tests: [`tests/test_live_training.py`](tests/test_live_training.py)
(streaming contract, runs without torch) and
[`tests/live_training_e2e.py`](tests/live_training_e2e.py) (real server,
run by [`.github/workflows/live-training-check.yml`](.github/workflows/live-training-check.yml)
inside a 512 MB / 0.1 CPU container, comparing every streamed number with
the Flower experiment outputs).

### Live-training feasibility (measured 2026-09-26)

A throwaway GitHub Actions spike ran one real 20-round federated training on
this 4-site data inside a container limited to **512 MB RAM (no swap) and
0.1 CPU** -- the same as Render's free web-service instance -- and without
limits for reference (torch 2.14.0+cpu, flwr 1.38.0, ray 2.55.1, Python 3.11):

| Run (20 rounds, 4 clients) | Limits | Outcome | Wall time | Peak memory |
|---|---|---|---|---|
| Flower `run_simulation` (Ray) -- the project's runners | 512 MB / 0.1 CPU | **OOM-killed in round 1** | died at ~52 s | > 512 MB |
| Flower `run_simulation` (Ray) | none | completed | 21 s | 2,447 MB |
| Same client code + Flower's `aggregate()`, no Ray | 512 MB / 0.1 CPU | **completed** | ~61 s (training 25-28 s) | ~320 MB |
| Same, no Ray | none | completed | 4 s | ~318 MB |

The Ray-free loop reproduced the Flower runs' per-round, per-hospital
accuracies **80/80 identically** for both FedAvg and FedProx. Flower 1.38
logs that `run_simulation` is deprecated, so `flwr` is now pinned in
`requirements.txt`.

## Live dashboard (GitHub Pages)

**https://teena2812.github.io/PBL-5/** -- deployed by
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
on every push to `master`.

The site is fully static: every screen reads the committed
`experiments/results/dashboard_data/*.json` (copied into the build by
`src/frontend/scripts/copy-dashboard-data.mjs`), predictions run in the
browser, and URLs are hash-based (`.../PBL-5/#/present`) because GitHub
Pages can't route deep links. Local static build:

```bash
cd src/frontend && VITE_STATIC_DATA=true VITE_ROUTER=hash VITE_BASE=/PBL-5/ npm run build:static
```

CI: [`frontend-check.yml`](.github/workflows/frontend-check.yml) lints and
builds the frontend on every push;
[`export-model-artifacts.yml`](.github/workflows/export-model-artifacts.yml)
regenerates the torch-dependent artifacts (sample and test patients).

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
cd src/frontend && npm install && npm run dev
```

> **Environment note.** On the development machine, Windows **Smart App
> Control** blocks native ML library DLLs (torch, scikit-learn, shap) and
> cannot be turned off without reinstalling Windows. Experiments therefore
> run on **Google Colab** (or GitHub Actions); the local machine runs the
> frontend, the pandas-only export scripts and git. `pandas` and
> `matplotlib` are pinned in `requirements.txt` to the versions used for
> every committed result, and `vite` is pinned to 5.x (Vite 8's native
> bundler hits the same block).

## Running the experiments on Google Colab

Open [`notebooks/phase6_colab.ipynb`](notebooks/phase6_colab.ipynb) in Colab
(`https://colab.research.google.com/github/Teena2812/PBL-5/blob/master/notebooks/phase6_colab.ipynb`),
then Runtime → Run all. It clones the repo, installs `requirements.txt`,
runs Phases 3-6 on the 4-site data and saves `multisite_results.zip`. No GPU
needed. Phase 2's site report runs locally:
`venv\Scripts\python experiments\phase2_site_report.py`.

## Folder structure

```
PBL-5/
├── docs/proposal/        ← approved planning docs (problem def, concept, lit survey, report)
├── docs/archive/         ← the earlier Cleveland-only README and results
├── research_papers/      ← literature collection + papers_index.csv
├── src/
│   ├── data/              ← 4-site loading and cleaning (multisite.py), preprocessing
│   ├── models/            ← sklearn baselines + HeartDiseaseNet
│   ├── federated/         ← Flower client/server/strategy, Ray-free loop, personalization
│   ├── explainability/    ← SHAP
│   ├── backend/           ← FastAPI app (data, prediction, live training)
│   └── frontend/          ← React dashboard
├── experiments/          ← phase scripts, exporters, results/ (CSVs, models, dashboard_data, logs)
├── notebooks/            ← Colab notebook
├── tests/                ← live-training tests
├── requirements.txt
└── README.md
```
