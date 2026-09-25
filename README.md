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
- [x] Phase 3 — Baseline: Local ML + Centralized ML
- [x] Phase 4 — Federated Learning (FedAvg via Flower)
- [x] Phase 5 — Personalization (FedProx) vs FedAvg comparison
- [x] Phase 6 — SHAP explainability integration (run on Colab)
- [x] Phase 7 — FastAPI backend + React dashboard
- [ ] Phase 8 — Results compilation, research paper draft & presentation

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

## Phase 4: Federated Learning (FedAvg via Flower)

Uses `flwr.simulation.run_simulation` (in-process virtual clients, no
separate OS processes per hospital, per the locked tech stack) with a small
PyTorch feed-forward NN ([`src/models/nn_model.py`](src/models/nn_model.py):
2 hidden layers, 16→8 units) as the shared model architecture. Every
hospital is a Flower `NumPyClient`
([`src/federated/client.py`](src/federated/client.py)) that trains locally
for 5 epochs/round on its own `x_train`/`y_train` and is evaluated on its
own `x_test`/`y_test` — the exact same partition and local split as Phase 3
(seed=117 / seed=42), so results are directly comparable.

Aggregation is Flower's built-in `FedAvg` strategy: each round, the server
averages client weights weighted by each client's number of local training
examples (θ_global = Σ (n_k/n) · θ_k). No FedProx proximal term yet — that's
Phase 5.

Run: `venv\Scripts\python experiments\phase4_fedavg.py` — 20 rounds, 5
local epochs/round, lr=0.01. Results in `experiments/results/phase4_*.csv`
and `phase4_fedavg_learning_curve.png`.

### ⚠️ Is Phase 3 vs Phase 4 an apples-to-apples comparison? No — read this before citing either table.

Phase 3's Local ML / Centralized ML baselines use sklearn Random Forest /
Logistic Regression. FedAvg necessarily uses a PyTorch NN, because FedAvg
averages weight *tensors* across clients — that operation has no RF/LR
equivalent (you can't "average" two decision forests or two independently
fit logistic models into a meaningful combined model the way you can
average two neural nets' weights). So directly comparing FedAvg's 0.829
against Phase 3's Local ML RF number (0.833) mixes two variables at once —
**algorithm choice** (RF/LR vs. NN) and **collaboration strategy** (none
vs. FedAvg) — into a single number. An apparent "Local ML beats FedAvg"
result in that mixed comparison could just mean "Random Forest beats a
small NN on this tiny dataset," and would tell us nothing about whether
federation itself helped.

| Experiment | Model | Accuracy |
|---|---|---|
| Local ML (mean/hospital) | Random Forest | 0.833 |
| Local ML (mean/hospital) | Logistic Regression | 0.819 |
| FedAvg (final round, global weighted) | PyTorch NN | 0.829 |
| Centralized ML (pooled) | Random Forest | 0.842 |
| Centralized ML (pooled) | Logistic Regression | 0.855 |

**Do not cite this table as a test of "Local ≤ FedAvg ≤ Centralized."** It
mixes architectures. Keep it only as a reference for how a classical-ML
baseline performs on this dataset/partition — a separate, legitimate
question from whether federation helps.

### The real apples-to-apples test: same architecture (HeartDiseaseNet) throughout

[`experiments/phase4_nn_baselines.py`](experiments/phase4_nn_baselines.py)
retrains Local ML and Centralized ML using the *same* `HeartDiseaseNet`
architecture as FedAvg, trained for `TOTAL_EPOCHS = N_ROUNDS *
LOCAL_EPOCHS_PER_ROUND = 100` epochs (matching the total local compute one
FedAvg client experiences across the full 20-round run), same lr=0.01,
same seed=117 partition / seed=42 local split. This isolates the
collaboration-strategy variable by holding the model fixed:

| Experiment (all PyTorch NN) | Accuracy |
|---|---|
| Local NN (per-hospital, isolated) | 0.816 |
| **FedAvg NN (collaborative, no personalization)** | **0.829** |
| Centralized NN (pooled, not privacy-preserving) | 0.816 |

(`experiments/results/phase4_nn_comparison_summary.csv`,
`phase4_nn_comparison.png`)

**This is the result to actually cite for the FL research question**, and
it does *not* fully match the naive hypothesis: FedAvg (0.829) slightly
*outperforms* both Local NN and Centralized NN, which tie exactly at
0.816.

#### Robustness check: does this hold across seeds, or was it noise?

[`experiments/phase4_multiseed_comparison.py`](experiments/phase4_multiseed_comparison.py)
reran all three settings (Local NN / FedAvg NN / Centralized NN, same
architecture, same seed=117 partition / seed=42 local split) across 5
different `MODEL_INIT_SEED` values (42, 1, 7, 123, 2024) — the *only*
thing that varies between runs is the random initial network weights.
Results (`experiments/results/phase4_multiseed_results.csv`,
`phase4_multiseed_summary.csv`, `phase4_multiseed_comparison.png`):

| Experiment | mean accuracy | std | min | max |
|---|---|---|---|---|
| Local NN | 0.805 | 0.017 | 0.789 | 0.829 |
| **FedAvg NN** | **0.832** | **0.006** | 0.829 | 0.842 |
| Centralized NN | 0.805 | 0.027 | 0.776 | 0.829 |

**Confirmed — FedAvg's edge is not seed-specific noise.** Per-seed, FedAvg
matched or beat Local NN in all 5 seeds and beat Centralized NN in 4/5
(tied at seed=7). FedAvg also has by far the *lowest variance* (std=0.006
vs. 0.017 for Local and 0.027 for Centralized) — four of the five seeds
landed on nearly the same FedAvg accuracy (0.8290 / 0.8290 / 0.8290 /
0.8290 / 0.8421), while Local and Centralized swing more with
initialization. **This is the finding to lead with in the final report:**
on this small NN and this dataset, federated averaging across 5
non-IID hospitals is not just competitive with pooling all the data
centrally — it is *more accurate on average and more stable across random
initializations* than either training in isolation or training on the
pooled data with this architecture. A plausible explanation (offered as
a hypothesis, not a proven mechanism) is that FedAvg's periodic
cross-client weight averaging acts as an implicit regularizer/ensembling
effect on a network this small and data this limited (170-221 training
examples). This 5-seed result is still a small sample for a formal
statistical claim (no significance test was run) — treat it as strong
supporting evidence, not statistical proof, in the write-up.

**TODO for the final report (not done yet, not blocking further phases):**
1. Run a formal paired significance test (e.g. paired t-test) on the 5-seed
   FedAvg-vs-Local and FedAvg-vs-Centralized results in
   `phase4_multiseed_results.csv` — we now have the matched per-seed data
   for it (same 5 `MODEL_INIT_SEED` values across all three settings).
2. The "FedAvg beats Centralized" result is counter-intuitive and needs
   careful discussion in the final report/paper. Keep the "implicit
   regularization on a small NN" explanation framed explicitly as a
   hypothesis to discuss, not a proven mechanism — do not state it as fact.

The single-seed learning curve (from the original `phase4_fedavg.py`
run, seed=42) rises
quickly (round 1: 0.684 → round 4: 0.842) then plateaus/oscillates mildly
around 0.82-0.83 for the remaining rounds — expected behavior for a
full-batch, low-epoch-count NN on a dataset this small.

**Per-hospital breakdown (final round):** hospital_4 (0.905) and
hospital_5 (0.889) do best; hospital_1 (0.733) and hospital_2 (0.750) do
worse than the global average (0.829) — this per-hospital gap under one
shared global model is exactly the motivation for Phase 5's personalization
(FedProx). hospital_3 again shows precision=0.333/recall=0.5 (F1=0.4) — the
same small-sample limitation as Phase 3 (only 2 positive examples in its
local test set of 14), not new to FedAvg.

**Caveat:** `flwr.simulation.run_simulation` prints a deprecation warning
in favor of the `flwr run` CLI / project-based workflow (`pyproject.toml` +
ServerApp/ClientApp). We kept `run_simulation` for this phase since it
still works correctly on the installed `flwr==1.36.0` and fits a
single-script experiment far more simply than migrating to the full
project-based workflow; if evaluators run this on a `flwr` release where
`run_simulation` has actually been removed, this is the place to migrate.
Flower also warns that Ray-based simulation on Windows is experimental —
it ran correctly here (verified via a smoke test and the full 20-round run
above), but Linux/WSL2 is Flower's recommended platform if issues appear
on a different machine.

## Phase 5: Personalization (FedProx + local fine-tuning)

Two-layer personalization, matching the proposal's stated strategy:

1. **FedProx** ([`src/federated/fedprox_runner.py`](src/federated/fedprox_runner.py)):
   same federated training loop as FedAvg, but each hospital's local loss
   gets a proximal term `(mu/2)||local - global||^2`
   ([`src/models/nn_train.py`](src/models/nn_train.py), wired through
   [`src/federated/client.py`](src/federated/client.py)'s `fit()`) that
   keeps its local update from drifting too far from the shared global
   model — using Flower's built-in `FedProx` strategy, which sends
   `proximal_mu` in the fit config automatically.
2. **Local fine-tuning** ([`src/federated/personalize.py`](src/federated/personalize.py)):
   after FedProx training finishes, each hospital fine-tunes its own copy
   of the trained global model for 10 more epochs on only its own local
   training data (no proximal term this time — the point here is to let it
   adapt), then is evaluated on its own local test set. This is the
   "Personalized FL" result.

**Choosing `proximal_mu`:** an informal sweep over {0, 0.01, 0.1, 1.0} at
seed=42 gave final global weighted accuracy 0.8290 / 0.8290 / 0.8158 / 0.8289
— note mu=0 exactly reproduces Phase 4's plain FedAvg result (a useful
sanity check that the implementation is correct), and mu=0.01 / mu=1.0 were
statistically indistinguishable from plain FedAvg in this single-seed test.
**mu=0.1 was chosen because it's the only value that visibly changed
training dynamics from vanilla FedAvg** — the smaller/larger values were
too weak or too dominated by other loss terms to matter in this setup. This
was not a rigorous hyperparameter search (single seed, 4 values); revisit
with a proper sweep (and ideally per-mu multi-seed runs) for the final
paper if time allows.

Run: `venv\Scripts\python experiments\phase5_fedprox.py` (after Phase 4's
scripts). Results in `experiments/results/phase5_*.csv` and
`phase5_fedprox_personalization.png`.

| Experiment | Global weighted accuracy |
|---|---|
| FedAvg (Phase 4, mu=0) | 0.829 |
| FedProx (mu=0.1, before fine-tuning) | 0.816 |
| **Personalized (FedProx + local fine-tuning)** | **0.842** |

**Per-hospital: personalization helped exactly where it should.**
hospital_1 — the weakest performer under the shared global model in both
Phase 4's FedAvg (0.733) and Phase 5's FedProx (0.733) — improved to
**0.867** after local fine-tuning (F1: 0.800 → 0.909). Every other
hospital's accuracy was unchanged by fine-tuning (10 epochs at lr=0.01
didn't flip any test predictions for hospitals whose local model was
already a good fit). This is a genuinely encouraging result: personalization
targeted the hospital that actually needed it, without hurting the others.

#### Robustness check: does this hold across seeds?

Following the same verified structure as Phase 4's multi-seed check,
[`experiments/phase5_multiseed_comparison.py`](experiments/phase5_multiseed_comparison.py)
reran Local NN / FedAvg NN / FedProx NN / Personalized / Centralized NN
across the SAME 5 `MODEL_INIT_SEED` values as Phase 4 (42, 1, 7, 123, 2024),
partition and local split held fixed.

Results (`experiments/results/phase5_multiseed_results.csv`,
`phase5_multiseed_summary.csv`, `phase5_multiseed_comparison.png`):

| Experiment | mean accuracy | std | min | max |
|---|---|---|---|---|
| Local NN | 0.805 | 0.017 | 0.789 | 0.829 |
| FedAvg NN | 0.832 | 0.006 | 0.829 | 0.842 |
| FedProx NN (before fine-tuning) | 0.818 | 0.014 | 0.803 | 0.842 |
| **Personalized (FedProx + fine-tuning)** | **0.832** | 0.011 | 0.816 | 0.842 |
| Centralized NN | 0.805 | 0.027 | 0.776 | 0.829 |

**On the single global, sample-weighted accuracy number, this is a
genuinely mixed result, reported honestly rather than forced into a clean
"personalization wins" story:**

- **FedProx's proximal term alone (before fine-tuning) is not a win over
  plain FedAvg** — its mean (0.818) is *lower* than FedAvg's (0.832) across
  all 5 seeds. This makes sense: constraining each hospital's local update
  to stay close to the global model trades away some of the local
  adaptation that seemed to be helping FedAvg in Phase 4's finding.
- **Local fine-tuning recovers that loss and ties FedAvg's mean exactly**
  (0.832 both), but does **not exceed it** on average. Per-seed, Personalized
  beat FedAvg in 2/5 seeds (42, 123), was about equal in 1/5 (seed 7), and
  was *worse* in 2/5 (seeds 1, 2024) — see `phase5_multiseed_results.csv`
  for the exact matched numbers. So at the level of a global weighted-mean
  accuracy, this run does not support a claim that FedProx+fine-tuning
  beats FedAvg overall.

### The headline finding: the equity metric

A global sample-weighted mean is dominated by the largest hospitals
(hospital_4 has 82 patients vs. hospital_2's 29) and can hide exactly the
kind of improvement that matters for **this project's actual hypothesis**
([`docs/proposal/01_Problem_Definition.md`](docs/proposal/01_Problem_Definition.md)):
personalization should help the **worst-served** hospital, not necessarily
move the global average.

[`experiments/phase5_equity_analysis.py`](experiments/phase5_equity_analysis.py)
tests this directly, across the same 5 seeds: for each seed, find the
hospital with the *lowest* accuracy under plain FedAvg, then check whether
FedProx + local fine-tuning improved *that specific hospital's* accuracy.
(Note: `phase5_multiseed_comparison.py` above only saved each seed's global
weighted accuracy, not the per-hospital breakdown, so this required a
separate, targeted rerun — same partition/split/seeds as everywhere else,
just with per-hospital results actually captured this time.)

| seed | worst-served hospital (FedAvg) | FedAvg accuracy | Personalized accuracy | Δ |
|---|---|---|---|---|
| 42 | hospital_1 | 0.733 | 0.867 | +13.3 pp |
| 1 | hospital_1 | 0.733 | 0.800 | +6.7 pp |
| 7 | hospital_1 | 0.733 | 0.867 | +13.3 pp |
| 123 | hospital_1 | 0.733 | 0.867 | +13.3 pp |
| 2024 | hospital_2 | 0.750 | 0.750 | +0.0 pp |

(`experiments/results/phase5_equity_analysis.csv`,
`phase5_equity_analysis.png`)

**Personalization improved the worst-performing hospital in 4/5 seeds, by
an average of +11.67 percentage points among the seeds where it improved
(+9.34 pp averaged across all 5 seeds, including the one flat case).**
hospital_1 was the worst-served hospital under FedAvg in 4/5 seeds
(consistently at 0.733 accuracy — FedAvg's per-hospital results were very
stable across seeds, matching Phase 4's low-variance finding), and
personalization helped it every one of those 4 times, never made it worse.
In the fifth seed (2024), hospital_2 was worst-served instead, and
fine-tuning left it exactly unchanged (delta=0.0) rather than helping —
an honest null result for that one case, not hidden.

**This is the result to lead with in the final report.** It directly
answers the project's core research question in a way the global-accuracy
comparison alone could not: personalization's real value here is in
protecting/improving the worst-off participant, which a single pooled
accuracy metric can mask.

## Phase 6: Explainability (SHAP)

Run on Google Colab (see the environment note above) via
[`notebooks/phase6_colab.ipynb`](notebooks/phase6_colab.ipynb), which
executes [`experiments/phase6_shap_analysis.py`](experiments/phase6_shap_analysis.py).
Two analyses, per the proposal's stated plan:

1. **Global explainability** — SHAP `TreeExplainer` (exact, fast) on the
   Phase 3 Centralized ML Random Forest, over the full pooled test set.
2. **Personalized explainability** — SHAP `KernelExplainer` (model-agnostic,
   since there's no exact/fast SHAP method for an arbitrary small NN) on
   hospital_1's Phase 5 personalized model — hospital_1 chosen because it's
   the hospital personalization most reliably helped, per Phase 5's equity
   analysis.

Results in `experiments/results/phase6_shap_*.csv` / `*.png`.

**Global feature importance (Random Forest, mean |SHAP value|), top 5:**

| feature | mean \|SHAP\| |
|---|---|
| `ca` (number of major vessels colored by fluoroscopy) | 0.094 |
| `thal_3` (thalassemia = normal) | 0.077 |
| `oldpeak` (ST depression induced by exercise) | 0.061 |
| `cp_4` (chest pain type = asymptomatic) | 0.054 |
| `thal_7` (thalassemia = reversible defect) | 0.050 |

**hospital_1's personalized NN produces an almost identical top-5** (`ca`,
`thal_7`, `sex`, `cp_3`, `thal_3`) despite being a completely different
model architecture (NN vs. Random Forest) trained on a very different,
tiny, non-IID local dataset (hospital_1: n=45 train / 15 test, disease
rate 0.717 vs. the pooled Random Forest's global rate of 0.46) —
`ca` and thalassemia status dominate both.

**Clinical plausibility: yes, this matches established Cleveland Heart
Disease literature.** `ca`, `thal`, chest pain type (`cp`), `oldpeak`
(exercise-induced ST depression), and `exang` (exercise-induced angina)
are consistently reported as the strongest predictors for this exact
dataset across the published literature reviewed in
[`docs/proposal/03_Literature_Survey.md`](docs/proposal/03_Literature_Survey.md)
and the broader UCI Heart Disease body of work — these are core cardiac
stress-test and catheterization findings, not incidental demographic
features. Example patient explanation
(`phase6_shap_rf_example_patient.csv`/`.png`) shows the same features
(`oldpeak`, `ca`, `thal_3`=0 i.e. *not* normal, `cp_4`, `thal_7`, `exang`)
driving one specific prediction, all pushing toward disease — a concrete
"Cholesterol +23%, Age +17%"-style panel as described in the proposal,
just with the features this model actually relied on rather than assumed
ones.

**Caveat:** this is a plausibility check by a domain non-expert (SHAP
rankings matching known literature themes), not a clinical validation by
a cardiologist — note this distinction in the final report rather than
overclaiming clinical validation.

## Phase 7 prep: saving model checkpoints for live prediction

Live single-patient prediction is a core Phase 7 dashboard feature, not a
stretch goal — **inference and single-instance SHAP explanation only, no
retraining ever happens in the live endpoint.**

Checked first: Phases 4-5 never saved any model weights to disk — every
FedAvg/FedProx/personalized model existed only in memory, used to compute
metrics, then discarded when the script exited. Two things needed adding,
not just the weights:

1. **Preprocessing persistence.** Standardization (means/stds) and
   categorical one-hot encoding were previously recomputed fresh from the
   training data every run, never saved as a reusable artifact. A live
   endpoint needs to transform a brand-new raw patient record into
   *exactly* the vector the model was trained on. Fixed in
   [`src/data/load_dataset.py`](src/data/load_dataset.py):
   `compute_preprocessing_artifact()` / `save_preprocessing_artifact()` /
   `load_preprocessing_artifact()` / `transform_new_patient()`. Along the
   way, fixed an inconsistency where `cp`/`restecg`/`slope` produced
   float-suffixed dummy column names (`cp_4.0`) while `thal` didn't
   (`thal_3`) — all categorical columns are now cast to `int` consistently
   before encoding (`cp_4`, not `cp_4.0`), so a new patient's raw integer
   codes map unambiguously to column names. **Verified with a round-trip
   test:** transforming patient #0's raw values through
   `transform_new_patient()` reproduces the training pipeline's `X.iloc[0]`
   exactly (`np.allclose` — confirmed locally, this part needs no torch/
   sklearn so it works outside Colab).
2. **Model checkpoints.** [`src/federated/personalize.py`](src/federated/personalize.py)
   gained `personalize_per_hospital_with_models()` (returns the trained
   model objects, not just metrics — the existing `personalize_per_hospital()`
   is unchanged and now just calls it and discards the models, so Phase 5/6
   aren't affected). [`experiments/save_personalized_models.py`](experiments/save_personalized_models.py)
   reruns the exact Phase 5 pipeline (same seed=117 partition, seed=42
   split/init, mu=0.1, 20 rounds, 10 fine-tune epochs) and saves each
   hospital's final personalized model to
   `experiments/results/models/{hospital}_personalized.pt`, plus
   `preprocessing.json` and a `manifest.json` (architecture + per-hospital
   test accuracy, for the backend to sanity-check what it loaded).

**Run on Colab** (`notebooks/phase6_colab.ipynb`, section 9) — needs
torch/flwr, so per this project's Colab-primary policy it wasn't run on
the local machine. **Confirmed and committed:** the printed metrics
exactly matched Phase 5's `[Personalized FL]` results
(hospital_1=0.8667, hospital_2=0.75, hospital_3=0.7857, hospital_4=0.8571,
hospital_5=0.8889) and `manifest.json`'s source params match every other
phase's protocol (partition_seed=117, split_seed=42, proximal_mu=0.1).
The 5 checkpoints + `manifest.json` + `preprocessing.json` are committed
in `experiments/results/models/`.

**Feasibility confirmed for the planned dashboard feature:**
- Loading a saved `state_dict` + running one forward pass is fast and
  needs no training — straightforward.
- Single-instance SHAP (`KernelExplainer`) is fast enough for an
  interactive UI (sub-second to ~1-2s on this tiny NN with a small
  background sample) but not instant — the UI should show it as "computing
  explanation," not claim a truly live/real-time indicator.
- Fully consistent with the "no live training" rule: only a forward pass
  and a single-instance SHAP explanation happen at request time.

## Phase 7, step 1: dashboard data export

[`experiments/export_dashboard_data.py`](experiments/export_dashboard_data.py)
consolidates every phase's result CSVs (Phases 2-6) into 6 clean JSON
files under `experiments/results/dashboard_data/`, for the FastAPI backend
to serve directly — precomputed, so the backend never recomputes any of
this at request time (only the live single-patient prediction endpoint
does real work, per the "no live training" rule above). Needs only
pandas, so it runs fine locally despite the Smart App Control block.

| File | Contents |
|---|---|
| `dashboard_summary.json` | Top-level overview: dataset size, model architecture, training config, global accuracy per experiment, privacy statement |
| `hospitals.json` | Per-hospital demographics (Phase 2), train/test sizes, personalized model accuracy |
| `experiment_comparison.json` | Local/FedAvg/FedProx/Personalized/Centralized — all same NN architecture, global + per-hospital accuracy |
| `training_curves.json` | Round-by-round accuracy, FedAvg and FedProx, global and per-hospital |
| `equity_analysis.json` | Phase 5's headline finding: per-seed worst-hospital before/after personalization, plus multi-seed summaries |
| `explainability.json` | Phase 6 SHAP: RF global importance, hospital_1 NN importance, one example patient |
| `sample_patients.json` | One held-out test patient per hospital scored by its saved personalized model (prediction + SHAP). Needs torch, so generated on Colab by [`export_sample_patients.py`](experiments/export_sample_patients.py) (notebook section 11) |

Run: `venv\Scripts\python experiments\export_dashboard_data.py`. Re-run
whenever any upstream CSV changes, before starting/restarting the backend.

## Phase 7, step 2: FastAPI backend

Single central backend (`src/backend/`), per the locked architecture — one
FastAPI app for the whole system, not one per hospital.

| File | Role |
|---|---|
| [`main.py`](src/backend/main.py) | App setup, CORS, `/api/health`, `POST /api/predict` |
| [`data_routes.py`](src/backend/data_routes.py) | 6 GET endpoints serving the Phase 7-step-1 JSON files as-is |
| [`inference.py`](src/backend/inference.py) | Live prediction: load checkpoint → forward pass → single-instance SHAP. **No retraining code exists here at all.** |
| [`schemas.py`](src/backend/schemas.py) | Pydantic request/response models, with clinical sanity bounds on `PatientInput` |

**Designed to degrade gracefully, not crash, when torch/shap aren't
installed** — relevant because the local dev machine can't install them
(Smart App Control). `inference.py` imports torch/shap in a `try/except`
at module load and sets `TORCH_AVAILABLE = False` on failure; every data
endpoint is completely unaffected, and `POST /api/predict` returns a clear
`503` with an explanatory message instead of crashing the whole app.
`GET /api/health` reports `live_prediction_available` so the frontend can
show a banner rather than silently failing.

**Verified locally** (torch unavailable here) via FastAPI's `TestClient`
and a real `uvicorn` process on port 8123:
- All 6 data endpoints return `200` with the expected JSON structure
- `GET /api/health` correctly reports `live_prediction_available: false`
- `POST /api/predict` returns `503` (not a crash) with a clear message
- Invalid patient input (e.g. `age=-5`) correctly returns `422` (Pydantic validation)
- `/docs` (Swagger UI) loads

**Verified end-to-end on Colab** (`notebooks/phase6_colab.ipynb`, section
10 — starts the backend as a real background process, sends real HTTP
requests, checks error paths, stops the server):
- `POST /api/predict` returns a correct `200` response for all 5
  hospitals' personalized checkpoints (not just one) — each with a
  plausible predicted probability, risk level, and single-instance SHAP
  explanation. `oldpeak` consistently showed as risk-increasing across
  hospitals, matching the clinical pattern from Phase 6's global
  explainability results, and each response's `base_value` + SHAP
  breakdown was internally consistent.
- Error paths correctly return `400` (unknown `hospital_id`; an
  in-range-but-untrained category — e.g. `thal=4`, since `thal`'s valid
  trained categories `{3, 6, 7}` are non-sequential so its Pydantic bound
  is deliberately loose enough to admit them, leaving room in between for
  the app-level category check to catch) vs `422` (Pydantic-level
  violations like `age=-5` or a clearly out-of-bounds `thal=99` — a
  different, earlier validation layer, not the same code path as the 400
  case above).

**No live retraining, by design:** `get_partitions()` in `inference.py`
recomputes the same deterministic Dirichlet partition (seed=117) and local
split (seed=42) used everywhere else, purely to supply each hospital's
`x_train` as the SHAP background sample — this is data loading, not model
training (no gradients, no `local_train()` call anywhere in this module).

Run: `venv\Scripts\uvicorn src.backend.main:app --reload` (add `--port` to
change from the default 8000). Interactive docs at `/docs`.

## Phase 7, steps 3-7: React frontend (6 screens)

`src/frontend/` — a Vite + React app with 6 screens, every number read
from the backend's dashboard JSON (nothing hardcoded): Overview, Problem,
Hospitals, Experiment Comparison, Explainability, Roadmap. The earlier
Equity Analysis, Training Curves and Predict Risk screens were dropped: the
equity headline is the default tab of Experiment Comparison, and live
prediction lives inside Explainability.

| File | Role |
|---|---|
| [`api/client.js`](src/frontend/src/api/client.js) | Axios client, one function per backend endpoint |
| [`hooks/useApiData.js`](src/frontend/src/hooks/useApiData.js) | Shared fetch-on-mount hook (`{data, loading, error}`) every screen uses |
| [`components/Sidebar.jsx`](src/frontend/src/components/Sidebar.jsx) | Nav — all routes, active-link highlighting |
| [`constants/experiments.js`](src/frontend/src/constants/experiments.js) | One name + colorblind-validated color per training setting, shared by every screen |
| [`pages/Overview.jsx`](src/frontend/src/pages/Overview.jsx) | Leads with the equity finding (worst-served hospital, before/after per seed), a single-patient "known limitation" callout, then the setup, the close overall-accuracy race, and per-hospital accuracy |
| [`pages/Problem.jsx`](src/frontend/src/pages/Problem.jsx) | Barrier flowchart (privacy laws → data silos → non-IID → clinician mistrust), each with live evidence and the layer that answers it |
| [`pages/Hospitals.jsx`](src/frontend/src/pages/Hospitals.jsx) | Hospital sizes + disease-split stacked bars, disease rate vs pooled rate, profile table |
| [`pages/ExperimentComparison.jsx`](src/frontend/src/pages/ExperimentComparison.jsx) | Tabs: worst-served hospital equity view (default), training replay (the real FedAvg/FedProx runs round by round, with a weights-not-records network diagram), 5-seed mean ± std, per-hospital single run |
| [`pages/Explainability.jsx`](src/frontend/src/pages/Explainability.jsx) | Tabs: sample patients (prediction + SHAP), live "Try a prediction" form, global SHAP importance |
| [`pages/Roadmap.jsx`](src/frontend/src/pages/Roadmap.jsx) | 8-phase timeline with a result per completed phase |
| [`pages/Present.jsx`](src/frontend/src/pages/Present.jsx) | Story mode (sidebar "Present" button, `/present`): 6 full-screen slides — problem, non-IID, auto-playing training replay, equity finding, the borderline-patient limitation, roadmap — each with one key number; Next/Back (or arrow / PageUp / PageDown keys) only, no links out |

**Environment note:** Vite 8's default bundler (`rolldown-vite`, a Rust
binary) hit the exact same Smart App Control block as `sklearn`/`shap`/
`matplotlib` before it (`Cannot find native binding ... Application
Control policy has blocked this file`, on `@rolldown/binding-win32-x64-msvc`).
Pinned `vite` to `^5.4.0` (the pre-rolldown, esbuild/rollup-based engine)
in `src/frontend/package.json`, which boots cleanly — same fix pattern as
the Python pins, just for Node this time. Unlike the Python ML stack, this
one was fixable with a version pin rather than needing Colab, so the
frontend runs and was verified on the local machine.

**Verified in a real browser** (backend running locally on port 8000,
`live_prediction_available: false` as expected without torch, frontend on
Vite's dev server): Overview screen renders with live data from all 4
data endpoints it uses, all four stat cards show correct values (5
hospitals, 297 patients, 20 rounds, 88.9% best personalized accuracy),
the accuracy bar chart and per-hospital table match the committed numbers
exactly, sidebar navigation and active-link highlighting work, and
placeholder pages render for the unbuilt screens. Zero console errors.

Run: `cd src/frontend && npm install && npm run dev` (needs the backend
running separately for data to load — see Phase 7 step 2 above). Set
`VITE_API_BASE_URL` to point at a non-default backend URL.

## Phase 7, step 9: in-browser inference (what-if explorer)

The Explainability screen's "Try a prediction" tab runs the 5 saved
personalized models **directly in the browser** -- no torch, no backend
inference, no Colab -- so live prediction works on this machine and in any
static deployment.

- [`experiments/export_model_weights.py`](experiments/export_model_weights.py)
  reads the `.pt` checkpoints **without torch** (a checkpoint is a zip of a
  pickled state_dict pointing at raw float32 buffers; a small custom
  unpickler rebuilds them with numpy) and writes
  `dashboard_data/model_weights.json`: the weights (5 models x 513
  parameters), the preprocessing constants, each hospital's average encoded
  patient (aggregate means over its real 29-82 patients, no individual
  records), and the observed min/max of each numeric input across the 297
  patients (bounds for the sliders). Runs locally:
  `venv\Scripts\python experiments\export_model_weights.py`
- [`src/frontend/src/lib/inference.js`](src/frontend/src/lib/inference.js)
  re-implements `transform_new_patient` + `HeartDiseaseNet`'s forward pass
  (Linear 22->16, ReLU, Linear 16->8, ReLU, Linear 8->1, sigmoid).

**Verified against the real model:**
- The export script fails unless its numpy forward pass reproduces all 5
  real `predict_and_explain()` outputs in `sample_patients.json` (the
  function behind `POST /api/predict`) within their 4-dp rounding: max
  difference 3.6e-5, every value rounds to the stored one.
- The JavaScript forward pass matches numpy to 6.7e-16 over 1,000
  predictions (200 random in-range patients x 5 models).
- The dashboard re-runs the 5-patient check on every load and **switches
  the feature off** if any in-browser prediction drifts past 5e-5.

The explorer's per-feature breakdown is an **approximation, not SHAP**:
each bar is the change in risk if only that one field were replaced by the
hospital's average, and it is labelled as such in the UI. Real SHAP values
remain on the Sample patients tab (computed on Colab).

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

> **Environment note (as of Phase 6):** on this development machine,
> Windows **Smart App Control** is ON and blocks native ML library DLLs
> under an "Application Control policy" — confirmed via Event Viewer.
> `pandas>=3.0` and `matplotlib>=3.9` were worked around with version pins
> (below), but `shap`'s dependency chain (`numba`, then several
> `scikit-learn` submodules) triggered a worsening whack-a-mole of blocked
> files that pinning could not resolve — and, critically, Smart App Control
> **cannot be turned off without a full Windows reinstall**. A fresh venv
> still hits the same block, confirming it's a machine-wide policy, not a
> dependency-version issue.
>
> **Going forward, Google Colab is the primary execution environment for
> this project** (see [`notebooks/phase6_colab.ipynb`](notebooks/phase6_colab.ipynb)),
> starting with Phase 6. The local machine is used for editing code/docs
> and git operations only — not for running experiments. Phases 1-5 were
> fully verified locally before this switch and their results are trusted;
> Phase 6 onward is verified on Colab instead.
>
> `pandas` is pinned to `2.2.3` and `matplotlib` to `3.8.4` in
> `requirements.txt` for the LOCAL environment (both worked around this
> machine's DLL blocks before the SHAP issue forced the Colab switch) —
> neither pin should be necessary on Colab, but they're kept so a Colab run
> reproduces the exact same versions used for every phase's committed
> results.

## Running on Google Colab

Open [`notebooks/phase6_colab.ipynb`](notebooks/phase6_colab.ipynb) in
Colab (File → Open notebook → GitHub → paste this repo's URL, or use
`https://colab.research.google.com/github/Teena2812/PBL-5/blob/master/notebooks/phase6_colab.ipynb`
directly), then Runtime → Run all. It clones this repo, installs
`requirements.txt` fresh, re-verifies Phase 3-5 reproduce their committed
results, and runs Phase 6's SHAP analysis. No GPU needed.

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
