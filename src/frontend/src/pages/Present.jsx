import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  getDashboardSummary,
  getEquityAnalysis,
  getExperimentComparison,
  getHospitals,
  getTrainingCurves,
} from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import TrainingReplay from "../components/TrainingReplay";
import SeedComparisonChart from "../components/SeedComparisonChart";
import { fmtPct, hospitalLabel, hospitalsWhereAloneWins } from "../constants/experiments";
import { PHASES } from "../constants/roadmap";
import "./Present.css";

async function getPresentData() {
  const [summary, hospitals, equity, curves, comparison] = await Promise.all([
    getDashboardSummary(),
    getHospitals(),
    getEquityAnalysis(),
    getTrainingCurves(),
    getExperimentComparison(),
  ]);
  return { summary, hospitals: hospitals.hospitals, equity, curves, comparison };
}

// Slide order is fixed; navigation is Next/Back only (plus arrow / PageUp /
// PageDown keys, which presentation clickers send). No links inside slides,
// so nothing can navigate away mid-talk.
const SLIDES = [
  { id: "problem", render: (d) => <ProblemSlide {...d} /> },
  { id: "noniid", render: (d) => <NonIidSlide {...d} /> },
  { id: "replay", render: (d) => <ReplaySlide {...d} /> },
  { id: "headline", render: (d) => <HeadlineSlide {...d} /> },
  { id: "accuracy", render: (d) => <NaiveAccuracySlide {...d} /> },
  { id: "roadmap", render: () => <RoadmapSlide /> },
];

export default function Present() {
  const { data, loading, error } = useApiData(getPresentData, []);
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [isFullscreen, setIsFullscreen] = useState(Boolean(document.fullscreenElement));

  // 1-based slide number in the URL; anything invalid falls back to 1.
  const requested = Number.parseInt(searchParams.get("slide") ?? "1", 10);
  const index = Number.isFinite(requested) ? Math.min(Math.max(requested, 1), SLIDES.length) - 1 : 0;

  // Track the latest slide in a ref so several clicks/keys within one render
  // each count, instead of all reading the same stale index.
  const indexRef = useRef(index);
  useLayoutEffect(() => {
    indexRef.current = index;
  }, [index]);
  const step = useCallback(
    (delta) => {
      const clamped = Math.min(Math.max(indexRef.current + delta, 0), SLIDES.length - 1);
      if (clamped === indexRef.current) return;
      indexRef.current = clamped;
      setSearchParams({ slide: String(clamped + 1) }, { replace: true });
    },
    [setSearchParams]
  );

  useEffect(() => {
    const onKey = (e) => {
      if (e.target instanceof HTMLInputElement) return; // let sliders use arrows
      if (["ArrowRight", "PageDown"].includes(e.key)) {
        e.preventDefault();
        step(1);
      } else if (["ArrowLeft", "PageUp"].includes(e.key)) {
        e.preventDefault();
        step(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  useEffect(() => {
    const onChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      // Fullscreen can be refused (browser policy); the slides work without it.
    }
  };

  const exit = async () => {
    if (document.fullscreenElement) {
      try {
        await document.exitFullscreen();
      } catch {
        /* ignore */
      }
    }
    navigate("/");
  };

  const isFirst = index === 0;
  const isLast = index === SLIDES.length - 1;

  return (
    <div className="present">
      <header className="present-bar">
        <span className="present-brand">
          <span className="sidebar-brand-mark">FL</span>
          Personalized FL for heart-disease risk
        </span>
        <span className="present-bar-actions">
          <button type="button" className="present-ghost" onClick={toggleFullscreen}>
            {isFullscreen ? "Exit full screen" : "Full screen"}
          </button>
          <button type="button" className="present-ghost" onClick={exit}>
            ✕ Exit presentation
          </button>
        </span>
      </header>

      <main className="present-stage" aria-live="polite">
        {loading && <Loading />}
        {error && <ErrorState error={error} />}
        {data && (
          <section key={SLIDES[index].id} className="present-slide" aria-label={`Slide ${index + 1} of ${SLIDES.length}`}>
            {SLIDES[index].render(data)}
          </section>
        )}
      </main>

      <footer className="present-nav">
        <button type="button" className="button-secondary present-nav-btn" onClick={() => step(-1)} disabled={isFirst}>
          ← Back
        </button>
        <span className="present-progress" aria-label={`Slide ${index + 1} of ${SLIDES.length}`}>
          {SLIDES.map((s, i) => (
            <span key={s.id} className={"present-dot" + (i === index ? " active" : i < index ? " done" : "")} />
          ))}
          <span className="present-count">
            {index + 1} / {SLIDES.length}
          </span>
        </span>
        {isLast ? (
          <button type="button" className="button-primary present-nav-btn" onClick={exit}>
            Finish ✓
          </button>
        ) : (
          <button type="button" className="button-primary present-nav-btn" onClick={() => step(1)}>
            Next →
          </button>
        )}
      </footer>
    </div>
  );
}

function SlideHead({ step, title, children }) {
  return (
    <div className="slide-head">
      <span className="slide-step">{step}</span>
      <h1 className="slide-title">{title}</h1>
      {children && <p className="slide-lede">{children}</p>}
    </div>
  );
}

function KeyNumber({ value, label, tone = "primary" }) {
  return (
    <div className={`key-number key-number-${tone}`}>
      <span className="key-number-value">{value}</span>
      <span className="key-number-label">{label}</span>
    </div>
  );
}

/* 1. Problem */
function ProblemSlide({ summary, hospitals, equity }) {
  const sizes = hospitals.map((h) => h.n_patients);
  const h = equity.headline;
  const barriers = ["Privacy laws", "Data silos", "Non-IID patients", "Clinician mistrust"];
  return (
    <>
      <SlideHead step="The problem" title="Can hospitals learn together without sharing patient records?">
        A research simulation on {summary.dataset.n_patients} patients from {summary.n_hospitals} real hospitals (
        {summary.dataset.sites}; public UCI Heart Disease data).
      </SlideHead>
      <KeyNumber
        value={`${Math.min(...sizes)}–${Math.max(...sizes)}`}
        label={`patients per hospital — trained alone, the hospitals' models reach ${fmtPct(h.local_mean_accuracy)} overall accuracy`}
      />
      <ol className="slide-chain">
        {barriers.map((b, i) => (
          <li key={b}>
            <span className="slide-chain-index">{i + 1}</span>
            {b}
          </li>
        ))}
      </ol>
      <p className="slide-note">Each barrier feeds the next: privacy keeps data siloed, silos hide how different the patients are, and an unexplained model isn&apos;t trusted.</p>
    </>
  );
}

/* 2. Non-IID heterogeneity */
function NonIidSlide({ hospitals }) {
  const rates = hospitals.map((h) => h.disease_rate);
  const total = hospitals.reduce((s, h) => s + h.n_patients, 0);
  const pooled = hospitals.reduce((s, h) => s + Math.round(h.n_patients * h.disease_rate), 0) / total;
  return (
    <>
      <SlideHead step="Non-IID data" title="Every hospital sees a different patient mix">
        A single shared model has to serve all of them at once.
      </SlideHead>
      <KeyNumber
        value={`${fmtPct(Math.min(...rates))} – ${fmtPct(Math.max(...rates))}`}
        label={`disease rate across the ${hospitals.length} hospitals (all patients pooled: ${fmtPct(pooled)})`}
        tone="warning"
      />
      <div className="slide-bars">
        {hospitals.map((h) => (
          <div key={h.hospital} className="slide-bar-row">
            <span className="slide-bar-label">
              <strong>{hospitalLabel(h.hospital)}</strong> {h.n_patients} patients
            </span>
            <span className="slide-bar-track">
              <span className="slide-bar-fill" style={{ width: `${h.disease_rate * 100}%` }} />
              <span className="slide-bar-pooled" style={{ left: `${pooled * 100}%` }} />
            </span>
            <span className="slide-bar-value">{fmtPct(h.disease_rate)}</span>
          </div>
        ))}
        <span className="slide-note">Dark line = pooled rate; an even (IID) split would put every bar on it.</span>
      </div>
    </>
  );
}

/* 3. Training replay (auto-plays each time the slide is shown) */
function ReplaySlide({ curves, hospitals, summary }) {
  return (
    <>
      <SlideHead step="Federated training" title="They train together by sharing model weights, never records" />
      <KeyNumber value="0" label="patient records leave any hospital — only the model's weights travel" tone="success" />
      <div className="slide-replay">
        <TrainingReplay curves={curves} hospitals={hospitals} summary={summary} autoPlay compact />
      </div>
    </>
  );
}

/* 4. Headline finding */
function HeadlineSlide({ equity, comparison }) {
  const h = equity.headline;
  const aloneWins = hospitalsWhereAloneWins(comparison);
  return (
    <>
      <SlideHead step="Headline finding" title="Overall, training together beats training alone — without sharing records">
        Same model, same data, {h.n_seeds} random starting points. Pooling all records in one place (which privacy rules
        forbid) averaged {fmtPct(h.centralized_mean_accuracy)}.
      </SlideHead>
      <KeyNumber
        value={`+${h.delta_pp} pp`}
        label={`overall accuracy: federated ${fmtPct(h.fedavg_mean_accuracy)} vs alone ${fmtPct(h.local_mean_accuracy)} · better in ${h.seeds_fedavg_beats_local}/${h.n_seeds} seeds · matches or beats pooling in ${h.seeds_fedavg_matches_or_beats_centralized}/${h.n_seeds}`}
        tone="success"
      />
      <div className="slide-panel">
        <SeedComparisonChart seeds={h.per_seed} />
      </div>
      {aloneWins.length > 0 && (
        <p className="slide-note">
          <strong>Not at every hospital:</strong> in the seed-42 run,{" "}
          {aloneWins
            .map((x) => `${hospitalLabel(x.hospital)}'s own model scores ${fmtPct(x.local)} vs federated ${fmtPct(x.fedavg)}`)
            .join("; ")}{" "}
          &mdash; next slide.
        </p>
      )}
    </>
  );
}

/* 5. What naive accuracy reporting misses */
function NaiveAccuracySlide({ hospitals }) {
  const s = hospitals.find((x) => x.hospital === "switzerland");
  if (!s) return <SlideHead step="Accuracy" title="Per-hospital results" />;
  const rows = [
    { label: "Accuracy", value: s.personalized_accuracy, text: fmtPct(s.personalized_accuracy) },
    { label: "Always saying \u201cdisease\u201d", value: s.majority_class_accuracy, text: fmtPct(s.majority_class_accuracy) },
    { label: "Balanced accuracy", value: s.personalized_balanced_accuracy, text: fmtPct(s.personalized_balanced_accuracy) },
    { label: "AUC", value: s.personalized_auc, text: s.personalized_auc.toFixed(3) },
  ];
  return (
    <>
      <SlideHead step="What naive accuracy misses" title={`${hospitalLabel(s.hospital)}: ${fmtPct(s.personalized_accuracy)} accurate — and close to chance`}>
        Its test set: {s.test_disease} patients with heart disease, {s.test_no_disease} without. The model recognises{" "}
        {Math.round(s.personalized_specificity * s.test_no_disease)} of the {s.test_no_disease} healthy patients.
      </SlideHead>
      <KeyNumber value={s.personalized_auc.toFixed(3)} label="AUC — 0.5 is chance; the high accuracy comes from class imbalance" tone="warning" />
      <div className="slide-bars">
        {rows.map((r) => (
          <div key={r.label} className="slide-bar-row">
            <span className="slide-bar-label">
              <strong>{r.label}</strong>
            </span>
            <span className="slide-bar-track">
              <span className="slide-bar-fill risk" style={{ width: `${r.value * 100}%` }} />
              <span className="slide-bar-pooled" style={{ left: "50%" }} />
            </span>
            <span className="slide-bar-value">{r.text}</span>
          </div>
        ))}
        <span className="slide-note">Dark line = 0.5 (chance for balanced accuracy and AUC).</span>
      </div>
      <p className="slide-note">
        <strong>Why it matters:</strong> reporting accuracy alone would make this the best-performing hospital.
      </p>
    </>
  );
}

/* 6. Roadmap */
function RoadmapSlide() {
  const done = PHASES.filter((p) => p.done).length;
  return (
    <>
      <SlideHead step="Where it stands" title="Roadmap" />
      <KeyNumber value={`${done} of ${PHASES.length}`} label="phases complete" tone="success" />
      <ol className="slide-phases">
        {PHASES.map((p) => (
          <li key={p.n} className={p.done ? "done" : "pending"}>
            <span className="slide-phase-mark">{p.done ? "✓" : p.n}</span>
            <span>
              <strong>Phase {p.n}</strong> {p.title}
              {!p.done && <em> — upcoming</em>}
            </span>
          </li>
        ))}
      </ol>
    </>
  );
}
