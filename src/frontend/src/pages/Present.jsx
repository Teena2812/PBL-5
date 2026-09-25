import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  getDashboardSummary,
  getEquityAnalysis,
  getHospitals,
  getModelWeights,
  getSamplePatients,
  getTrainingCurves,
} from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import TrainingReplay from "../components/TrainingReplay";
import WorstServedChart from "../components/WorstServedChart";
import { fmtPct, fmtProb, hospitalLabel } from "../constants/experiments";
import { predict, verifyAgainstBackend } from "../lib/inference";
import { PHASES } from "../constants/roadmap";
import "./Present.css";

async function getPresentData() {
  const [summary, hospitals, equity, curves, samples, weights] = await Promise.all([
    getDashboardSummary(),
    getHospitals(),
    getEquityAnalysis(),
    getTrainingCurves(),
    getSamplePatients(),
    getModelWeights(),
  ]);
  return { summary, hospitals: hospitals.hospitals, equity, curves, samples: samples.samples, weights };
}

// Slide order is fixed; navigation is Next/Back only (plus arrow / PageUp /
// PageDown keys, which presentation clickers send). No links inside slides,
// so nothing can navigate away mid-talk.
const SLIDES = [
  { id: "problem", render: (d) => <ProblemSlide {...d} /> },
  { id: "noniid", render: (d) => <NonIidSlide {...d} /> },
  { id: "replay", render: (d) => <ReplaySlide {...d} /> },
  { id: "equity", render: (d) => <EquitySlide {...d} /> },
  { id: "limitation", render: (d) => <LimitationSlide {...d} /> },
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
function ProblemSlide({ summary, hospitals }) {
  const sizes = hospitals.map((h) => h.n_patients);
  const barriers = ["Privacy laws", "Data silos", "Non-IID patients", "Clinician mistrust"];
  return (
    <>
      <SlideHead step="The problem" title="Can hospitals learn together without sharing patient records?">
        A research simulation: {summary.dataset.n_patients} patients from the public {summary.dataset.name} dataset,
        split into {summary.n_hospitals} simulated hospitals.
      </SlideHead>
      <KeyNumber
        value={`${Math.min(...sizes)}–${Math.max(...sizes)}`}
        label="patients per hospital — too few for any one hospital to train a reliable model alone"
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

/* 4. Equity finding */
function EquitySlide({ equity }) {
  const seeds = equity.per_seed;
  const improved = seeds.filter((r) => r.delta > 0);
  const worse = seeds.filter((r) => r.delta < 0);
  const mean = (list) => list.reduce((s, r) => s + r.delta, 0) / list.length;
  return (
    <>
      <SlideHead step="Headline finding" title="Personalization lifts the hospital the shared model serves worst">
        Worst-served = the hospital FedAvg scored lowest, re-identified in each of {seeds.length} seeds.
      </SlideHead>
      <KeyNumber
        value={`+${(mean(improved) * 100).toFixed(1)} pp`}
        label={`in ${improved.length} of ${seeds.length} seeds · +${(mean(seeds) * 100).toFixed(1)} pp across all ${seeds.length} · ${worse.length === 0 ? "never made it worse" : `worse in ${worse.length}`}`}
        tone="success"
      />
      <div className="slide-panel">
        <WorstServedChart seeds={seeds} />
      </div>
    </>
  );
}

/* 5. Borderline patient / known limitation */
function LimitationSlide({ samples, weights, hospitals }) {
  const missed = samples.filter((s) => s.predicted_label !== s.actual_label);
  const verified = verifyAgainstBackend(weights).ok;
  if (missed.length !== 1) {
    return <SlideHead step="Known limitation" title="No single missed sample patient to show" />;
  }
  const s = missed[0];
  const own = hospitals.find((h) => h.hospital === s.hospital_id);
  const lowestRate = Math.min(...hospitals.map((h) => h.disease_rate));
  const scores = verified
    ? hospitals.map((h) => ({ id: h.hospital, rate: h.disease_rate, p: predict(s.patient, h.hospital, weights) }))
    : [];

  return (
    <>
      <SlideHead step="Known limitation" title="Personalization can cut both ways">
        One real test patient who <strong>had heart disease</strong>, scored by {hospitalLabel(s.hospital_id)}&apos;s own
        model &mdash; and by the other hospitals&apos; models.
      </SlideHead>
      <KeyNumber
        value={fmtProb(s.predicted_probability)}
        label={`${hospitalLabel(s.hospital_id)}'s model — just under the 50% threshold, so a miss`}
        tone="warning"
      />
      {verified && (
        <div className="slide-bars">
          {scores.map((r) => (
            <div key={r.id} className={"slide-bar-row" + (r.id === s.hospital_id ? " highlight" : "")}>
              <span className="slide-bar-label">
                <strong>{hospitalLabel(r.id)} model</strong> {fmtPct(r.rate)} disease rate
              </span>
              <span className="slide-bar-track">
                <span className="slide-bar-fill risk" style={{ width: `${r.p * 100}%` }} />
                <span className="slide-bar-pooled" style={{ left: "50%" }} />
              </span>
              <span className="slide-bar-value">{fmtProb(r.p)}</span>
            </div>
          ))}
          <span className="slide-note">Dark line = 50% decision threshold.</span>
        </div>
      )}
      <p className="slide-note">
        {own.disease_rate === lowestRate
          ? `${hospitalLabel(s.hospital_id)} has the lowest disease rate of the ${hospitals.length} (${fmtPct(own.disease_rate)}); fitting that population may pull risk down for patients unlike it. `
          : ""}
        <strong>A single-patient observation, not a measured effect.</strong>
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
