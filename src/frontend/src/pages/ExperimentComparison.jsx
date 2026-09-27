import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  BarChart,
  Bar,
  ComposedChart,
  Scatter,
  ErrorBar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList,
  Cell,
  ReferenceLine,
} from "recharts";
import {
  getDashboardSummary,
  getEquityAnalysis,
  getExperimentComparison,
  getHospitals,
  getTrainingCurves,
} from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import StatCard from "../components/StatCard";
import Tabs from "../components/Tabs";
import SeedComparisonChart from "../components/SeedComparisonChart";
import TradeoffExplorer from "../components/TradeoffExplorer";
import TrainingReplay from "../components/TrainingReplay";
import {
  EXPERIMENT_ORDER,
  EXPERIMENT_LABELS,
  EXPERIMENT_COLORS,
  MULTISEED_NAME_TO_KEY,
  fmtPct,
  hospitalLabel,
} from "../constants/experiments";

const TABS = [
  { id: "headline", label: "Federated vs alone vs pooled", badge: "Headline" },
  { id: "replay", label: "Training replay" },
  { id: "tradeoff", label: "Fairness vs accuracy" },
  { id: "per-hospital", label: "Per hospital" },
  { id: "multiseed", label: "5-seed mean ± std" },
  { id: "worst-served", label: "Worst-served hospital" },
];

const mutedNote = { color: "var(--color-text-muted)", fontSize: "0.85rem", marginTop: -4 };
const gridStroke = "#e2e8f0";
const hoverCursor = { fill: "rgba(148, 163, 184, 0.12)" };

function fmtPts(delta) {
  const pts = delta * 100;
  return `${pts >= 0 ? "+" : "−"}${Math.abs(pts).toFixed(1)} pp`;
}

export default function ExperimentComparison() {
  const equity = useApiData(getEquityAnalysis, []);
  const comparison = useApiData(getExperimentComparison, []);
  const hospitals = useApiData(getHospitals, []);
  const [searchParams, setSearchParams] = useSearchParams();

  const requested = searchParams.get("view");
  const active = TABS.some((t) => t.id === requested) ? requested : "headline";
  const setActive = (id) => setSearchParams(id === "headline" ? {} : { view: id }, { replace: true });

  if (equity.loading || comparison.loading || hospitals.loading) return <Loading />;
  for (const r of [equity, comparison, hospitals]) if (r.error) return <ErrorState error={r.error} />;
  const sites = hospitals.data.hospitals;

  return (
    <div>
      <div className="page-header">
        <h1>Experiment Comparison</h1>
        <p>
          Each hospital alone vs federated (FedAvg, FedProx, personalized) vs all data pooled &middot; the same model in
          every setting, {sites.length} real hospitals
        </p>
      </div>

      <Tabs tabs={TABS} active={active} onChange={setActive} label="Comparison views" />

      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {active === "headline" && <HeadlineView equity={equity.data} sites={sites} />}
        {active === "replay" && <ReplayView />}
        {active === "tradeoff" && <TradeoffExplorer comparison={comparison.data} hospitals={sites} />}
        {active === "per-hospital" && <PerHospitalView data={comparison.data} sites={sites} />}
        {active === "multiseed" && <MultiSeedView data={equity.data} />}
        {active === "worst-served" && <WorstServedView equity={equity.data} sites={sites} />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Headline: federated vs each hospital alone vs all data pooled       */
/* ------------------------------------------------------------------ */

function HeadlineView({ equity, sites }) {
  const h = equity.headline;
  const nTest = sites.reduce((s, x) => s + x.n_test, 0);
  return (
    <>
      <div className="headline-callout">
        <span className="headline-callout-label">Headline finding</span>
        <span>{h.text}</span>
      </div>

      <div className="grid grid-stats" style={{ marginTop: 20 }}>
        <StatCard
          label="Beats training alone"
          value={`${h.seeds_fedavg_beats_local}/${h.n_seeds} seeds`}
          sublabel={`federated ${fmtPct(h.fedavg_mean_accuracy)} vs alone ${fmtPct(h.local_mean_accuracy)}`}
          accent="success"
        />
        <StatCard label="Average gain" value={`+${h.delta_pp} pp`} sublabel="overall accuracy, 5-seed mean" accent="primary" />
        <StatCard
          label="Matches or beats pooling"
          value={`${h.seeds_fedavg_matches_or_beats_centralized}/${h.n_seeds} seeds`}
          sublabel={`pooled data averaged ${fmtPct(h.centralized_mean_accuracy)}`}
          accent="teal"
        />
        <StatCard label="Patient records shared" value="0" sublabel="only model weights travel" accent="success" />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Overall accuracy per seed</h3>
        <p style={mutedNote}>
          Correct predictions across all {nTest} held-out test patients &middot; same model, five random starting
          points
        </p>
        <SeedComparisonChart seeds={h.per_seed} />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Reading this honestly</h3>
        <ul style={{ fontSize: "0.9rem", lineHeight: 1.55, margin: 0, paddingLeft: 18 }}>
          <li>
            &ldquo;Training alone&rdquo; means each hospital trains the same model on only its own patients; the
            comparison is overall accuracy across all four hospitals&apos; test patients.
          </li>
          <li>
            Not every hospital gains: Switzerland&apos;s own model scores higher on accuracy than the federated one,
            but on a test set of {sites.find((x) => x.hospital === "switzerland")?.test_disease} diseased and{" "}
            {sites.find((x) => x.hospital === "switzerland")?.test_no_disease} healthy patients, where accuracy says
            little (see the <em>Per hospital</em> tab with balanced accuracy or AUC).
          </li>
          <li>
            Runs on Flower&apos;s Ray engine can differ by one or two test patients between identical runs
            (&plusmn;0.5&ndash;1 pp), so single-seed differences that small are noise.
          </li>
        </ul>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Personalization at the worst-served hospital (a mixed result)       */
/* ------------------------------------------------------------------ */

function WorstServedView({ equity, sites }) {
  const ws = equity.worst_served;
  const rows = ws.per_seed.map((r) => ({
    ...r,
    label: `Seed ${r.seed}`,
    hospitalName: hospitalLabel(r.worst_hospital),
  }));
  const nSeeds = rows.length;
  const worstCounts = rows.reduce((acc, r) => ({ ...acc, [r.worst_hospital]: (acc[r.worst_hospital] || 0) + 1 }), {});
  const [mostWorst, mostWorstCount] = Object.entries(worstCounts).sort((a, b) => b[1] - a[1])[0];
  const worstSite = sites.find((x) => x.hospital === mostWorst);
  const onePatientPp = worstSite ? 100 / worstSite.n_test : null;

  return (
    <>
      <div className="grid grid-stats">
        <StatCard
          label="Worst-served hospital"
          value={hospitalLabel(mostWorst)}
          sublabel={`lowest FedAvg accuracy in ${mostWorstCount}/${nSeeds} seeds`}
          accent="warning"
        />
        <StatCard label="Personalization helped" value={`${ws.seeds_improved}/${nSeeds} seeds`} sublabel="same hospital" accent="primary" />
        <StatCard label="No change / hurt" value={`${ws.seeds_flat} / ${ws.seeds_worse}`} sublabel="seeds" accent="teal" />
        <StatCard
          label="Mean change"
          value={`${ws.mean_delta_pp > 0 ? "+" : ""}${ws.mean_delta_pp} pp`}
          sublabel={onePatientPp ? `1 patient = ${onePatientPp.toFixed(1)} pp at ${hospitalLabel(mostWorst)}` : "all seeds"}
          accent={ws.mean_delta_pp < 0 ? "warning" : "success"}
        />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Worst-served hospital: FedAvg vs personalized</h3>
        <p style={mutedNote}>
          For each model-init seed, the hospital with the lowest FedAvg accuracy &mdash; and the same hospital after
          FedProx + local fine-tuning
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={rows} margin={{ top: 24, right: 10, left: -10, bottom: 0 }} barGap={2}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="label" interval={0} tick={<SeedTick rows={rows} />} height={44} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 12 }} />
            <Tooltip content={<EquityTooltip />} cursor={hoverCursor} />
            <Legend
              wrapperStyle={{ fontSize: 12 }}
              itemSorter={null}
              formatter={(value) => <span style={{ color: "var(--color-text)" }}>{value}</span>}
            />
            <Bar dataKey="fedavg_accuracy" name="FedAvg" fill={EXPERIMENT_COLORS.fedavg} radius={[4, 4, 0, 0]}>
              <LabelList dataKey="fedavg_accuracy" position="top" formatter={(v) => fmtPct(v)} style={{ fontSize: 11, fill: "var(--color-text)" }} />
            </Bar>
            <Bar dataKey="personalized_accuracy" name="Personalized" fill={EXPERIMENT_COLORS.personalized} radius={[4, 4, 0, 0]}>
              <LabelList dataKey="personalized_accuracy" position="top" formatter={(v) => fmtPct(v)} style={{ fontSize: 11, fill: "var(--color-text)" }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Per-seed results</h3>
          <table>
            <thead>
              <tr>
                <th>Seed</th>
                <th>Worst-served</th>
                <th>FedAvg</th>
                <th>Personalized</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.seed}>
                  <td>{r.seed}</td>
                  <td>{r.hospitalName}</td>
                  <td>{fmtPct(r.fedavg_accuracy)}</td>
                  <td>
                    <strong>{fmtPct(r.personalized_accuracy)}</strong>
                  </td>
                  <td>
                    <span className={`badge ${r.delta > 0 ? "badge-low" : r.delta < 0 ? "badge-high" : "badge-neutral"}`}>
                      {r.delta > 0 ? `▲ ${fmtPts(r.delta)}` : r.delta < 0 ? `▼ ${fmtPts(r.delta)}` : "No change"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Reading this honestly</h3>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.55, margin: "0 0 10px 0" }}>
            The project set out to test whether personalization helps the hospital a shared model serves worst. On the
            real four-hospital data it does not do so reliably: {ws.seeds_improved} seeds better, {ws.seeds_flat} flat,{" "}
            {ws.seeds_worse} worse.
          </p>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.55, margin: 0 }}>
            {onePatientPp
              ? `Each step is one or two of ${hospitalLabel(mostWorst)}'s ${worstSite.n_test} test patients (${onePatientPp.toFixed(1)} pp each), the same size as run-to-run noise.`
              : "The changes are one or two test patients, the same size as run-to-run noise."}{" "}
            An earlier version of this project, on a synthetic split of Cleveland alone, found a large improvement
            here; it did not carry over to the real multi-hospital data.
          </p>
        </div>
      </div>
    </>
  );
}

function SeedTick({ x, y, payload, rows }) {
  const row = rows.find((r) => r.label === payload.value);
  return (
    <g transform={`translate(${x},${y})`}>
      <text dy={14} textAnchor="middle" fontSize={12} fill="var(--color-text)">
        {payload.value}
      </text>
      <text dy={30} textAnchor="middle" fontSize={11} fill="var(--color-text-muted)">
        {row?.hospitalName}
      </text>
    </g>
  );
}

/* ------------------------------------------------------------------ */
/* Training replay: the real FedAvg / FedProx runs, round by round     */
/* ------------------------------------------------------------------ */

async function getReplayData() {
  const [curves, hospitals, summary] = await Promise.all([getTrainingCurves(), getHospitals(), getDashboardSummary()]);
  return { curves, hospitals: hospitals.hospitals, summary };
}

function ReplayView() {
  const { data, loading, error } = useApiData(getReplayData, []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} />;
  return <TrainingReplay curves={data.curves} hospitals={data.hospitals} summary={data.summary} />;
}

/* ------------------------------------------------------------------ */
/* Multi-seed global accuracy: mean ± std across 5 model-init seeds    */
/* ------------------------------------------------------------------ */

function MultiSeedView({ data }) {
  const rows = data.phase5_multiseed_summary
    .map((r) => ({ ...r, key: MULTISEED_NAME_TO_KEY[r.experiment] }))
    .filter((r) => r.key)
    .sort((a, b) => EXPERIMENT_ORDER.indexOf(a.key) - EXPERIMENT_ORDER.indexOf(b.key))
    .map((r) => ({ ...r, label: EXPERIMENT_LABELS[r.key] }));

  const nSeeds = rows[0]?.n_seeds;
  const lows = rows.map((r) => Math.min(r.mean_accuracy - r.std_accuracy, r.min_accuracy));
  const highs = rows.map((r) => Math.max(r.mean_accuracy + r.std_accuracy, r.max_accuracy));
  const yMin = Math.floor(Math.min(...lows) * 50 - 1) / 50;
  const yMax = Math.ceil(Math.max(...highs) * 50 + 1) / 50;
  const yTicks = [];
  for (let t = yMin; t <= yMax + 1e-9; t += 0.02) yTicks.push(Number(t.toFixed(2)));

  // Compare at the precision the results are reported (3 dp).
  const bestMean = Math.max(...rows.map((r) => r.mean_accuracy));
  const leaders = rows.filter((r) => r.mean_accuracy.toFixed(3) === bestMean.toFixed(3));
  const mostStable = rows.reduce((a, b) => (b.std_accuracy < a.std_accuracy ? b : a));

  return (
    <>
      <div className="grid grid-stats">
        <StatCard
          label="Highest mean accuracy"
          value={fmtPct(bestMean)}
          sublabel={leaders.map((r) => r.label).join(" & ") + (leaders.length > 1 ? " (tied)" : "")}
          accent="primary"
        />
        <StatCard
          label="Most stable"
          value={mostStable.label}
          sublabel={`std ${fmtPct(mostStable.std_accuracy)} across seeds`}
          accent="teal"
        />
        <StatCard
          label="Seeds per setting"
          value={nSeeds}
          sublabel="partition + local split held fixed"
          accent="primary"
        />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Global accuracy, mean ± std</h3>
        <p style={mutedNote}>
          Dot is the mean over {nSeeds} model-init seeds, whiskers are ±1 std. Axis zoomed to{" "}
          {fmtPct(yMin, 0)}–{fmtPct(yMax, 0)} so the spread is visible &mdash; the differences are only a few points.
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={rows} margin={{ top: 20, right: 10, left: -4, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="label" interval={0} tick={{ fontSize: 12 }} padding={{ left: 60, right: 60 }} />
            <YAxis
              domain={[yMin, yMax]}
              ticks={yTicks}
              tickFormatter={(v) => fmtPct(v, 0)}
              tick={{ fontSize: 12 }}
              allowDataOverflow
            />
            <Tooltip content={<MultiSeedTooltip />} cursor={hoverCursor} />
            <Scatter dataKey="mean_accuracy" name="Mean accuracy" shape={<MeanDot />}>
              {rows.map((r) => (
                <Cell key={r.key} fill={EXPERIMENT_COLORS[r.key]} />
              ))}
              <ErrorBar dataKey="std_accuracy" width={10} strokeWidth={2} stroke="var(--color-text-muted)" direction="y" />
              <LabelList
                dataKey="mean_accuracy"
                position="right"
                offset={12}
                formatter={(v) => fmtPct(v)}
                style={{ fontSize: 12, fill: "var(--color-text)" }}
              />
            </Scatter>
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Summary across {nSeeds} seeds</h3>
          <table>
            <thead>
              <tr>
                <th>Setting</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key}>
                  <td>
                    <Swatch color={EXPERIMENT_COLORS[r.key]} />
                    {r.label}
                  </td>
                  <td>
                    <strong>{fmtPct(r.mean_accuracy)}</strong>
                  </td>
                  <td>±{fmtPct(r.std_accuracy)}</td>
                  <td>{fmtPct(r.min_accuracy)}</td>
                  <td>{fmtPct(r.max_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Reading this honestly</h3>
          <ul style={{ fontSize: "0.9rem", lineHeight: 1.55, margin: 0, paddingLeft: 18 }}>
            <li>Federated training (FedAvg) beats each hospital training alone and matches or beats pooling all data.</li>
            <li>FedProx&apos;s proximal term alone (before fine-tuning) lands within noise of FedAvg.</li>
            <li>
              Personalized has the highest mean, but much of that gain is Switzerland&apos;s model predicting
              &ldquo;disease&rdquo; for everyone &mdash; see the <em>Per hospital</em> tab with AUC or balanced
              accuracy.
            </li>
          </ul>
        </div>
      </div>
    </>
  );
}

function MeanDot({ cx, cy, fill }) {
  if (cx == null || cy == null) return null;
  return <circle cx={cx} cy={cy} r={7} fill={fill} stroke="#ffffff" strokeWidth={2} />;
}

/* ------------------------------------------------------------------ */
/* Single run (model-init seed 42): per-hospital accuracy by setting   */
/* ------------------------------------------------------------------ */

const PER_HOSPITAL_METRICS = [
  { id: "accuracy", label: "Accuracy" },
  { id: "balanced_accuracy", label: "Balanced accuracy" },
  { id: "auc", label: "AUC" },
];

function PerHospitalView({ data, sites }) {
  const [metric, setMetric] = useState("accuracy");
  const keys = EXPERIMENT_ORDER.filter((k) => data.per_hospital[k]);
  const metricLabel = PER_HOSPITAL_METRICS.find((m) => m.id === metric).label;

  // Each experiment's per-hospital list comes in its own order; index by hospital.
  const byHospital = {};
  for (const key of keys) {
    for (const row of data.per_hospital[key]) {
      byHospital[row.hospital] = { ...byHospital[row.hospital], [key]: row[metric] };
    }
  }
  const order = sites.map((x) => x.hospital);
  const chartData = order.map((h) => ({ hospital: h, label: hospitalLabel(h), ...byHospital[h] }));
  const site = Object.fromEntries(sites.map((x) => [x.hospital, x]));

  return (
    <>
      <div className="card">
        <div className="tradeoff-head">
          <div>
            <h3>{metricLabel} per hospital, by training setting</h3>
            <p style={{ ...mutedNote, marginBottom: 0 }}>
              Model-init seed 42 &middot; each hospital&apos;s own held-out test set
              {metric !== "accuracy" && " · dashed line = chance (0.5)"}
            </p>
          </div>
          <div className="segmented" role="radiogroup" aria-label="Metric">
            {PER_HOSPITAL_METRICS.map((m) => (
              <button
                key={m.id}
                type="button"
                role="radio"
                aria-checked={metric === m.id}
                className={"segmented-option" + (metric === m.id ? " active" : "")}
                onClick={() => setMetric(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }} barGap={2}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="label" interval={0} tick={{ fontSize: 12 }} />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v) => (metric === "auc" ? v.toFixed(1) : `${Math.round(v * 100)}%`)}
              tick={{ fontSize: 12 }}
            />
            {metric !== "accuracy" && <ReferenceLine y={0.5} stroke="var(--color-text-muted)" strokeDasharray="4 4" />}
            <Tooltip content={<PerHospitalTooltip keys={keys} metric={metric} />} cursor={hoverCursor} />
            <Legend
              wrapperStyle={{ fontSize: 12 }}
              itemSorter={null}
              formatter={(value) => <span style={{ color: "var(--color-text)" }}>{value}</span>}
            />
            {keys.map((key) => (
              <Bar key={key} dataKey={key} name={EXPERIMENT_LABELS[key]} fill={EXPERIMENT_COLORS[key]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>{metricLabel} table</h3>
        <p style={mutedNote}>
          Best setting per hospital in bold
          {metric === "accuracy" && " · last column: accuracy of always predicting the majority class"}
        </p>
        <table>
          <thead>
            <tr>
              <th>Hospital</th>
              <th>Test (disease : healthy)</th>
              {keys.map((key) => (
                <th key={key}>
                  <Swatch color={EXPERIMENT_COLORS[key]} />
                  {EXPERIMENT_LABELS[key]}
                </th>
              ))}
              {metric === "accuracy" && <th>Majority-class</th>}
            </tr>
          </thead>
          <tbody>
            {chartData.map((row) => {
              const best = Math.max(...keys.map((k) => row[k] ?? 0));
              const fmt = (v) => (v == null ? "—" : metric === "auc" ? v.toFixed(3) : fmtPct(v));
              return (
                <tr key={row.hospital}>
                  <td>{row.label}</td>
                  <td className="muted-inline">
                    {site[row.hospital].test_disease} : {site[row.hospital].test_no_disease}
                  </td>
                  {keys.map((key) => (
                    <td key={key}>{row[key] === best ? <strong>{fmt(row[key])}</strong> : fmt(row[key])}</td>
                  ))}
                  {metric === "accuracy" && <td className="muted-inline">{fmtPct(site[row.hospital].majority_class_accuracy)}</td>}
                </tr>
              );
            })}
            {metric === "accuracy" && (
              <tr className="table-total">
                <td>Overall (all test patients)</td>
                <td />
                {keys.map((key) => (
                  <td key={key}>{fmtPct(data.global_accuracy[key])}</td>
                ))}
                <td />
              </tr>
            )}
          </tbody>
        </table>
        <p style={{ ...mutedNote, marginTop: 12, marginBottom: 0 }}>{data.note}</p>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Shared bits                                                         */
/* ------------------------------------------------------------------ */

const tooltipBox = {
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: "var(--radius-sm)",
  boxShadow: "var(--shadow-md)",
  padding: "8px 12px",
  fontSize: "0.82rem",
  lineHeight: 1.5,
};

function Swatch({ color }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 10,
        height: 10,
        borderRadius: 2,
        background: color,
        marginRight: 6,
        verticalAlign: "baseline",
      }}
    />
  );
}

function EquityTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={tooltipBox}>
      <strong>
        {d.label} &middot; {d.hospitalName}
      </strong>
      <div>
        <Swatch color={EXPERIMENT_COLORS.fedavg} />
        FedAvg: {fmtPct(d.fedavg_accuracy)}
      </div>
      <div>
        <Swatch color={EXPERIMENT_COLORS.personalized} />
        Personalized: {fmtPct(d.personalized_accuracy)}
      </div>
      <div style={{ color: "var(--color-text-muted)" }}>Change: {fmtPts(d.delta)}</div>
    </div>
  );
}

function MultiSeedTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={tooltipBox}>
      <strong>
        <Swatch color={EXPERIMENT_COLORS[d.key]} />
        {d.label}
      </strong>
      <div>
        Mean {fmtPct(d.mean_accuracy)} ± {fmtPct(d.std_accuracy)}
      </div>
      <div style={{ color: "var(--color-text-muted)" }}>
        Range {fmtPct(d.min_accuracy)}–{fmtPct(d.max_accuracy)} over {d.n_seeds} seeds
      </div>
    </div>
  );
}

function PerHospitalTooltip({ active, payload, keys, metric = "accuracy" }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={tooltipBox}>
      <strong>{d.label}</strong>
      {keys.map((key) => (
        <div key={key}>
          <Swatch color={EXPERIMENT_COLORS[key]} />
          {EXPERIMENT_LABELS[key]}: {d[key] == null ? "—" : metric === "auc" ? d[key].toFixed(3) : fmtPct(d[key])}
        </div>
      ))}
    </div>
  );
}
