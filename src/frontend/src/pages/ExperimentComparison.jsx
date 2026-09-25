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
} from "recharts";
import { getEquityAnalysis, getExperimentComparison } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import StatCard from "../components/StatCard";
import Tabs from "../components/Tabs";
import {
  EXPERIMENT_ORDER,
  EXPERIMENT_LABELS,
  EXPERIMENT_COLORS,
  MULTISEED_NAME_TO_KEY,
  fmtPct,
  hospitalLabel,
} from "../constants/experiments";

const TABS = [
  { id: "equity", label: "Worst-served hospital", badge: "Headline" },
  { id: "multiseed", label: "5-seed mean ± std" },
  { id: "per-hospital", label: "Per hospital (single run)" },
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
  const [searchParams, setSearchParams] = useSearchParams();

  const requested = searchParams.get("view");
  const active = TABS.some((t) => t.id === requested) ? requested : "equity";
  const setActive = (id) => setSearchParams(id === "equity" ? {} : { view: id }, { replace: true });

  if (equity.loading || comparison.loading) return <Loading />;
  if (equity.error) return <ErrorState error={equity.error} />;
  if (comparison.error) return <ErrorState error={comparison.error} />;

  return (
    <div>
      <div className="page-header">
        <h1>Experiment Comparison</h1>
        <p>
          Local ML vs Centralized vs FedAvg vs FedProx vs Personalized FL &middot; identical model architecture in
          every setting
        </p>
      </div>

      <Tabs tabs={TABS} active={active} onChange={setActive} label="Comparison views" />

      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {active === "equity" && <EquityView data={equity.data} />}
        {active === "multiseed" && <MultiSeedView data={equity.data} />}
        {active === "per-hospital" && <PerHospitalView data={comparison.data} />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Equity: worst-served hospital per seed, before/after personalization */
/* ------------------------------------------------------------------ */

function EquityView({ data }) {
  const rows = data.per_seed.map((r) => ({
    ...r,
    label: `Seed ${r.seed}`,
    hospitalName: hospitalLabel(r.worst_hospital),
  }));
  const nSeeds = rows.length;
  const improved = rows.filter((r) => r.delta > 0);
  const worse = rows.filter((r) => r.delta < 0);
  const meanDelta = (list) => list.reduce((sum, r) => sum + r.delta, 0) / list.length;

  const worstCounts = rows.reduce((acc, r) => ({ ...acc, [r.worst_hospital]: (acc[r.worst_hospital] || 0) + 1 }), {});
  const [mostWorst, mostWorstCount] = Object.entries(worstCounts).sort((a, b) => b[1] - a[1])[0];

  return (
    <>
      <div className="headline-callout">
        <span className="headline-callout-label">Headline finding</span>
        <span>{data.headline}</span>
      </div>

      <div className="grid grid-stats" style={{ marginTop: 20 }}>
        <StatCard
          label="Worst hospital improved"
          value={`${improved.length}/${nSeeds} seeds`}
          sublabel="personalized vs FedAvg, same hospital"
          accent="success"
        />
        <StatCard
          label="Avg gain when improved"
          value={improved.length ? fmtPts(meanDelta(improved)) : "—"}
          sublabel={`over the ${improved.length} improved seeds`}
          accent="primary"
        />
        <StatCard
          label="Avg gain, all seeds"
          value={fmtPts(meanDelta(rows))}
          sublabel="including the flat case"
          accent="teal"
        />
        <StatCard
          label="Seeds made worse"
          value={worse.length}
          sublabel={`${hospitalLabel(mostWorst)} was worst-served in ${mostWorstCount}/${nSeeds} seeds`}
          accent={worse.length ? "warning" : "success"}
        />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Worst-served hospital: FedAvg vs Personalized</h3>
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
              <LabelList
                dataKey="fedavg_accuracy"
                position="top"
                formatter={(v) => fmtPct(v)}
                style={{ fontSize: 11, fill: "var(--color-text)" }}
              />
            </Bar>
            <Bar
              dataKey="personalized_accuracy"
              name="Personalized"
              fill={EXPERIMENT_COLORS.personalized}
              radius={[4, 4, 0, 0]}
            >
              <LabelList
                dataKey="personalized_accuracy"
                position="top"
                formatter={(v) => fmtPct(v)}
                style={{ fontSize: 11, fill: "var(--color-text)" }}
              />
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
          <h3>Why this is the headline</h3>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.55, margin: "0 0 10px 0" }}>
            A global accuracy number is weighted by patient count, so the largest hospitals dominate it and a gain at
            a small, poorly-served hospital barely moves it. On that global number, Personalized only ties FedAvg (see
            the <em>5-seed mean ± std</em> tab).
          </p>
          <p style={{ fontSize: "0.9rem", lineHeight: 1.55, margin: 0 }}>
            The project&apos;s hypothesis is about the worst-off participant: does personalization help the hospital
            the shared model serves worst? That is what this view measures &mdash; and the one seed with no gain is
            shown as a flat result, not hidden.
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
            <li>Federated training (FedAvg) matches or beats both training in isolation and pooling all data.</li>
            <li>
              FedProx&apos;s proximal term alone (before fine-tuning) scores <em>below</em> FedAvg; local fine-tuning
              recovers that loss but only ties FedAvg on this global number.
            </li>
            <li>
              Global accuracy is not where personalization&apos;s value shows &mdash; the{" "}
              <em>Worst-served hospital</em> tab is.
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

function PerHospitalView({ data }) {
  const keys = EXPERIMENT_ORDER.filter((k) => data.per_hospital[k]);

  // Each experiment's per-hospital list comes in its own order; index by hospital.
  const byHospital = {};
  for (const key of keys) {
    for (const row of data.per_hospital[key]) {
      byHospital[row.hospital] = { ...byHospital[row.hospital], [key]: row.accuracy };
    }
  }
  const hospitals = Object.keys(byHospital).sort();
  const chartData = hospitals.map((h) => ({ hospital: h, label: hospitalLabel(h), ...byHospital[h] }));

  return (
    <>
      <div className="card">
        <h3>Accuracy per hospital, by training setting</h3>
        <p style={mutedNote}>Model-init seed 42 &middot; each hospital&apos;s own local test set</p>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }} barGap={2}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="label" interval={0} tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 12 }} />
            <Tooltip content={<PerHospitalTooltip keys={keys} />} cursor={hoverCursor} />
            <Legend
              wrapperStyle={{ fontSize: 12 }}
              itemSorter={null}
              formatter={(value) => <span style={{ color: "var(--color-text)" }}>{value}</span>}
            />
            {keys.map((key) => (
              <Bar
                key={key}
                dataKey={key}
                name={EXPERIMENT_LABELS[key]}
                fill={EXPERIMENT_COLORS[key]}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Accuracy table</h3>
        <p style={mutedNote}>Best setting per hospital in bold</p>
        <table>
          <thead>
            <tr>
              <th>Hospital</th>
              {keys.map((key) => (
                <th key={key}>
                  <Swatch color={EXPERIMENT_COLORS[key]} />
                  {EXPERIMENT_LABELS[key]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {chartData.map((row) => {
              const best = Math.max(...keys.map((k) => row[k] ?? 0));
              return (
                <tr key={row.hospital}>
                  <td>{row.label}</td>
                  {keys.map((key) => (
                    <td key={key}>{row[key] === best ? <strong>{fmtPct(row[key])}</strong> : fmtPct(row[key])}</td>
                  ))}
                </tr>
              );
            })}
            <tr className="table-total">
              <td>Global (weighted)</td>
              {keys.map((key) => (
                <td key={key}>{fmtPct(data.global_accuracy[key])}</td>
              ))}
            </tr>
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

function PerHospitalTooltip({ active, payload, keys }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={tooltipBox}>
      <strong>{d.label}</strong>
      {keys.map((key) => (
        <div key={key}>
          <Swatch color={EXPERIMENT_COLORS[key]} />
          {EXPERIMENT_LABELS[key]}: {fmtPct(d[key])}
        </div>
      ))}
    </div>
  );
}
