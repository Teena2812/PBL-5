import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  LabelList,
} from "recharts";
import { getHospitals, getDashboardSummary } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import StatCard from "../components/StatCard";
import { fmtPct, hospitalLabel } from "../constants/experiments";

// Disease vs. no-disease pair, validated for colorblind separation
// (deutan dE 13.1) against the light card surface.
const DISEASE_COLOR = "#dc2626";
const HEALTHY_COLOR = "#0d9488";

const CP_TYPE_LABELS = {
  1: "Typical angina",
  2: "Atypical angina",
  3: "Non-anginal pain",
  4: "Asymptomatic",
};

const mutedNote = { color: "var(--color-text-muted)", fontSize: "0.85rem", marginTop: -4 };

export default function Hospitals() {
  const hospitals = useApiData(getHospitals, []);
  const summary = useApiData(getDashboardSummary, []);

  if (hospitals.loading || summary.loading) return <Loading />;
  if (hospitals.error) return <ErrorState error={hospitals.error} />;
  if (summary.error) return <ErrorState error={summary.error} />;

  const list = hospitals.data.hospitals;
  const totalPatients = list.reduce((sum, h) => sum + h.n_patients, 0);

  // disease_rate is exported rounded to 3 dp, so round back to whole patients.
  const chartData = list.map((h) => {
    const disease = Math.round(h.n_patients * h.disease_rate);
    return {
      key: h.hospital,
      label: hospitalLabel(h.hospital),
      disease,
      healthy: h.n_patients - disease,
      n_patients: h.n_patients,
      disease_rate: h.disease_rate,
      share: h.n_patients / totalPatients,
    };
  });

  const totalDisease = chartData.reduce((sum, h) => sum + h.disease, 0);
  const pooledRate = totalDisease / totalPatients;

  const sizes = list.map((h) => h.n_patients);
  const rates = list.map((h) => h.disease_rate);
  const minSize = Math.min(...sizes);
  const maxSize = Math.max(...sizes);
  const minRate = Math.min(...rates);
  const maxRate = Math.max(...rates);
  const minClassCount = Math.min(...chartData.flatMap((h) => [h.disease, h.healthy]));

  return (
    <div>
      <div className="page-header">
        <h1>Hospitals</h1>
        <p>
          {hospitals.data.n_hospitals} simulated hospitals &middot; {totalPatients} patients from{" "}
          {summary.data.dataset.name} &middot; non-IID Dirichlet label-skew partition
        </p>
      </div>

      <div className="grid grid-stats">
        <StatCard
          label="Hospital size range"
          value={`${minSize}–${maxSize}`}
          sublabel={`patients · largest holds ${fmtPct(maxSize / totalPatients)} of all data`}
          accent="primary"
        />
        <StatCard
          label="Disease-rate range"
          value={`${fmtPct(minRate)}–${fmtPct(maxRate)}`}
          sublabel={`${(maxRate / minRate).toFixed(1)}× spread between hospitals`}
          accent="warning"
        />
        <StatCard
          label="Pooled disease rate"
          value={fmtPct(pooledRate)}
          sublabel={`${totalDisease} of ${totalPatients} patients · what IID would give every hospital`}
          accent="teal"
        />
        <StatCard
          label="Smallest class count"
          value={minClassCount}
          sublabel="every hospital has both classes"
          accent="success"
        />
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Patients per hospital</h3>
          <p style={mutedNote}>Bar height is hospital size; the split shows each hospital's label mix</p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="label" interval={0} tick={{ fontSize: 12 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
              <Tooltip content={<SizeTooltip />} cursor={{ fill: "rgba(148, 163, 184, 0.12)" }} />
              <Legend
                wrapperStyle={{ fontSize: 12 }}
              itemSorter={null}
                formatter={(value) => <span style={{ color: "var(--color-text)" }}>{value}</span>}
              />
              <Bar
                dataKey="disease"
                name="Heart disease"
                stackId="patients"
                fill={DISEASE_COLOR}
                stroke="#ffffff"
                strokeWidth={1}
              />
              <Bar
                dataKey="healthy"
                name="No disease"
                stackId="patients"
                fill={HEALTHY_COLOR}
                stroke="#ffffff"
                strokeWidth={1}
                radius={[4, 4, 0, 0]}
              >
                <LabelList dataKey="n_patients" position="top" style={{ fontSize: 12, fill: "var(--color-text)" }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Disease rate per hospital</h3>
          <p style={mutedNote}>Dashed line is the pooled rate &mdash; under an IID split every bar would sit on it</p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} margin={{ top: 20, right: 80, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="label" interval={0} tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 12 }} />
              <Tooltip content={<RateTooltip pooledRate={pooledRate} />} cursor={{ fill: "rgba(148, 163, 184, 0.12)" }} />
              <ReferenceLine
                y={pooledRate}
                stroke="var(--color-text-muted)"
                strokeDasharray="6 4"
                label={{
                  value: `Pooled ${fmtPct(pooledRate)}`,
                  position: "right",
                  fontSize: 11,
                  fill: "var(--color-text-muted)",
                }}
              />
              <Bar dataKey="disease_rate" name="Disease rate" fill={DISEASE_COLOR} radius={[4, 4, 0, 0]}>
                <LabelList
                  dataKey="disease_rate"
                  position="top"
                  formatter={(v) => fmtPct(v)}
                  style={{ fontSize: 12, fill: "var(--color-text)" }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3>Hospital profiles</h3>
        <p style={mutedNote}>
          Label skew also shifts the case mix: disease-heavy hospitals are dominated by asymptomatic chest pain
        </p>
        <table>
          <thead>
            <tr>
              <th>Hospital</th>
              <th>Patients</th>
              <th>Share</th>
              <th>Disease rate</th>
              <th>Mean age</th>
              <th>Female</th>
              <th>Dominant chest pain</th>
              <th>Train / test</th>
            </tr>
          </thead>
          <tbody>
            {list.map((h) => (
              <tr key={h.hospital}>
                <td>{hospitalLabel(h.hospital)}</td>
                <td>{h.n_patients}</td>
                <td>{fmtPct(h.n_patients / totalPatients)}</td>
                <td>
                  <strong>{fmtPct(h.disease_rate)}</strong>
                </td>
                <td>{h.mean_age.toFixed(1)}</td>
                <td>{fmtPct(h.pct_female)}</td>
                <td>
                  {CP_TYPE_LABELS[h.dominant_cp_type] || "Unknown"}{" "}
                  <span style={{ color: "var(--color-text-muted)" }}>(type {h.dominant_cp_type})</span>
                </td>
                <td>
                  {h.n_train} / {h.n_test}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

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
      style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: color, marginRight: 6 }}
    />
  );
}

function SizeTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={tooltipBox}>
      <strong>{d.label}</strong>
      <div>
        {d.n_patients} patients ({fmtPct(d.share)} of total)
      </div>
      <div>
        <Swatch color={DISEASE_COLOR} />
        Heart disease: {d.disease}
      </div>
      <div>
        <Swatch color={HEALTHY_COLOR} />
        No disease: {d.healthy}
      </div>
    </div>
  );
}

function RateTooltip({ active, payload, pooledRate }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const diffPts = (d.disease_rate - pooledRate) * 100;
  return (
    <div style={tooltipBox}>
      <strong>{d.label}</strong>
      <div>
        Disease rate: {fmtPct(d.disease_rate)} ({d.disease} of {d.n_patients})
      </div>
      <div style={{ color: "var(--color-text-muted)" }}>
        {diffPts >= 0 ? "+" : "−"}
        {Math.abs(diffPts).toFixed(1)} pts vs. pooled
      </div>
    </div>
  );
}
