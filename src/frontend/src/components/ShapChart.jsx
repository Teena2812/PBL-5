import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";
import { featureLabel, featureValueText } from "../constants/features";
import "./PredictionResult.css";
import { useMediaQuery } from "../hooks/useMediaQuery";

// Same validated disease / no-disease pair as the Hospitals screen.
export const TOWARD_DISEASE = "#dc2626";
export const TOWARD_HEALTHY = "#0d9488";

function fmtPts(v) {
  const pts = v * 100;
  return `${pts >= 0 ? "+" : "−"}${Math.abs(pts).toFixed(1)}`;
}

/**
 * Diverging horizontal bars of per-feature SHAP values (probability units)
 * for one patient. contributions = [{ feature, feature_value, shap_value }],
 * already ranked by |shap_value|. rawPatient (optional) lets feature values
 * show in clinical units instead of the model's encoded values.
 */
export default function ShapChart({
  contributions,
  rawPatient,
  legend = ["Pushes toward no disease", "Pushes toward disease"],
}) {
  const narrow = useMediaQuery("(max-width: 768px)");
  const data = contributions.map((c) => ({
    ...c,
    label: `${featureLabel(c.feature)} = ${featureValueText(c.feature, c.feature_value, rawPatient)}`,
  }));
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.shap_value)), 0.01);
  // Multiple of 2 pts so the half-way ticks land on whole points.
  const bound = Math.ceil(maxAbs * 1.25 * 50) / 50;

  return (
    <>
      <div className="shap-legend">
        <span>
          <span className="shap-swatch" style={{ background: TOWARD_HEALTHY }} />
          {legend[0]}
        </span>
        <span>
          <span className="shap-swatch" style={{ background: TOWARD_DISEASE }} />
          {legend[1]}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(160, data.length * 38 + 40)}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
          <XAxis
            type="number"
            domain={[-bound, bound]}
            ticks={[-bound, -bound / 2, 0, bound / 2, bound]}
            tickFormatter={(v) => `${fmtPts(v)} pts`}
            tick={{ fontSize: 11 }}
          />
          <YAxis type="category" dataKey="label" width={narrow ? 150 : 320} tick={{ fontSize: narrow ? 10 : 12 }} interval={0} />
          <Tooltip content={<ShapTooltip />} cursor={{ fill: "rgba(148, 163, 184, 0.12)" }} />
          <ReferenceLine x={0} stroke="var(--color-text-muted)" />
          <Bar dataKey="shap_value" radius={4} barSize={20}>
            {data.map((d) => (
              <Cell key={d.feature} fill={d.shap_value >= 0 ? TOWARD_DISEASE : TOWARD_HEALTHY} />
            ))}
            <LabelList
              dataKey="shap_value"
              content={<ValueLabel />}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

// Value label just past the bar's end, on whichever side the bar points.
function ValueLabel({ x, y, width, height, value }) {
  if (value == null) return null;
  const end = x + width;
  const positive = value >= 0;
  return (
    <text
      x={positive ? end + 6 : end - 6}
      y={y + height / 2}
      dy={4}
      textAnchor={positive ? "start" : "end"}
      fontSize={12}
      fill="var(--color-text)"
    >
      {fmtPts(value)}
    </text>
  );
}

function ShapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{d.label}</strong>
      <div>
        {fmtPts(d.shap_value)} percentage points {d.shap_value >= 0 ? "toward disease" : "toward no disease"}
      </div>
    </div>
  );
}
