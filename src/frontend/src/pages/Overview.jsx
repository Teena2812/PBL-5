import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";
import { getDashboardSummary, getHospitals } from "../api/client";
import { useApiData } from "../hooks/useApiData";
import { Loading, ErrorState } from "../components/LoadingAndError";
import StatCard from "../components/StatCard";
import PrivacyBanner from "../components/PrivacyBanner";
import { EXPERIMENT_ORDER, EXPERIMENT_LABELS, EXPERIMENT_COLORS, fmtPct } from "../constants/experiments";

export default function Overview() {
  const summary = useApiData(getDashboardSummary, []);
  const hospitals = useApiData(getHospitals, []);

  if (summary.loading || hospitals.loading) return <Loading />;
  if (summary.error) return <ErrorState error={summary.error} />;
  if (hospitals.error) return <ErrorState error={hospitals.error} />;

  const s = summary.data;
  const chartData = EXPERIMENT_ORDER.filter((key) => key in s.global_accuracy).map((key) => ({
    key,
    label: EXPERIMENT_LABELS[key],
    accuracy: s.global_accuracy[key],
  }));

  const bestPersonalized = Math.max(...hospitals.data.hospitals.map((h) => h.personalized_accuracy));

  return (
    <div>
      <div className="page-header">
        <h1>Overview</h1>
        <p>
          {s.dataset.name} &middot; {s.n_hospitals} simulated hospitals &middot; non-IID Dirichlet partition
        </p>
      </div>

      <PrivacyBanner text={s.privacy_statement} />

      <div className="grid grid-stats" style={{ marginTop: 20 }}>
        <StatCard label="Simulated hospitals" value={s.n_hospitals} accent="primary" />
        <StatCard
          label="Total patients"
          value={s.dataset.n_patients}
          sublabel={`${s.dataset.n_features} features`}
          accent="teal"
        />
        <StatCard
          label="Training rounds"
          value={s.training.n_rounds}
          sublabel={`${s.training.local_epochs} local epochs/round`}
          accent="primary"
        />
        <StatCard
          label="Best personalized accuracy"
          value={fmtPct(bestPersonalized)}
          sublabel="single hospital, best case"
          accent="success"
        />
      </div>

      <div className="grid grid-two" style={{ marginTop: 20 }}>
        <div className="card">
          <h3>Global accuracy by experiment</h3>
          <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem", marginTop: -4 }}>
            All settings use the identical model architecture (apples-to-apples comparison)
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="label" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => fmtPct(v)} />
              <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                {chartData.map((entry) => (
                  <Cell key={entry.key} fill={EXPERIMENT_COLORS[entry.key]} />
                ))}
                <LabelList
                  dataKey="accuracy"
                  position="top"
                  formatter={(v) => fmtPct(v)}
                  style={{ fontSize: 12, fill: "var(--color-text)" }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Per-hospital personalized accuracy</h3>
          <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem", marginTop: -4 }}>
            FedProx + local fine-tuning, each hospital's own test set
          </p>
          <table>
            <thead>
              <tr>
                <th>Hospital</th>
                <th>Patients</th>
                <th>Disease rate</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {hospitals.data.hospitals.map((h) => (
                <tr key={h.hospital}>
                  <td>{h.hospital.replace("_", " ")}</td>
                  <td>{h.n_patients}</td>
                  <td>{fmtPct(h.disease_rate)}</td>
                  <td>
                    <strong>{fmtPct(h.personalized_accuracy)}</strong>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 0 }}>
        <h3>Model</h3>
        <p style={{ margin: 0, fontSize: "0.9rem", color: "var(--color-text-muted)" }}>
          {s.model_architecture.class} &middot; hidden layers {s.model_architecture.hidden_sizes.join(" → ")}{" "}
          &middot; FedProx proximal &mu; = {s.training.proximal_mu} &middot; {s.training.fine_tune_epochs} local
          fine-tuning epochs per hospital
        </p>
      </div>
    </div>
  );
}
