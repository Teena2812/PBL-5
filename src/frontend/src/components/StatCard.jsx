import "./StatCard.css";

export default function StatCard({ label, value, sublabel, accent = "primary" }) {
  return (
    <div className={`stat-card stat-card-${accent}`}>
      <div className="stat-card-label">{label}</div>
      <div className="stat-card-value">{value}</div>
      {sublabel && <div className="stat-card-sublabel">{sublabel}</div>}
    </div>
  );
}
