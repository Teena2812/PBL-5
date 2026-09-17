export default function ComingSoon({ title }) {
  return (
    <div>
      <div className="page-header">
        <h1>{title}</h1>
      </div>
      <div className="card">
        <p style={{ margin: 0, color: "var(--color-text-muted)" }}>
          This screen hasn't been built yet &mdash; coming in a later step.
        </p>
      </div>
    </div>
  );
}
