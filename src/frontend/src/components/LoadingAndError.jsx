// Static builds (GitHub Pages) have no backend to suggest starting.
const STATIC_BUILD = import.meta.env.VITE_STATIC_DATA === "true";

export function Loading() {
  return <div className="spinner" role="status" aria-label="Loading" />;
}

export function ErrorState({ error }) {
  const message = error?.response?.data?.detail || error?.message || "Something went wrong.";
  const isNetworkError = !error?.response;

  return (
    <div className="error-box">
      <strong>Couldn't load this data.</strong>
      <div style={{ marginTop: 6 }}>{message}</div>
      {isNetworkError && STATIC_BUILD && (
        <div style={{ marginTop: 6 }}>Check your connection and reload the page.</div>
      )}
      {isNetworkError && !STATIC_BUILD && (
        <div style={{ marginTop: 6 }}>
          Is the backend running? Try:{" "}
          <code>venv\Scripts\uvicorn src.backend.main:app --reload</code>
        </div>
      )}
    </div>
  );
}
