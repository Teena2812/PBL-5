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
      {isNetworkError && (
        <div style={{ marginTop: 6 }}>
          Is the backend running? Try:{" "}
          <code>venv\Scripts\uvicorn src.backend.main:app --reload</code>
        </div>
      )}
    </div>
  );
}
