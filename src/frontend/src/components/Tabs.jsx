import "./Tabs.css";

/**
 * Controlled tab bar. tabs = [{ id, label, badge? }]; the caller owns the
 * active id (e.g. in the URL) and renders the matching panel itself.
 */
export default function Tabs({ tabs, active, onChange, label }) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          id={`tab-${tab.id}`}
          aria-selected={tab.id === active}
          aria-controls={`panel-${tab.id}`}
          className={"tabs-tab" + (tab.id === active ? " active" : "")}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
          {tab.badge && <span className="tabs-badge">{tab.badge}</span>}
        </button>
      ))}
    </div>
  );
}
