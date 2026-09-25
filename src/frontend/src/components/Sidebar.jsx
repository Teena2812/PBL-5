import { Link, NavLink } from "react-router-dom";
import "./Sidebar.css";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: "⌂", end: true },
  { to: "/problem", label: "Problem", icon: "\u{1F3AF}" },
  { to: "/hospitals", label: "Hospitals", icon: "\u{1F3E5}" },
  { to: "/experiments", label: "Experiment Comparison", icon: "\u{1F4CA}" },
  { to: "/explainability", label: "Explainability", icon: "\u{1F50D}" },
  { to: "/roadmap", label: "Roadmap", icon: "\u{1F5FA}" },
];

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-mark">FL</span>
        <div>
          <div className="sidebar-brand-title">Heart Disease Risk</div>
          <div className="sidebar-brand-subtitle">Personalized FL Dashboard</div>
        </div>
      </div>
      <Link to="/present" className="sidebar-present">
        ▶ Present
      </Link>
      <ul className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              end={item.end}
              className={({ isActive }) => "sidebar-link" + (isActive ? " active" : "")}
            >
              <span className="sidebar-link-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="sidebar-footer">Research prototype &middot; not for clinical use</div>
    </nav>
  );
}
