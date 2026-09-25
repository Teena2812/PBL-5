import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import Hospitals from "./pages/Hospitals";
import ExperimentComparison from "./pages/ExperimentComparison";
import Explainability from "./pages/Explainability";
import Problem from "./pages/Problem";
import Roadmap from "./pages/Roadmap";
import Present from "./pages/Present";

function DashboardShell() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/problem" element={<Problem />} />
          <Route path="/hospitals" element={<Hospitals />} />
          <Route path="/experiments" element={<ExperimentComparison />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/roadmap" element={<Roadmap />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      {/* Story mode takes the whole screen, without the sidebar. */}
      <Route path="/present" element={<Present />} />
      <Route path="*" element={<DashboardShell />} />
    </Routes>
  );
}
