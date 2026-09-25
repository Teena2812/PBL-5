import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import Hospitals from "./pages/Hospitals";
import ExperimentComparison from "./pages/ExperimentComparison";
import Explainability from "./pages/Explainability";
import ComingSoon from "./pages/ComingSoon";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/problem" element={<ComingSoon title="Problem" />} />
          <Route path="/hospitals" element={<Hospitals />} />
          <Route path="/experiments" element={<ExperimentComparison />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/roadmap" element={<ComingSoon title="Roadmap" />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
