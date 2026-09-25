import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import Hospitals from "./pages/Hospitals";
import ExperimentComparison from "./pages/ExperimentComparison";
import ComingSoon from "./pages/ComingSoon";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/hospitals" element={<Hospitals />} />
          <Route path="/experiments" element={<ExperimentComparison />} />
          <Route path="/training" element={<ComingSoon title="Training Curves" />} />
          <Route path="/explainability" element={<ComingSoon title="Explainability" />} />
          <Route path="/predict" element={<ComingSoon title="Predict Risk" />} />
        </Routes>
      </main>
    </div>
  );
}
