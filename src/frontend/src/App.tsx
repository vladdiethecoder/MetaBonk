import { Suspense, lazy, useEffect, useState } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import ContextBar from "./components/ContextBar";
import CommandPalette from "./components/CommandPalette";
import ContextDrawer from "./components/ContextDrawer";
import IssuesDrawer from "./components/IssuesDrawer";
import ErrorBoundary from "./components/ErrorBoundary";
import FrontendHealthOverlay from "./components/FrontendHealthOverlay";
import Overview from "./pages/Overview";
import NeuroSynaptic from "./pages/NeuroSynaptic";
import Runs from "./pages/Runs";
import Instances from "./pages/Instances";
import Skills from "./pages/Skills";
import Reasoning from "./pages/Reasoning";
import Spy from "./pages/Spy";
import Stream from "./pages/Stream";
import Supervisor from "./pages/Supervisor";

const BuildLab = lazy(() => import("./pages/BuildLab"));
const CCTV3D = lazy(() => import("./pages/CCTV3D"));

export default function App() {
  const loc = useLocation();
  const [issuesOpen, setIssuesOpen] = useState(false);
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const resetScrollLock = () => {
      document.body.style.overflow = "";
      document.body.style.paddingRight = "";
      document.body.style.touchAction = "";
      document.documentElement.style.overflow = "";
    };
    (window as any).__mbResetScrollLock = resetScrollLock;
    return () => {
      delete (window as any).__mbResetScrollLock;
    };
  }, []);
  // Stream overlay wants a clean fullscreen surface (OBS browser source).
  if (loc.pathname.startsWith("/stream") || loc.pathname.startsWith("/broadcast")) {
    return (
      <ErrorBoundary label="Stream">
        <Stream />
      </ErrorBoundary>
    );
  }
  if (loc.pathname.startsWith("/cctv")) {
    return (
      <Suspense fallback={<div className="card">Booting System...</div>}>
        <ErrorBoundary label="CCTV">
          <CCTV3D />
        </ErrorBoundary>
      </Suspense>
    );
  }
  if (loc.pathname === "/") {
    return (
      <ErrorBoundary label="NeuroSynaptic">
        <NeuroSynaptic />
      </ErrorBoundary>
    );
  }
  const nav = [
    { to: "/", label: "Neuro" },
    { to: "/overview", label: "Overview" },
    { to: "/supervisor", label: "Supervisor" },
    { to: "/runs", label: "Runs" },
    { to: "/instances", label: "Instances" },
    { to: "/reasoning", label: "Reasoning" },
    { to: "/build", label: "Build Lab" },
    { to: "/skills", label: "Skills" },
    { to: "/spy", label: "Spy" },
    { to: "/stream", label: "Stream" },
    { to: "/cctv", label: "CCTV" },
  ];
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">MetaBonk Dev UI</div>
        <nav className="nav">
          {nav.map((n) => (
            <Link key={n.to} to={`${n.to}${loc.search}`} className={loc.pathname === n.to ? "active" : ""}>
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="topbar-actions">
          <button className="btn btn-ghost" onClick={() => setIssuesOpen(true)}>
            issues
          </button>
        </div>
      </header>
      <ContextBar />
      <main className="main">
        <div className="page-scroll">
          <Suspense fallback={<div className="card">loading…</div>}>
            <Routes>
              <Route path="/overview" element={<ErrorBoundary label="Overview"><Overview /></ErrorBoundary>} />
              <Route path="/supervisor" element={<ErrorBoundary label="Supervisor"><Supervisor /></ErrorBoundary>} />
              <Route path="/runs" element={<ErrorBoundary label="Runs"><Runs /></ErrorBoundary>} />
              <Route path="/instances" element={<ErrorBoundary label="Instances"><Instances /></ErrorBoundary>} />
              <Route path="/reasoning" element={<ErrorBoundary label="Reasoning"><Reasoning /></ErrorBoundary>} />
              <Route path="/build" element={<ErrorBoundary label="Build Lab"><BuildLab /></ErrorBoundary>} />
              <Route path="/skills" element={<ErrorBoundary label="Skills"><Skills /></ErrorBoundary>} />
              <Route path="/spy" element={<ErrorBoundary label="Spy"><Spy /></ErrorBoundary>} />
              <Route path="/stream" element={<ErrorBoundary label="Stream"><Stream /></ErrorBoundary>} />
              <Route path="/broadcast" element={<ErrorBoundary label="Broadcast"><Stream /></ErrorBoundary>} />
              <Route path="/cctv" element={<ErrorBoundary label="CCTV"><CCTV3D /></ErrorBoundary>} />
            </Routes>
          </Suspense>
        </div>
      </main>
      <IssuesDrawer open={issuesOpen} onClose={() => setIssuesOpen(false)} />
      <ContextDrawer />
      <CommandPalette />
      <FrontendHealthOverlay />
    </div>
  );
}
