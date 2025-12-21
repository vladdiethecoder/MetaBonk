import { Suspense, lazy } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import ContextBar from "./components/ContextBar";
import Overview from "./pages/Overview";
import Runs from "./pages/Runs";
import Instances from "./pages/Instances";
import Skills from "./pages/Skills";
import Spy from "./pages/Spy";
import Stream from "./pages/Stream";

const BuildLab = lazy(() => import("./pages/BuildLab"));

export default function App() {
  const loc = useLocation();
  // Stream overlay wants a clean fullscreen surface (OBS browser source).
  if (loc.pathname.startsWith("/stream")) {
    return <Stream />;
  }
  const nav = [
    { to: "/", label: "Overview" },
    { to: "/runs", label: "Runs" },
    { to: "/instances", label: "Instances" },
    { to: "/build", label: "Build Lab" },
    { to: "/skills", label: "Skills" },
    { to: "/spy", label: "Spy" },
    { to: "/stream", label: "Stream" },
  ];
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">MetaBonk Dev UI</div>
        <nav className="nav">
          {nav.map((n) => (
            <Link key={n.to} to={n.to} className={loc.pathname === n.to ? "active" : ""}>
              {n.label}
            </Link>
          ))}
        </nav>
      </header>
      <ContextBar />
      <main className="main">
        <Suspense fallback={<div className="card">loading…</div>}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/instances" element={<Instances />} />
            <Route path="/build" element={<BuildLab />} />
            <Route path="/skills" element={<Skills />} />
            <Route path="/spy" element={<Spy />} />
            <Route path="/stream" element={<Stream />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
