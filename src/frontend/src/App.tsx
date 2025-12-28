import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import ContextBar from "./components/ContextBar";
import CommandPalette from "./components/CommandPalette";
import ContextDrawer from "./components/ContextDrawer";
import IssuesDrawer from "./components/IssuesDrawer";
import ErrorBoundary from "./components/ErrorBoundary";
import FrontendHealthOverlay from "./components/FrontendHealthOverlay";
import OnboardingModal from "./components/Onboarding/OnboardingModal";
import WelcomeWizard from "./components/Onboarding/WelcomeWizard";
import Lobby from "./pages/Lobby";
import NeuralInterface from "./pages/NeuralInterface";
import NeuroSynaptic from "./pages/NeuroSynaptic";
import Runs from "./pages/Runs";
import Instances from "./pages/Instances";
import Skills from "./pages/Skills";
import Reasoning from "./pages/Reasoning";
import Spy from "./pages/Spy";
import Stream from "./pages/Stream";
import Supervisor from "./pages/Supervisor";
import Analytics from "./pages/Analytics";
import Discovery from "./pages/Discovery";
import Settings from "./pages/Settings";
import useLaunchConfig from "./hooks/useLaunchConfig";
import useLocalStorageState from "./hooks/useLocalStorageState";
import { tauriInvoke, type DiscoveryStatus } from "./lib/tauri_api";
import { useTauriRuntime } from "./tauri/RuntimeProvider";

const BuildLab = lazy(() => import("./pages/BuildLab"));
const CCTV3D = lazy(() => import("./pages/CCTV3D"));

export default function App() {
  const loc = useLocation();
  const { tauriReady, omegaRunning } = useTauriRuntime();
  const [issuesOpen, setIssuesOpen] = useState(false);
  const [launchBusy, setLaunchBusy] = useState(false);
  const [launchOpen, setLaunchOpen] = useState(false);
  const [advancedUi, setAdvancedUi] = useLocalStorageState<boolean>("mb:advancedUi", false, {
    serialize: (v) => (v ? "1" : "0"),
    deserialize: (raw) => raw === "1",
  });
  const [launchCfg, setLaunchCfg] = useLaunchConfig();
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [discoveryStatus, setDiscoveryStatus] = useState<DiscoveryStatus | null>(null);
  const omegaRef = useRef<HTMLDivElement | null>(null);

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

  useEffect(() => {
    if (!tauriReady) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await tauriInvoke<DiscoveryStatus>("discovery_status", { env_id: launchCfg.envId });
        if (!cancelled) setDiscoveryStatus(status);
      } catch {
        if (!cancelled) setDiscoveryStatus(null);
      }
    };
    poll();
    const t = window.setInterval(poll, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [tauriReady, launchCfg.envId]);

  useEffect(() => {
    if (!launchOpen) return undefined;
    const onClick = (evt: MouseEvent) => {
      if (!omegaRef.current) return;
      if (!omegaRef.current.contains(evt.target as Node)) {
        setLaunchOpen(false);
      }
    };
    const onKey = (evt: KeyboardEvent) => {
      if (evt.key === "Escape") setLaunchOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onKey);
    };
  }, [launchOpen]);

  const startOmega = async () => {
    if (!tauriReady || launchBusy) return;
    setLaunchBusy(true);
    setLaunchError(null);
    try {
      await tauriInvoke("start_omega", {
        mode: launchCfg.mode,
        workers: launchCfg.workers,
        env_id: launchCfg.envId,
        synthetic_eye: launchCfg.syntheticEye,
        synthetic_eye_lockstep: launchCfg.syntheticEyeLockstep,
        obs_backend: launchCfg.obsBackend || null,
        use_discovered_actions: launchCfg.useDiscoveredActions,
        silicon_cortex: launchCfg.siliconCortex,
      });
    } catch (e: any) {
      setLaunchError(String(e?.message ?? e));
    } finally {
      setLaunchBusy(false);
    }
  };

  const stopOmega = async () => {
    if (!tauriReady || launchBusy) return;
    setLaunchBusy(true);
    setLaunchError(null);
    try {
      await tauriInvoke("stop_omega");
    } catch (e: any) {
      setLaunchError(String(e?.message ?? e));
    } finally {
      setLaunchBusy(false);
    }
  };

  const isProdTauri = tauriReady && !import.meta.env.DEV;

  const nav = useMemo(
    () => [
      { to: "/", label: "Home" },
      { to: "/agents", label: "Agents" },
      { to: "/discovery", label: "Discovery" },
      { to: "/analytics", label: "Analytics" },
      { to: "/settings", label: "Settings" },
    ],
    [],
  );

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

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">{isProdTauri ? "MetaBonk" : "MetaBonk Dev UI"}</div>
        <nav className="nav">
          {nav.map((n) => (
            <Link
              key={n.to}
              to={`${n.to}${loc.search}`}
              className={loc.pathname === n.to || (n.to !== "/" && loc.pathname.startsWith(n.to)) ? "active" : ""}
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="topbar-actions">
          {tauriReady ? (
            <div className="omega-ctl" ref={omegaRef}>
              <button className="btn btn-ghost" onClick={() => setLaunchOpen((v) => !v)}>
                <span className={`omega-dot ${omegaRunning ? "live" : "idle"}`} />
                omega: {omegaRunning ? "live" : "idle"}
              </button>
              {launchOpen ? (
                <div className="omega-popover" role="dialog" aria-label="Omega launch control">
                  <div className="omega-row" style={{ flexWrap: "wrap" }}>
                    <label>
                      <span>mode</span>
                      <select
                        value={launchCfg.mode}
                        onChange={(e) => setLaunchCfg((prev) => ({ ...prev, mode: e.target.value as any }))}
                      >
                        <option value="train">train</option>
                        <option value="play">play</option>
                        <option value="dream">dream</option>
                      </select>
                    </label>
                    <label>
                      <span>workers</span>
                      <input
                        type="number"
                        min={1}
                        value={launchCfg.workers}
                        onChange={(e) =>
                          setLaunchCfg((prev) => ({ ...prev, workers: Math.max(1, Number(e.target.value || 1)) }))
                        }
                      />
                    </label>
                    <label style={{ minWidth: 220 }}>
                      <span>env id</span>
                      <input
                        value={launchCfg.envId}
                        onChange={(e) => setLaunchCfg((prev) => ({ ...prev, envId: e.target.value }))}
                        placeholder="MegaBonk"
                      />
                    </label>
                  </div>
                  <div className="omega-row" style={{ flexWrap: "wrap" }}>
                    <label>
                      <span>synthetic eye</span>
                      <select
                        value={launchCfg.syntheticEye ? "on" : "off"}
                        onChange={(e) => setLaunchCfg((prev) => ({ ...prev, syntheticEye: e.target.value === "on" }))}
                      >
                        <option value="on">enabled</option>
                        <option value="off">disabled</option>
                      </select>
                    </label>
                    <label>
                      <span>lockstep</span>
                      <select
                        value={launchCfg.syntheticEyeLockstep ? "on" : "off"}
                        onChange={(e) =>
                          setLaunchCfg((prev) => ({ ...prev, syntheticEyeLockstep: e.target.value === "on" }))
                        }
                        disabled={!launchCfg.syntheticEye}
                      >
                        <option value="off">off</option>
                        <option value="on">on</option>
                      </select>
                    </label>
                    <label>
                      <span>obs backend</span>
                      <select
                        value={launchCfg.obsBackend}
                        onChange={(e) => setLaunchCfg((prev) => ({ ...prev, obsBackend: e.target.value as any }))}
                      >
                        <option value="">auto</option>
                        <option value="detections">detections</option>
                        <option value="pixels">pixels</option>
                        <option value="hybrid">hybrid</option>
                      </select>
                    </label>
                    <label>
                      <span>actions</span>
                      <select
                        value={launchCfg.useDiscoveredActions ? "learned" : "raw"}
                        onChange={(e) =>
                          setLaunchCfg((prev) => ({ ...prev, useDiscoveredActions: e.target.value === "learned" }))
                        }
                      >
                        <option value="learned">learned</option>
                        <option value="raw">raw</option>
                      </select>
                    </label>
                    <label>
                      <span>torch.compile</span>
                      <select
                        value={launchCfg.siliconCortex ? "on" : "off"}
                        onChange={(e) => setLaunchCfg((prev) => ({ ...prev, siliconCortex: e.target.value === "on" }))}
                      >
                        <option value="off">off</option>
                        <option value="on">on</option>
                      </select>
                    </label>
                  </div>
                  <div className="omega-actions">
                    <button className="btn btn-compact" disabled={launchBusy || omegaRunning} onClick={startOmega}>
                      {launchBusy ? "starting..." : "start"}
                    </button>
                    <button className="btn btn-compact btn-ghost" disabled={launchBusy || !omegaRunning} onClick={stopOmega}>
                      stop
                    </button>
                    <Link className="btn btn-compact btn-ghost" to="/stream">
                      stream
                    </Link>
                    <Link className="btn btn-compact btn-ghost" to="/settings">
                      settings
                    </Link>
                  </div>
                  <div className="muted" style={{ marginTop: 8 }}>
                    discovery:{" "}
                    {discoveryStatus?.ready_for_training ? (
                      <span className="badge">ready</span>
                    ) : (
                      <span className="badge warn">missing</span>
                    )}
                  </div>
                  {launchError ? <div className="omega-error">{launchError}</div> : null}
                </div>
              ) : null}
            </div>
          ) : null}
          <button
            className="btn btn-ghost"
            onClick={() => {
              setAdvancedUi((v) => !v);
            }}
          >
            {advancedUi ? "basic" : "advanced"}
          </button>
          <button className="btn btn-ghost" onClick={() => setIssuesOpen(true)}>
            issues
          </button>
        </div>
      </header>
      {advancedUi ? <ContextBar /> : null}
      <main className="main">
        <div className="page-scroll">
          <Suspense fallback={<div className="card">loading…</div>}>
            <Routes>
              <Route path="/" element={<ErrorBoundary label="Home"><Lobby /></ErrorBoundary>} />
              <Route path="/agents" element={<ErrorBoundary label="Agents"><NeuralInterface /></ErrorBoundary>} />
              <Route path="/discovery" element={<ErrorBoundary label="Discovery"><Discovery /></ErrorBoundary>} />
              <Route path="/analytics" element={<ErrorBoundary label="Analytics"><Analytics /></ErrorBoundary>} />
              <Route path="/settings" element={<ErrorBoundary label="Settings"><Settings /></ErrorBoundary>} />
              <Route path="/neural" element={<Navigate to="/agents" replace />} />
              <Route path="/lab" element={<Navigate to="/analytics" replace />} />
              <Route path="/knowledge" element={<Navigate to="/analytics" replace />} />
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
              <Route path="/brain" element={<ErrorBoundary label="NeuroSynaptic"><NeuroSynaptic /></ErrorBoundary>} />
            </Routes>
          </Suspense>
        </div>
      </main>
      <IssuesDrawer open={issuesOpen} onClose={() => setIssuesOpen(false)} />
      {advancedUi ? <ContextDrawer /> : null}
      {advancedUi ? <CommandPalette /> : null}
      {advancedUi ? <FrontendHealthOverlay /> : null}
      <OnboardingModal />
      <WelcomeWizard />
    </div>
  );
}
