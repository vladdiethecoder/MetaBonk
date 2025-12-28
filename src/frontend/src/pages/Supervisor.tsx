import { useEffect, useMemo, useState } from "react";
import PageShell from "../components/PageShell";
import { isTauri } from "../lib/tauri";

type Mode = "train" | "play" | "dream";
type ObsBackend = "" | "detections" | "pixels" | "hybrid";
type DiscoveryAdapter = "mock" | "synthetic-eye";

type LogLine = {
  ts: number;
  stream: "stdout" | "stderr";
  line: string;
};

type DiscoveryStatus = {
  env_id: string;
  cache_dir: string;
  phase_0_complete: boolean;
  phase_1_complete: boolean;
  phase_2_complete: boolean;
  phase_3_complete: boolean;
  ready_for_training: boolean;
  artifacts?: Record<string, string>;
  has_suggested_env?: boolean;
  has_ppo_config?: boolean;
};

type BenchResult = {
  ok?: boolean;
  fps?: number;
  per_frame_ms?: { mean?: number; p50?: number; p95?: number };
  frames?: number;
  elapsed_s?: number;
  mode?: string;
  audit_log?: string;
  sock?: string;
};

export default function Supervisor() {
  const [tauriReady, setTauriReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<Mode>("train");
  const [workers, setWorkers] = useState(1);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [envId, setEnvId] = useState("MegaBonk");
  const [syntheticEye, setSyntheticEye] = useState(true);
  const [syntheticEyeLockstep, setSyntheticEyeLockstep] = useState(false);
  const [obsBackend, setObsBackend] = useState<ObsBackend>("");
  const [useDiscoveredActions, setUseDiscoveredActions] = useState(true);
  const [siliconCortex, setSiliconCortex] = useState(false);

  const [discoveryRunning, setDiscoveryRunning] = useState(false);
  const [discoveryAdapter, setDiscoveryAdapter] = useState<DiscoveryAdapter>("mock");
  const [discoveryBudget, setDiscoveryBudget] = useState(2000);
  const [discoveryHoldFrames, setDiscoveryHoldFrames] = useState(30);
  const [discoveryLogs, setDiscoveryLogs] = useState<LogLine[]>([]);
  const [discoveryStatus, setDiscoveryStatus] = useState<DiscoveryStatus | null>(null);
  const [benchBusy, setBenchBusy] = useState(false);
  const [benchResult, setBenchResult] = useState<BenchResult | null>(null);

  const lines = useMemo(() => logs.slice(-400), [logs]);
  const discoveryLines = useMemo(() => discoveryLogs.slice(-260), [discoveryLogs]);

  useEffect(() => {
    let unlistenOut: null | (() => void) = null;
    let unlistenErr: null | (() => void) = null;
    let unlistenExit: null | (() => void) = null;
    let unlistenDiscOut: null | (() => void) = null;
    let unlistenDiscErr: null | (() => void) = null;
    let unlistenDiscExit: null | (() => void) = null;
    let cancelled = false;

    const init = async () => {
      const ok = isTauri();
      setTauriReady(ok);
      if (!ok) return;
      try {
        const [{ listen }, { invoke }] = await Promise.all([
          import("@tauri-apps/api/event"),
          import("@tauri-apps/api/core"),
        ]);

        unlistenOut = await listen<string>("omega-stdout", (event) => {
          setLogs((prev) => [...prev.slice(-600), { ts: Date.now(), stream: "stdout", line: event.payload }]);
        });
        unlistenErr = await listen<string>("omega-stderr", (event) => {
          setLogs((prev) => [...prev.slice(-600), { ts: Date.now(), stream: "stderr", line: event.payload }]);
        });
        unlistenExit = await listen<boolean>("omega-exit", () => {
          setRunning(false);
          setLogs((prev) => [
            ...prev.slice(-600),
            { ts: Date.now(), stream: "stderr", line: "[tauri] omega exited" },
          ]);
        });

        unlistenDiscOut = await listen<string>("discovery-stdout", (event) => {
          setDiscoveryLogs((prev) => [...prev.slice(-500), { ts: Date.now(), stream: "stdout", line: event.payload }]);
        });
        unlistenDiscErr = await listen<string>("discovery-stderr", (event) => {
          setDiscoveryLogs((prev) => [...prev.slice(-500), { ts: Date.now(), stream: "stderr", line: event.payload }]);
        });
        unlistenDiscExit = await listen<boolean>("discovery-exit", () => {
          setDiscoveryRunning(false);
          setDiscoveryLogs((prev) => [
            ...prev.slice(-500),
            { ts: Date.now(), stream: "stderr", line: "[tauri] discovery exited" },
          ]);
        });

        const isRunning = await invoke<boolean>("omega_running");
        if (!cancelled) setRunning(Boolean(isRunning));

        const isDisc = await invoke<boolean>("discovery_running");
        if (!cancelled) setDiscoveryRunning(Boolean(isDisc));
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message ?? e));
      }
    };

    init();
    return () => {
      cancelled = true;
      try {
        unlistenOut?.();
      } catch {}
      try {
        unlistenErr?.();
      } catch {}
      try {
        unlistenExit?.();
      } catch {}
      try {
        unlistenDiscOut?.();
      } catch {}
      try {
        unlistenDiscErr?.();
      } catch {}
      try {
        unlistenDiscExit?.();
      } catch {}
    };
  }, []);

  useEffect(() => {
    if (!tauriReady) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const [status, omegaOk, discOk] = await Promise.all([
          invoke<DiscoveryStatus>("discovery_status", { env_id: envId }),
          invoke<boolean>("omega_running"),
          invoke<boolean>("discovery_running"),
        ]);
        if (!cancelled) setDiscoveryStatus(status);
        if (!cancelled) setRunning(Boolean(omegaOk));
        if (!cancelled) setDiscoveryRunning(Boolean(discOk));
      } catch {
        // Ignore: status is best-effort.
      }
    };
    poll();
    const t = window.setInterval(poll, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [tauriReady, envId]);

  const start = async () => {
    setError(null);
    if (!tauriReady) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_omega", {
        mode,
        workers,
        env_id: envId,
        synthetic_eye: syntheticEye,
        synthetic_eye_lockstep: syntheticEyeLockstep,
        obs_backend: obsBackend || null,
        use_discovered_actions: useDiscoveredActions,
        silicon_cortex: siliconCortex,
      });
      setRunning(true);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const stop = async () => {
    setError(null);
    if (!tauriReady) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("stop_omega");
      setRunning(false);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const startDiscovery = async () => {
    setError(null);
    if (!tauriReady) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_discovery", {
        env_id: envId,
        env_adapter: discoveryAdapter,
        exploration_budget: discoveryBudget,
        hold_frames: discoveryHoldFrames,
      });
      setDiscoveryRunning(true);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const stopDiscovery = async () => {
    setError(null);
    if (!tauriReady) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("stop_discovery");
      setDiscoveryRunning(false);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const runBench = async () => {
    setError(null);
    if (!tauriReady || benchBusy) return;
    setBenchBusy(true);
    setBenchResult(null);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const res = await invoke<BenchResult>("run_synthetic_eye_bench", {
        width: 1280,
        height: 720,
        lockstep: syntheticEyeLockstep,
        seconds: 3.0,
        full_bridge: true,
      });
      setBenchResult(res);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBenchBusy(false);
    }
  };

  return (
    <PageShell title="Supervisor" subtitle="Tauri-controlled Omega process (dev)">
      <div className="grid cols-2">
        <div className="card">
          <div className="row" style={{ gap: 12, alignItems: "center" }}>
            <div className="pill">{tauriReady ? "tauri" : "browser"}</div>
            <div className="pill">{running ? "running" : "stopped"}</div>
          </div>
          <div className="muted" style={{ marginTop: 10 }}>
            Launch configuration is applied only when starting Omega from this page.
          </div>
          <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <label className="field">
              <div className="label">mode</div>
              <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
                <option value="train">train</option>
                <option value="play">play</option>
                <option value="dream">dream</option>
              </select>
            </label>
            <label className="field">
              <div className="label">env</div>
              <input value={envId} onChange={(e) => setEnvId(e.target.value)} placeholder="MegaBonk" />
            </label>
            <label className="field">
              <div className="label">workers</div>
              <input
                type="number"
                min={1}
                max={64}
                value={workers}
                onChange={(e) => setWorkers(Math.max(1, Math.min(64, Number(e.target.value || 1))))}
              />
            </label>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10, marginTop: 12 }}>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">Synthetic Eye</div>
              <select value={syntheticEye ? "on" : "off"} onChange={(e) => setSyntheticEye(e.target.value === "on")}>
                <option value="on">enabled</option>
                <option value="off">disabled</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">obs backend</div>
              <select value={obsBackend} onChange={(e) => setObsBackend(e.target.value as ObsBackend)}>
                <option value="">auto</option>
                <option value="detections">detections</option>
                <option value="pixels">pixels</option>
                <option value="hybrid">hybrid</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">lockstep</div>
              <select
                value={syntheticEyeLockstep ? "on" : "off"}
                onChange={(e) => setSyntheticEyeLockstep(e.target.value === "on")}
              >
                <option value="off">off</option>
                <option value="on">on</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">SiliconCortex</div>
              <select value={siliconCortex ? "on" : "off"} onChange={(e) => setSiliconCortex(e.target.value === "on")}>
                <option value="off">off</option>
                <option value="on">torch.compile</option>
              </select>
            </label>
          </div>
          <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">discovered actions</div>
              <select
                value={useDiscoveredActions ? "on" : "off"}
                onChange={(e) => setUseDiscoveredActions(e.target.value === "on")}
              >
                <option value="on">use (if cached)</option>
                <option value="off">disable</option>
              </select>
            </label>
            <div className="pill" style={{ alignSelf: "flex-end" }}>
              discovery {discoveryStatus?.ready_for_training ? "ready" : "missing"}
            </div>
          </div>
          <div className="row" style={{ gap: 12, marginTop: 12 }}>
            <button className="btn" disabled={!tauriReady || running} onClick={start}>
              start omega
            </button>
            <button className="btn btn-ghost" disabled={!tauriReady || !running} onClick={stop}>
              stop (kills stragglers)
            </button>
          </div>
          {!tauriReady && (
            <div className="muted" style={{ marginTop: 12 }}>
              Launch via <code>npx tauri dev</code> to enable start/stop controls.
            </div>
          )}
          {discoveryStatus && (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="label">autonomous discovery</div>
              <div className="kv" style={{ marginTop: 8 }}>
                <div className="k">Phase 0</div>
                <div className="v">{discoveryStatus.phase_0_complete ? "✓" : "—"}</div>
                <div className="k">Phase 1</div>
                <div className="v">{discoveryStatus.phase_1_complete ? "✓" : "—"}</div>
                <div className="k">Phase 2</div>
                <div className="v">{discoveryStatus.phase_2_complete ? "✓" : "—"}</div>
                <div className="k">Phase 3</div>
                <div className="v">{discoveryStatus.phase_3_complete ? "✓" : "—"}</div>
              </div>
              <div className="muted" style={{ marginTop: 8, wordBreak: "break-all" }}>
                {discoveryStatus.cache_dir}
              </div>
              <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
                <label className="field" style={{ margin: 0 }}>
                  <div className="label">adapter</div>
                  <select value={discoveryAdapter} onChange={(e) => setDiscoveryAdapter(e.target.value as DiscoveryAdapter)}>
                    <option value="mock">mock</option>
                    <option value="synthetic-eye">synthetic-eye</option>
                  </select>
                </label>
                <label className="field" style={{ margin: 0 }}>
                  <div className="label">budget</div>
                  <input
                    type="number"
                    min={50}
                    max={20000}
                    value={discoveryBudget}
                    onChange={(e) => setDiscoveryBudget(Math.max(50, Math.min(20000, Number(e.target.value || 2000))))}
                  />
                </label>
                <label className="field" style={{ margin: 0 }}>
                  <div className="label">hold frames</div>
                  <input
                    type="number"
                    min={1}
                    max={120}
                    value={discoveryHoldFrames}
                    onChange={(e) => setDiscoveryHoldFrames(Math.max(1, Math.min(120, Number(e.target.value || 30))))}
                  />
                </label>
                <button className="btn" disabled={!tauriReady || discoveryRunning} onClick={startDiscovery}>
                  run discovery (0-3)
                </button>
                <button className="btn btn-ghost" disabled={!tauriReady || !discoveryRunning} onClick={stopDiscovery}>
                  stop discovery
                </button>
              </div>
            </div>
          )}
          <div className="card" style={{ marginTop: 12 }}>
            <div className="row-between" style={{ alignItems: "center" }}>
              <div className="label">synthetic eye bench</div>
              <button className="btn btn-ghost" disabled={!tauriReady || benchBusy} onClick={runBench}>
                {benchBusy ? "running…" : "run 3s"}
              </button>
            </div>
            {!benchResult ? (
              <div className="muted" style={{ marginTop: 8 }}>
                Runs `scripts/synthetic_eye_bench.py` and returns ingest FPS (DMABUF+fence → CUDA import).
              </div>
            ) : (
              <div className="kv" style={{ marginTop: 8 }}>
                <div className="k">fps</div>
                <div className="v">{benchResult.fps ? benchResult.fps.toFixed(1) : "—"}</div>
                <div className="k">p50</div>
                <div className="v">{benchResult.per_frame_ms?.p50 != null ? `${benchResult.per_frame_ms.p50.toFixed(2)}ms` : "—"}</div>
                <div className="k">p95</div>
                <div className="v">{benchResult.per_frame_ms?.p95 != null ? `${benchResult.per_frame_ms.p95.toFixed(2)}ms` : "—"}</div>
                <div className="k">mode</div>
                <div className="v">{benchResult.mode ?? "—"}</div>
              </div>
            )}
          </div>
          {error && (
            <div className="card" style={{ marginTop: 12, borderColor: "#ff8b5a" }}>
              <div className="label">error</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{error}</div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
            <div className="label">omega logs</div>
            <button className="btn btn-ghost" onClick={() => setLogs([])}>
              clear
            </button>
          </div>
          <pre style={{ marginTop: 8, maxHeight: 520, overflow: "auto" }}>
            {lines.map((l) => `${new Date(l.ts).toLocaleTimeString()} ${l.stream === "stderr" ? "!" : " "} ${l.line}`).join("\n")}
          </pre>

          <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginTop: 12 }}>
            <div className="label">discovery logs</div>
            <button className="btn btn-ghost" onClick={() => setDiscoveryLogs([])}>
              clear
            </button>
          </div>
          <pre style={{ marginTop: 8, maxHeight: 320, overflow: "auto" }}>
            {discoveryLines
              .map((l) => `${new Date(l.ts).toLocaleTimeString()} ${l.stream === "stderr" ? "!" : " "} ${l.line}`)
              .join("\n")}
          </pre>
        </div>
      </div>
    </PageShell>
  );
}
