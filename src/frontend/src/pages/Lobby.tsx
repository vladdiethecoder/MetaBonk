import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { fetchOverviewHealth, fetchOverviewIssues, fetchStatus, fetchWorkers } from "../api";
import useLaunchConfig, { type LaunchMode, type ObsBackend } from "../hooks/useLaunchConfig";
import useLocalStorageState from "../hooks/useLocalStorageState";
import { fmtFixed, fmtNum, fmtPct01, timeAgo } from "../lib/format";
import { tauriInvoke, type BenchResult, type DiscoveryStatus } from "../lib/tauri_api";
import { useTauriRuntime } from "../tauri/RuntimeProvider";

export default function Lobby() {
  const {
    tauriReady,
    omegaRunning,
    discoveryRunning,
    omegaLogs,
    discoveryLogs,
    lastError,
    clearOmegaLogs,
    clearDiscoveryLogs,
  } = useTauriRuntime();
  const [launchCfg, setLaunchCfg] = useLaunchConfig();
  const [launchBusy, setLaunchBusy] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [benchBusy, setBenchBusy] = useState(false);
  const [benchResult, setBenchResult] = useState<BenchResult | null>(null);
  const [discoveryStatus, setDiscoveryStatus] = useState<DiscoveryStatus | null>(null);
  const [discoveryError, setDiscoveryError] = useState<string | null>(null);
  const [discoveryAdapter, setDiscoveryAdapter] = useLocalStorageState<"mock" | "synthetic-eye">(
    "mb:discoveryAdapter",
    "mock",
  );
  const [discoveryBudget, setDiscoveryBudget] = useLocalStorageState<number>("mb:discoveryBudget", 2000, {
    serialize: (v) => String(Number(v || 0)),
    deserialize: (raw) => Number(raw || 0),
  });
  const [discoveryHoldFrames, setDiscoveryHoldFrames] = useLocalStorageState<number>("mb:discoveryHoldFrames", 30, {
    serialize: (v) => String(Number(v || 0)),
    deserialize: (raw) => Number(raw || 0),
  });
  const omegaLines = useMemo(() => omegaLogs.slice(-260), [omegaLogs]);
  const discoveryLines = useMemo(() => discoveryLogs.slice(-200), [discoveryLogs]);

  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 3000 });
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 3000 });
  const healthQ = useQuery({
    queryKey: ["overviewHealth"],
    queryFn: () => fetchOverviewHealth(240),
    refetchInterval: 5000,
  });
  const issuesQ = useQuery({
    queryKey: ["overviewIssues"],
    queryFn: () => fetchOverviewIssues(600),
    refetchInterval: 6000,
  });

  const workers = Object.values(workersQ.data ?? {});
  const running = workers.filter((w) => w.status === "running");
  const lastSeen = workers
    .map((w) => w.ts)
    .filter(Boolean)
    .sort((a, b) => b - a)[0];
  const health = healthQ.data;
  const issues = issuesQ.data ?? [];
  const streamTotal = health ? health.stream.ok + health.stream.stale + health.stream.missing : 0;
  const streamOkPct = health && streamTotal ? Math.round((health.stream.ok / streamTotal) * 100) : null;

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

  const startOmega = async () => {
    if (!tauriReady || launchBusy) return;
    setLaunchBusy(true);
    setLaunchError(null);
    try {
      toast.message("Starting Omega…");
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
      toast.success("Omega started");
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      setLaunchError(msg);
      toast.error("Failed to start Omega", { description: msg });
    } finally {
      setLaunchBusy(false);
    }
  };

  const stopOmega = async () => {
    if (!tauriReady || launchBusy) return;
    setLaunchBusy(true);
    setLaunchError(null);
    try {
      toast.message("Stopping Omega…");
      await tauriInvoke("stop_omega");
      toast.success("Omega stopped");
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      setLaunchError(msg);
      toast.error("Failed to stop Omega", { description: msg });
    } finally {
      setLaunchBusy(false);
    }
  };

  const runBench = async () => {
    if (!tauriReady || benchBusy) return;
    setDiscoveryError(null);
    setBenchBusy(true);
    setBenchResult(null);
    try {
      toast.message("Running Synthetic Eye bench…");
      const res = await tauriInvoke<BenchResult>("run_synthetic_eye_bench", {
        width: 1280,
        height: 720,
        lockstep: launchCfg.syntheticEyeLockstep,
        seconds: 3.0,
        full_bridge: true,
      });
      setBenchResult(res);
      toast.success("Synthetic Eye bench complete", { description: res?.fps ? `${res.fps.toFixed(1)} FPS` : "done" });
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      setDiscoveryError(msg);
      toast.error("Synthetic Eye bench failed", { description: msg });
    } finally {
      setBenchBusy(false);
    }
  };

  const startDiscovery = async () => {
    if (!tauriReady || discoveryRunning) return;
    setDiscoveryError(null);
    try {
      toast.message("Starting discovery…");
      await tauriInvoke("start_discovery", {
        env_id: launchCfg.envId,
        env_adapter: discoveryAdapter,
        exploration_budget: discoveryBudget,
        hold_frames: discoveryHoldFrames,
      });
      toast.success("Discovery started");
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      setDiscoveryError(msg);
      toast.error("Failed to start discovery", { description: msg });
    }
  };

  const stopDiscovery = async () => {
    if (!tauriReady || !discoveryRunning) return;
    setDiscoveryError(null);
    try {
      toast.message("Stopping discovery…");
      await tauriInvoke("stop_discovery");
      toast.success("Discovery stopped");
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      setDiscoveryError(msg);
      toast.error("Failed to stop discovery", { description: msg });
    }
  };

  return (
    <div className="page lobby-page">
      <section className="card lobby-hero">
        <div>
          <h1>Lobby</h1>
          <p className="muted">
            Command center for health, launch control, and discovery artifacts. The goal: zero CLI clicks for your day-to-day loop.
          </p>
          <div className="lobby-links">
            <Link to="/neural" className="btn btn-ghost">
              Neural Interface
            </Link>
            <Link to="/lab" className="btn btn-ghost">
              Laboratory
            </Link>
            <Link to="/codex" className="btn btn-ghost">
              Codex
            </Link>
            <Link to="/neural/broadcast" className="btn btn-ghost">
              Broadcast
            </Link>
          </div>
        </div>
        <div className="kpis kpis-wrap">
          <div className="kpi">
            <div className="label">System</div>
            <div className="value">{statusQ.isError ? "offline" : "live"}</div>
          </div>
          <div className="kpi">
            <div className="label">Workers</div>
            <div className="value">{fmtNum(workers.length)}</div>
          </div>
          <div className="kpi">
            <div className="label">Running</div>
            <div className="value">{fmtNum(running.length)}</div>
          </div>
          <div className="kpi">
            <div className="label">Last pulse</div>
            <div className="value">{lastSeen ? timeAgo(lastSeen) : "—"}</div>
          </div>
        </div>
      </section>

      <div className="lobby-grid">
        <section className="card flex-card lobby-launch">
          <h3>Supervisor</h3>
          <div className="muted">
            {tauriReady ? (
              "Tauri-controlled processes (Omega + Discovery)."
            ) : (
              <>
                Launch via <code>./start --tauri</code> (runs <code>npx tauri dev</code>) to enable process control.
              </>
            )}
          </div>

          <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <span className="pill">{tauriReady ? "tauri" : "browser"}</span>
            <span className="pill">{omegaRunning ? "omega live" : "omega idle"}</span>
            <span className="pill">{discoveryRunning ? "discovery running" : "discovery idle"}</span>
          </div>

          <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">mode</div>
              <select value={launchCfg.mode} onChange={(e) => setLaunchCfg((prev) => ({ ...prev, mode: e.target.value as LaunchMode }))}>
                <option value="train">train</option>
                <option value="play">play</option>
                <option value="dream">dream</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">workers</div>
              <input
                type="number"
                min={1}
                max={64}
                value={launchCfg.workers}
                onChange={(e) => setLaunchCfg((prev) => ({ ...prev, workers: Number(e.target.value || 1) }))}
              />
            </label>
            <label className="field" style={{ margin: 0, minWidth: 240 }}>
              <div className="label">env id</div>
              <input value={launchCfg.envId} onChange={(e) => setLaunchCfg((prev) => ({ ...prev, envId: e.target.value }))} />
            </label>
          </div>

          <div className="grid" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10, marginTop: 12 }}>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">Synthetic Eye</div>
              <select value={launchCfg.syntheticEye ? "on" : "off"} onChange={(e) => setLaunchCfg((prev) => ({ ...prev, syntheticEye: e.target.value === "on" }))}>
                <option value="on">enabled</option>
                <option value="off">disabled</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">lockstep</div>
              <select
                value={launchCfg.syntheticEyeLockstep ? "on" : "off"}
                onChange={(e) => setLaunchCfg((prev) => ({ ...prev, syntheticEyeLockstep: e.target.value === "on" }))}
                disabled={!launchCfg.syntheticEye}
              >
                <option value="off">off</option>
                <option value="on">on</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">obs backend</div>
              <select value={launchCfg.obsBackend} onChange={(e) => setLaunchCfg((prev) => ({ ...prev, obsBackend: e.target.value as ObsBackend }))}>
                <option value="">auto</option>
                <option value="detections">detections</option>
                <option value="pixels">pixels</option>
                <option value="hybrid">hybrid</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">SiliconCortex</div>
              <select value={launchCfg.siliconCortex ? "on" : "off"} onChange={(e) => setLaunchCfg((prev) => ({ ...prev, siliconCortex: e.target.value === "on" }))}>
                <option value="off">off</option>
                <option value="on">torch.compile</option>
              </select>
            </label>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">actions</div>
              <select value={launchCfg.useDiscoveredActions ? "learned" : "raw"} onChange={(e) => setLaunchCfg((prev) => ({ ...prev, useDiscoveredActions: e.target.value === "learned" }))}>
                <option value="learned">use discovered</option>
                <option value="raw">raw</option>
              </select>
            </label>
            <div className="pill" style={{ alignSelf: "flex-end" }}>
              discovery {discoveryStatus?.ready_for_training ? "ready" : "missing"}
            </div>
          </div>

          <div className="row lobby-actions" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <button className="btn" disabled={!tauriReady || omegaRunning || launchBusy} onClick={startOmega}>
              {launchBusy ? "starting…" : "start omega"}
            </button>
            <button className="btn btn-ghost" disabled={!tauriReady || !omegaRunning || launchBusy} onClick={stopOmega}>
              stop omega
            </button>
            <Link className="btn btn-ghost" to="/neural">
              enter neural
            </Link>
          </div>

          {launchError ? (
            <div className="card" style={{ marginTop: 12, borderColor: "#ff8b5a" }}>
              <div className="label">omega error</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{launchError}</div>
            </div>
          ) : null}
        </section>

        <section className="card flex-card lobby-discovery">
          <h3>Autonomous Discovery</h3>
          <div className="muted">Phases 0–3: enumerate inputs, measure effects, cluster semantics, construct action space.</div>

          <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <label className="field" style={{ margin: 0 }}>
              <div className="label">adapter</div>
              <select value={discoveryAdapter} onChange={(e) => setDiscoveryAdapter(e.target.value as any)}>
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
          </div>
          <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
            <button className="btn" disabled={!tauriReady || discoveryRunning} onClick={startDiscovery}>
              run discovery
            </button>
            <button className="btn btn-ghost" disabled={!tauriReady || !discoveryRunning} onClick={stopDiscovery}>
              stop discovery
            </button>
          </div>
          {discoveryError ? (
            <div className="card" style={{ marginTop: 12, borderColor: "#ff8b5a" }}>
              <div className="label">discovery error</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{discoveryError}</div>
            </div>
          ) : null}
          {discoveryStatus ? (
            <div className="kv" style={{ marginTop: 12 }}>
              <div className="k">phase 0</div>
              <div className="v">{discoveryStatus.phase_0_complete ? "✓" : "—"}</div>
              <div className="k">phase 1</div>
              <div className="v">{discoveryStatus.phase_1_complete ? "✓" : "—"}</div>
              <div className="k">phase 2</div>
              <div className="v">{discoveryStatus.phase_2_complete ? "✓" : "—"}</div>
              <div className="k">phase 3</div>
              <div className="v">{discoveryStatus.phase_3_complete ? "✓" : "—"}</div>
            </div>
          ) : null}
          <div className="muted" style={{ marginTop: 10, wordBreak: "break-all" }}>
            {discoveryStatus?.cache_dir ?? "Cache path: cache/discovery/<envId>"}
          </div>
        </section>

        <section className="card flex-card lobby-bench">
          <div className="row-between" style={{ alignItems: "center" }}>
            <h3>Synthetic Eye Bench</h3>
            <button className="btn btn-ghost" disabled={!tauriReady || benchBusy} onClick={runBench}>
              {benchBusy ? "running…" : "run 3s"}
            </button>
          </div>
          {!benchResult ? (
            <div className="muted" style={{ marginTop: 10 }}>
              Runs the zero-copy ingest benchmark (DMA-BUF + fence → CUDA import) and reports FPS.
            </div>
          ) : (
            <div className="kpis kpis-wrap" style={{ marginTop: 10 }}>
              <div className="kpi">
                <div className="label">fps</div>
                <div className="value">{benchResult.fps ? benchResult.fps.toFixed(1) : "—"}</div>
              </div>
              <div className="kpi">
                <div className="label">p50</div>
                <div className="value">{benchResult.per_frame_ms?.p50 != null ? `${benchResult.per_frame_ms.p50.toFixed(2)}ms` : "—"}</div>
              </div>
              <div className="kpi">
                <div className="label">p95</div>
                <div className="value">{benchResult.per_frame_ms?.p95 != null ? `${benchResult.per_frame_ms.p95.toFixed(2)}ms` : "—"}</div>
              </div>
              <div className="kpi">
                <div className="label">mode</div>
                <div className="value">{benchResult.mode ?? "—"}</div>
              </div>
            </div>
          )}
        </section>

        <section className="card flex-card lobby-logs">
          <div className="row-between" style={{ alignItems: "center" }}>
            <h3>Process Logs</h3>
            <div className="row" style={{ gap: 8 }}>
              <button className="btn btn-ghost btn-compact" onClick={clearOmegaLogs}>
                clear omega
              </button>
              <button className="btn btn-ghost btn-compact" onClick={clearDiscoveryLogs}>
                clear discovery
              </button>
            </div>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12, marginTop: 12 }}>
            <div>
              <div className="label">omega</div>
              <pre style={{ marginTop: 8, maxHeight: 260, overflow: "auto" }}>
                {omegaLines
                  .map((l) => `${new Date(l.ts).toLocaleTimeString()} ${l.stream === "stderr" ? "!" : " "} ${l.line}`)
                  .join("\n")}
              </pre>
            </div>
            <div>
              <div className="label">discovery</div>
              <pre style={{ marginTop: 8, maxHeight: 260, overflow: "auto" }}>
                {discoveryLines
                  .map((l) => `${new Date(l.ts).toLocaleTimeString()} ${l.stream === "stderr" ? "!" : " "} ${l.line}`)
                  .join("\n")}
              </pre>
            </div>
          </div>
          {lastError ? (
            <div className="card" style={{ marginTop: 12, borderColor: "#ff8b5a" }}>
              <div className="label">tauri</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{lastError}</div>
            </div>
          ) : null}
        </section>
      </div>

      <div className="lobby-grid">
        <section className="card flex-card">
          <h3>Worker Health</h3>
          <div className="muted">{workers.length ? "Live worker status" : "No workers detected yet"}</div>
          <div className="lobby-worker-list">
            {workers.length === 0 ? (
              <div className="muted">Start Omega to populate worker metrics.</div>
            ) : (
              workers.slice(0, 8).map((w) => (
                <div key={w.instance_id ?? w.display_name ?? w.policy_name} className="lobby-worker">
                  <div>
                    <strong>{w.display_name ?? w.instance_id ?? "Worker"}</strong>
                    <span className="muted">{w.policy_name ?? "policy"}</span>
                  </div>
                  <div className={`badge ${w.status === "running" ? "" : "warn"}`}>{w.status ?? "idle"}</div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="card flex-card">
          <h3>System Telemetry</h3>
          <div className="kpis kpis-wrap">
            <div className="kpi">
              <div className="label">API p95</div>
              <div className="value">{health ? `${fmtFixed(health.api.p95_ms, 1)}ms` : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Error rate</div>
              <div className="value">{health ? fmtPct01(health.api.error_rate, 1) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Heartbeat</div>
              <div className="value">{health ? fmtFixed(health.heartbeat.rate, 2) + "/s" : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Late nodes</div>
              <div className="value">{health ? fmtNum(health.heartbeat.late) : "—"}</div>
            </div>
          </div>
          <div className="muted" style={{ marginTop: 8 }}>
            TTL {health ? fmtFixed(health.heartbeat.ttl_s, 1) : "—"}s
          </div>
        </section>

        <section className="card flex-card">
          <h3>Stream Integrity</h3>
          <div className="kpis kpis-wrap">
            <div className="kpi">
              <div className="label">OK</div>
              <div className="value">{health ? fmtNum(health.stream.ok) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Stale</div>
              <div className="value">{health ? fmtNum(health.stream.stale) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Missing</div>
              <div className="value">{health ? fmtNum(health.stream.missing) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Quality</div>
              <div className="value">{streamOkPct == null ? "—" : `${streamOkPct}%`}</div>
            </div>
          </div>
          <div className="muted" style={{ marginTop: 8 }}>
            p95 frame age {health?.stream.p95_frame_age_s == null ? "—" : `${fmtFixed(health.stream.p95_frame_age_s, 1)}s`}
          </div>
        </section>

        <section className="card flex-card">
          <h3>Active Alerts</h3>
          {issues.length === 0 ? (
            <div className="muted">No incidents in the last 10 minutes.</div>
          ) : (
            <div className="events">
              {issues.slice(0, 6).map((issue) => (
                <div key={issue.id} className="event">
                  <span className="badge">{issue.severity}</span>
                  <div>
                    <strong>{issue.label}</strong>
                    <div className="muted">{issue.hint ?? "Anomaly detected"}</div>
                  </div>
                  <span className="muted" style={{ marginLeft: "auto" }}>
                    {issue.last_seen ? timeAgo(issue.last_seen) : "recent"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
