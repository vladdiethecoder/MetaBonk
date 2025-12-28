import { Link } from "react-router-dom";
import { useMemo, useState } from "react";
import useTauriEvent from "../hooks/useTauriEvent";
import { fmtFixed } from "../lib/format";
import { useTauriRuntime } from "../tauri/RuntimeProvider";
import { TrainingPanel } from "../components/TrainingPanel";
import { VideoProcessing } from "../components/VideoProcessing";

type TelemetryPayload = {
  ts?: number;
  cpu?: { usage_pct?: number; cores?: number };
  memory?: { used_mb?: number; total_mb?: number };
  gpu?: { util_pct?: number; mem_used_mb?: number; mem_total_mb?: number; temp_c?: number } | null;
};
type LogLine = { ts: number; source: "omega" | "discovery"; stream: "stdout" | "stderr"; line: string };

export default function Laboratory() {
  const { tauriReady, omegaLogs, discoveryLogs } = useTauriRuntime();
  const [telemetry, setTelemetry] = useState<TelemetryPayload | null>(null);
  const [showOmega, setShowOmega] = useState(true);
  const [showDiscovery, setShowDiscovery] = useState(true);
  const [showStderr, setShowStderr] = useState(true);

  useTauriEvent<TelemetryPayload>("system-telemetry", (payload) => {
    setTelemetry(payload);
  });

  const logs = useMemo(() => {
    const merged: LogLine[] = [];
    omegaLogs.forEach((l) => merged.push({ ts: l.ts, stream: l.stream, line: l.line, source: "omega" }));
    discoveryLogs.forEach((l) => merged.push({ ts: l.ts, stream: l.stream, line: l.line, source: "discovery" }));
    merged.sort((a, b) => a.ts - b.ts);
    return merged.slice(-600);
  }, [omegaLogs, discoveryLogs]);

  const filteredLogs = useMemo(() => {
    return logs.filter((l) => {
      const isErr = l.stream === "stderr";
      if (isErr && !showStderr) return false;
      if (l.source === "omega" && !showOmega) return false;
      if (l.source === "discovery" && !showDiscovery) return false;
      return true;
    });
  }, [logs, showOmega, showDiscovery, showStderr]);

  const cpu = telemetry?.cpu?.usage_pct;
  const memUsed = telemetry?.memory?.used_mb;
  const memTotal = telemetry?.memory?.total_mb;
  const memPct = memUsed && memTotal ? (memUsed / memTotal) * 100 : null;
  const gpu = telemetry?.gpu ?? null;

  return (
    <div className="page lab-page">
      <section className="card lab-hero">
        <h1>Laboratory</h1>
        <p className="muted">Experimentation surface: metrics, logs, runs, instances, and discovery artifacts.</p>
      </section>

      {tauriReady ? (
        <div className="lab-grid">
          <section className="card">
            <div className="card-header">
              <div>
                <h3>System Telemetry</h3>
                <p className="muted">CPU/GPU utilization and memory pressure.</p>
              </div>
            </div>
            <div className="card-body">
              <div className="kpis kpis-wrap">
                <div className="kpi">
                  <div className="label">CPU</div>
                  <div className="value">{cpu == null ? "—" : `${fmtFixed(cpu, 1)}%`}</div>
                </div>
                <div className="kpi">
                  <div className="label">RAM</div>
                  <div className="value">{memPct == null ? "—" : `${fmtFixed(memPct, 1)}%`}</div>
                </div>
                <div className="kpi">
                  <div className="label">GPU</div>
                  <div className="value">{gpu?.util_pct == null ? "—" : `${fmtFixed(gpu.util_pct, 1)}%`}</div>
                </div>
                <div className="kpi">
                  <div className="label">VRAM</div>
                  <div className="value">
                    {gpu?.mem_used_mb == null || gpu?.mem_total_mb == null
                      ? "—"
                      : `${fmtFixed(gpu.mem_used_mb, 0)}/${fmtFixed(gpu.mem_total_mb, 0)} MB`}
                  </div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 10 }}>
                {gpu?.temp_c == null ? "GPU telemetry unavailable" : `GPU temp ${fmtFixed(gpu.temp_c, 0)}°C`}
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>Backend Logs</h3>
                <p className="muted">Live stdout/stderr from Omega + Discovery.</p>
              </div>
              <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                <label className="toggle">
                  <input type="checkbox" checked={showOmega} onChange={(e) => setShowOmega(e.target.checked)} />
                  <span>omega</span>
                </label>
                <label className="toggle">
                  <input type="checkbox" checked={showDiscovery} onChange={(e) => setShowDiscovery(e.target.checked)} />
                  <span>discovery</span>
                </label>
                <label className="toggle">
                  <input type="checkbox" checked={showStderr} onChange={(e) => setShowStderr(e.target.checked)} />
                  <span>stderr</span>
                </label>
              </div>
            </div>
            <div className="card-body">
              <div className="events" style={{ maxHeight: 320 }}>
                {filteredLogs.length === 0 ? (
                  <div className="muted">No logs yet.</div>
                ) : (
                  filteredLogs.slice(-200).map((l, idx) => (
                    <div key={`${l.ts}-${idx}`} className="event">
                      <span className="badge">
                        {l.source}
                        {l.stream === "stderr" ? "-err" : ""}
                      </span>
                      <div className="mono" style={{ whiteSpace: "pre-wrap" }}>
                        {l.line}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>Video Processing</h3>
                <p className="muted">GUI wrapper for `scripts/video_to_trajectory.py` + `scripts/video_pretrain.py`.</p>
              </div>
            </div>
            <div className="card-body">
              <VideoProcessing />
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>Training Panel</h3>
                <p className="muted">Start a training run and persist the requested config.</p>
              </div>
            </div>
            <div className="card-body">
              <TrainingPanel />
            </div>
          </section>
        </div>
      ) : null}

      <div className="lab-grid">
        <section className="card">
          <h3>Runs</h3>
          <p className="muted">Training curves, artifacts, and comparisons.</p>
          <Link to="/lab/runs" className="btn btn-ghost">
            Open Runs
          </Link>
        </section>
        <section className="card">
          <h3>Instances</h3>
          <p className="muted">Live workers, telemetry, and stream integrity.</p>
          <Link to="/lab/instances" className="btn btn-ghost">
            Open Instances
          </Link>
        </section>
        <section className="card">
          <h3>Discovery Artifacts</h3>
          <p className="muted">Inspect learned action space and PPO config.</p>
          <Link to="/lab/discovery" className="btn btn-ghost">
            Open Discovery Inspector
          </Link>
        </section>
        <section className="card">
          <h3>Build Lab</h3>
          <p className="muted">Architecture experiments and hyperparameter probes.</p>
          <Link to="/lab/build" className="btn btn-ghost">
            Open Build Lab
          </Link>
        </section>
        <section className="card">
          <h3>Codex</h3>
          <p className="muted">Learned skills and neural atlas.</p>
          <Link to="/codex" className="btn btn-ghost">
            Open Codex
          </Link>
        </section>
      </div>
    </div>
  );
}
