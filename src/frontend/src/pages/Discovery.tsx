import { useEffect, useMemo, useState } from "react";
import useLaunchConfig from "../hooks/useLaunchConfig";
import { tauriInvoke } from "../lib/tauri_api";
import { useTauriRuntime } from "../tauri/RuntimeProvider";

type DiscoveryStatus = {
  env_id: string;
  cache_dir: string;
  phase_0_complete: boolean;
  phase_1_complete: boolean;
  phase_2_complete: boolean;
  phase_3_complete: boolean;
  ready_for_training: boolean;
  artifacts?: Record<string, string>;
  has_ppo_config?: boolean;
};

type ActionSpace = {
  discrete?: Array<Record<string, any>>;
  continuous?: Record<string, any>;
  metadata?: Record<string, any>;
};

export default function Discovery() {
  const { tauriReady, discoveryRunning, discoveryLogs } = useTauriRuntime();
  const [launchCfg, setLaunchCfg] = useLaunchConfig();
  const envId = launchCfg.envId;
  const [status, setStatus] = useState<DiscoveryStatus | null>(null);
  const [adapter, setAdapter] = useState("mock");
  const [budget, setBudget] = useState(2000);
  const [holdFrames, setHoldFrames] = useState(30);
  const [actionSpace, setActionSpace] = useState<ActionSpace | null>(null);
  const [ppoConfig, setPpoConfig] = useState<Record<string, string> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logs = useMemo(() => discoveryLogs.slice(-400), [discoveryLogs]);

  const refreshStatus = async () => {
    try {
      const res = await tauriInvoke<DiscoveryStatus>("discovery_status", { env_id: envId });
      setStatus(res);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  useEffect(() => {
    if (!tauriReady) return;
    refreshStatus();
    const t = window.setInterval(() => {
      refreshStatus();
    }, 4000);
    return () => window.clearInterval(t);
  }, [tauriReady, envId]);

  const startDiscovery = async () => {
    try {
      setError(null);
      await tauriInvoke("start_discovery", {
        env_id: envId,
        env_adapter: adapter,
        exploration_budget: budget,
        hold_frames: holdFrames,
      });
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const stopDiscovery = async () => {
    try {
      setError(null);
      await tauriInvoke("stop_discovery");
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const loadActionSpace = async () => {
    try {
      setError(null);
      const res = await tauriInvoke<ActionSpace>("load_action_space", { env_id: envId });
      setActionSpace(res || null);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const loadPpoConfig = async () => {
    try {
      setError(null);
      const res = await tauriInvoke<Record<string, string>>("load_ppo_config", { env_id: envId });
      setPpoConfig(res || null);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const discrete = actionSpace?.discrete ?? [];
  const continuous = actionSpace?.continuous ?? {};
  const metadata = actionSpace?.metadata ?? {};

  const continuousRows = useMemo(() => Object.entries(continuous), [continuous]);

  if (!tauriReady) {
    return (
      <div className="page discovery-page">
        <section className="card">
          <h1>Discovery</h1>
          <p className="muted">
            Discovery controls require the Tauri app. Launch via <code>./start --tauri</code> (runs{" "}
            <code>npx tauri dev</code>).
          </p>
        </section>
      </div>
    );
  }

  return (
    <div className="page discovery-page">
      <section className="card">
        <h1>Discovery</h1>
        <p className="muted">Autonomous discovery pipeline (phases 0–3) and action-space inspection.</p>
        <div className="row" style={{ gap: 12, flexWrap: "wrap", marginTop: 12 }}>
          <label className="field" style={{ margin: 0 }}>
            <div className="label">env id</div>
            <input
              value={envId}
              onChange={(e) => setLaunchCfg((prev) => ({ ...prev, envId: e.target.value }))}
              placeholder="MegaBonk"
            />
          </label>
          <label className="field" style={{ margin: 0 }}>
            <div className="label">adapter</div>
            <select value={adapter} onChange={(e) => setAdapter(e.target.value)}>
              <option value="mock">mock</option>
              <option value="synthetic-eye">synthetic-eye</option>
            </select>
          </label>
          <label className="field" style={{ margin: 0 }}>
            <div className="label">budget</div>
            <input type="number" min={50} value={budget} onChange={(e) => setBudget(Number(e.target.value || 0))} />
          </label>
          <label className="field" style={{ margin: 0 }}>
            <div className="label">hold frames</div>
            <input
              type="number"
              min={1}
              value={holdFrames}
              onChange={(e) => setHoldFrames(Number(e.target.value || 0))}
            />
          </label>
          <button className="btn" disabled={discoveryRunning} onClick={startDiscovery}>
            run discovery
          </button>
          <button className="btn btn-ghost" disabled={!discoveryRunning} onClick={stopDiscovery}>
            stop discovery
          </button>
        </div>
        {error ? <div className="muted" style={{ marginTop: 10 }}>error: {error}</div> : null}
        {status ? (
          <div className="row" style={{ gap: 16, marginTop: 16, flexWrap: "wrap" }}>
            <div className="kpi">
              <div className="label">phase 0</div>
              <div className="value">{status.phase_0_complete ? "✓" : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">phase 1</div>
              <div className="value">{status.phase_1_complete ? "✓" : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">phase 2</div>
              <div className="value">{status.phase_2_complete ? "✓" : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">phase 3</div>
              <div className="value">{status.phase_3_complete ? "✓" : "—"}</div>
            </div>
          </div>
        ) : null}
      </section>

      <div className="lab-grid" style={{ marginTop: 12 }}>
        <section className="card">
          <div className="card-header">
            <div>
              <h3>Action Space Inspector</h3>
              <p className="muted">Preview learned discrete + continuous actions.</p>
            </div>
            <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
              <button className="btn btn-ghost" onClick={loadActionSpace}>
                load action space
              </button>
              <button className="btn btn-ghost" onClick={loadPpoConfig}>
                load ppo config
              </button>
            </div>
          </div>
          <div className="card-body">
            {actionSpace ? (
              <>
                <div className="muted" style={{ marginBottom: 8 }}>
                  metadata: {Object.keys(metadata).length ? JSON.stringify(metadata) : "—"}
                </div>
                <div className="table-viewport" style={{ maxHeight: 260 }}>
                  <table className="table table-compact">
                    <thead>
                      <tr>
                        <th>input</th>
                        <th>binding</th>
                        <th>label</th>
                        <th>effect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {discrete.length === 0 ? (
                        <tr>
                          <td colSpan={4} className="muted">No discrete actions loaded.</td>
                        </tr>
                      ) : (
                        discrete.map((d, idx) => (
                          <tr key={`${d.input_id ?? idx}-${idx}`}>
                            <td>{d.input_id ?? "—"}</td>
                            <td>{d.binding?.name ?? d.binding?.type ?? "—"}</td>
                            <td>{d.semantic_label ?? d.label ?? "—"}</td>
                            <td className="mono">{d.expected_effect ? JSON.stringify(d.expected_effect) : "—"}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: 12 }}>
                  <div className="label">continuous</div>
                  {continuousRows.length === 0 ? (
                    <div className="muted">No continuous dims found.</div>
                  ) : (
                    <ul className="list" style={{ marginTop: 6 }}>
                      {continuousRows.map(([k, v]) => (
                        <li key={k} className="row" style={{ gap: 8 }}>
                          <strong>{k}</strong>
                          <span className="muted">{JSON.stringify(v)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                {ppoConfig ? (
                  <div style={{ marginTop: 12 }}>
                    <div className="label">ppo config</div>
                    <div className="mono" style={{ whiteSpace: "pre-wrap" }}>
                      {Object.entries(ppoConfig)
                        .map(([k, v]) => `${k}=${v}`)
                        .join("\n")}
                    </div>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="muted">Load action space to inspect learned controls.</div>
            )}
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h3>Discovery Logs</h3>
              <p className="muted">Live output from discovery phases.</p>
            </div>
          </div>
          <div className="card-body">
            <div className="events" style={{ maxHeight: 320 }}>
              {logs.length === 0 ? (
                <div className="muted">No logs yet.</div>
              ) : (
                logs.slice(-200).map((l, idx) => (
                  <div key={`${l.ts}-${idx}`} className="event">
                    <span className={`badge ${l.stream === "stderr" ? "warn" : ""}`}>{l.stream}</span>
                    <div className="mono" style={{ whiteSpace: "pre-wrap" }}>{l.line}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
