import { useEffect, useMemo, useState } from "react";
import PageShell from "../components/PageShell";
import { isTauri } from "../lib/tauri";

type Mode = "train" | "play" | "dream";

type LogLine = {
  ts: number;
  stream: "stdout" | "stderr";
  line: string;
};

export default function Supervisor() {
  const [tauriReady, setTauriReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<Mode>("train");
  const [workers, setWorkers] = useState(1);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);

  const lines = useMemo(() => logs.slice(-400), [logs]);

  useEffect(() => {
    let unlistenOut: null | (() => void) = null;
    let unlistenErr: null | (() => void) = null;
    let unlistenExit: null | (() => void) = null;
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

        const isRunning = await invoke<boolean>("omega_running");
        if (!cancelled) setRunning(Boolean(isRunning));
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
    };
  }, []);

  const start = async () => {
    setError(null);
    if (!tauriReady) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_omega", { mode, workers });
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

  return (
    <PageShell title="Supervisor" subtitle="Tauri-controlled Omega process (dev)">
      <div className="grid cols-2">
        <div className="card">
          <div className="row" style={{ gap: 12, alignItems: "center" }}>
            <div className="pill">{tauriReady ? "tauri" : "browser"}</div>
            <div className="pill">{running ? "running" : "stopped"}</div>
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
        </div>
      </div>
    </PageShell>
  );
}
