import { useState } from "react";
import { isTauri } from "../lib/tauri";
import useTrainingProgress from "../hooks/useTrainingProgress";
import { getTrainingStatus, startTraining, type TrainingConfig } from "../services/tauriApi";

export function TrainingPanel() {
  const [cfg, setCfg] = useState<TrainingConfig>({
    model: "omega_protocol",
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
  });
  const [busy, setBusy] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const progress = useTrainingProgress();

  const start = async () => {
    if (!isTauri() || busy) return;
    setBusy(true);
    setError(null);
    setStatus("");
    try {
      const id = await startTraining(cfg);
      setRunId(id);
      setStatus("running");
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const refresh = async () => {
    if (!isTauri() || !runId || busy) return;
    setBusy(true);
    setError(null);
    try {
      const s = await getTrainingStatus(runId);
      setStatus(s);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  if (!isTauri()) {
    return <div className="muted">Training controls are available in the desktop app.</div>;
  }

  return (
    <div className="row" style={{ flexDirection: "column", gap: 10 }}>
      <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
        <label>
          <span>model</span>
          <input value={cfg.model} onChange={(e) => setCfg((p) => ({ ...p, model: e.target.value }))} />
        </label>
        <label>
          <span>epochs</span>
          <input
            type="number"
            min={1}
            value={cfg.epochs}
            onChange={(e) => setCfg((p) => ({ ...p, epochs: Math.max(1, Number(e.target.value || 1)) }))}
          />
        </label>
        <label>
          <span>batch</span>
          <input
            type="number"
            min={1}
            value={cfg.batch_size}
            onChange={(e) => setCfg((p) => ({ ...p, batch_size: Math.max(1, Number(e.target.value || 1)) }))}
          />
        </label>
        <label>
          <span>lr</span>
          <input
            type="number"
            step="0.0001"
            min={0}
            value={cfg.learning_rate}
            onChange={(e) => setCfg((p) => ({ ...p, learning_rate: Number(e.target.value || 0) }))}
          />
        </label>
      </div>

      <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
        <button className="btn" onClick={start} disabled={busy}>
          {busy ? "Starting…" : "Start Training"}
        </button>
        <button className="btn btn-ghost" onClick={refresh} disabled={!runId || busy}>
          Refresh Status
        </button>
      </div>

      {runId ? (
        <div className="muted">
          run: <span className="mono">{runId}</span> · status: {status || "—"}
        </div>
      ) : null}
      {progress ? (
        <div className="muted">
          progress: epoch {progress.epoch} · loss {progress.loss.toFixed(4)} · acc {progress.accuracy.toFixed(3)}
        </div>
      ) : null}
      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}

