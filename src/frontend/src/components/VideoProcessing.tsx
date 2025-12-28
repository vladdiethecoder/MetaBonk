import { useEffect, useMemo, useState } from "react";
import { isTauri } from "../lib/tauri";
import useTauriEvent from "../hooks/useTauriEvent";
import useTrainingProgress from "../hooks/useTrainingProgress";
import { processVideos, videoProcessingRunning } from "../services/tauriApi";

type ExitPayload = {
  ok: boolean;
  run_id?: string;
  payload?: any;
  error?: string;
};

type StartPayload = {
  run_id?: string;
  video_dir?: string;
  npz_dir?: string;
  labeled_dir?: string;
  pt_dir?: string;
};

type LogLine = { ts: number; kind: "extract" | "pretrain"; stream: "stdout" | "stderr"; line: string };

export function VideoProcessing() {
  const [videoDir, setVideoDir] = useState("");
  const [fps, setFps] = useState(45);
  const [resizeW, setResizeW] = useState(224);
  const [resizeH, setResizeH] = useState(224);
  const [busy, setBusy] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [details, setDetails] = useState<StartPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const progress = useTrainingProgress();

  useEffect(() => {
    if (!isTauri()) return;
    let cancelled = false;
    (async () => {
      try {
        const running = await videoProcessingRunning();
        if (!cancelled) setBusy(Boolean(running));
      } catch {
        // ignore
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useTauriEvent<StartPayload>("video-processing-start", (payload) => {
    setBusy(true);
    setError(null);
    setDetails(payload);
    if (payload?.run_id) setRunId(payload.run_id);
  });

  useTauriEvent<ExitPayload>("video-processing-exit", (payload) => {
    setBusy(false);
    if (!payload?.ok) {
      setError(payload?.error ?? "video processing failed");
    }
    if (payload?.run_id) setRunId(payload.run_id);
  });

  const pushLog = (kind: LogLine["kind"], stream: LogLine["stream"], line: string) => {
    setLogs((prev) => [...prev.slice(-240), { ts: Date.now(), kind, stream, line }]);
  };
  useTauriEvent<string>("video-processing-stdout", (line) => pushLog("extract", "stdout", line));
  useTauriEvent<string>("video-processing-stderr", (line) => pushLog("extract", "stderr", line));
  useTauriEvent<string>("video-pretrain-stdout", (line) => pushLog("pretrain", "stdout", line));
  useTauriEvent<string>("video-pretrain-stderr", (line) => pushLog("pretrain", "stderr", line));

  const selectDirectory = async () => {
    if (!isTauri()) return;
    setError(null);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ directory: true, multiple: false });
      if (typeof selected === "string") setVideoDir(selected);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const start = async () => {
    if (!isTauri() || busy) return;
    setError(null);
    setLogs([]);
    setDetails(null);
    setRunId(null);
    try {
      setBusy(true);
      const id = await processVideos({ video_dir: videoDir, fps, resize: [resizeW, resizeH] });
      setRunId(id);
    } catch (e: any) {
      setBusy(false);
      setError(String(e?.message ?? e));
    }
  };

  const last = useMemo(() => logs.slice(-18), [logs]);

  if (!isTauri()) {
    return <div className="muted">Video processing is available in the desktop app.</div>;
  }

  return (
    <div className="row" style={{ flexDirection: "column", gap: 10 }}>
      <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
        <label style={{ minWidth: 320 }}>
          <span>video dir</span>
          <input value={videoDir} onChange={(e) => setVideoDir(e.target.value)} placeholder="gameplay_videos" />
        </label>
        <button className="btn btn-ghost" onClick={selectDirectory} disabled={busy}>
          Browse…
        </button>
      </div>

      <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
        <label>
          <span>fps</span>
          <input type="number" min={1} value={fps} onChange={(e) => setFps(Math.max(1, Number(e.target.value || 1)))} />
        </label>
        <label>
          <span>resize w</span>
          <input
            type="number"
            min={1}
            value={resizeW}
            onChange={(e) => setResizeW(Math.max(1, Number(e.target.value || 1)))}
          />
        </label>
        <label>
          <span>resize h</span>
          <input
            type="number"
            min={1}
            value={resizeH}
            onChange={(e) => setResizeH(Math.max(1, Number(e.target.value || 1)))}
          />
        </label>
        <button className="btn" onClick={start} disabled={!videoDir || busy}>
          {busy ? "Processing…" : "Process Videos"}
        </button>
      </div>

      {runId ? (
        <div className="muted">
          run: <span className="mono">{runId}</span>
        </div>
      ) : null}

      {details?.npz_dir ? (
        <div className="muted">
          npz: <span className="mono">{details.npz_dir}</span>
        </div>
      ) : null}
      {details?.pt_dir ? (
        <div className="muted">
          pt: <span className="mono">{details.pt_dir}</span>
        </div>
      ) : null}

      {progress ? (
        <div className="muted">
          pretrain: epoch {progress.epoch} · loss {progress.loss.toFixed(4)}
        </div>
      ) : null}

      {error ? <div className="error">{error}</div> : null}

      {last.length ? (
        <div className="events" style={{ maxHeight: 240 }}>
          {last.map((l, idx) => (
            <div key={`${l.ts}-${idx}`} className="event">
              <span className="badge">
                {l.kind}
                {l.stream === "stderr" ? "-err" : ""}
              </span>
              <div className="mono" style={{ whiteSpace: "pre-wrap" }}>
                {l.line}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

