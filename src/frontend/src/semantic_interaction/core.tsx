import { useEffect, useMemo, useState } from "react";
import { isTauri } from "../lib/tauri";
import { tauriInvoke } from "../lib/tauri_api";

export type SemanticIntent =
  | { kind: "start_omega"; mode: "train" | "play" | "dream"; workers: number; env_id: string }
  | { kind: "stop_omega" }
  | { kind: "start_discovery"; env_id: string; env_adapter?: "mock" | "synthetic-eye" }
  | { kind: "stop_discovery" }
  | { kind: "run_synthetic_eye_bench"; id?: string }
  | { kind: "unknown"; reason: string };

type SemanticCoreProps = {
  envId: string;
  defaultWorkers?: number;
  onIntent?: (intent: SemanticIntent) => void;
};

const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.floor(v)));

function parseSemantic(input: string, envId: string, defaultWorkers: number): SemanticIntent {
  const raw = String(input ?? "").trim();
  const s = raw.toLowerCase();
  if (!s) return { kind: "unknown", reason: "empty input" };

  const workersMatch = s.match(/workers?\\s+(\\d{1,2})/);
  const workers = workersMatch ? clampInt(Number(workersMatch[1]), 1, 64) : defaultWorkers;
  const envMatch = raw.match(/env\\s*[:=]?\\s*([A-Za-z0-9 _:-]{2,64})/i);
  const env_id = (envMatch ? String(envMatch[1]) : envId).trim() || envId;

  const has = (...needles: string[]) => needles.some((n) => s.includes(n));
  const mode: "train" | "play" | "dream" = has("dream") ? "dream" : has("play") ? "play" : "train";

  if (has("stop omega", "halt omega", "kill omega")) return { kind: "stop_omega" };
  if (has("start omega", "launch omega", "run omega", "omega live")) return { kind: "start_omega", mode, workers, env_id };

  if (has("stop discovery", "halt discovery", "kill discovery")) return { kind: "stop_discovery" };
  if (has("start discovery", "run discovery", "discover actions", "discover inputs")) {
    const adapter: "mock" | "synthetic-eye" | undefined = has("synthetic", "eye") ? "synthetic-eye" : undefined;
    return { kind: "start_discovery", env_id, env_adapter: adapter };
  }

  if (has("bench", "benchmark", "synthetic eye bench")) {
    const idMatch = raw.match(/id\\s*[:=]?\\s*([A-Za-z0-9 _:-]{2,64})/i);
    return { kind: "run_synthetic_eye_bench", id: idMatch ? String(idMatch[1]).trim() : undefined };
  }

  return { kind: "unknown", reason: "no intent matched (try: start omega / stop omega / start discovery / bench)" };
}

async function runIntent(intent: SemanticIntent): Promise<string> {
  if (!isTauri()) return "Tauri not available (open the desktop app to execute intents).";
  if (intent.kind === "start_omega") {
    await tauriInvoke("start_omega", {
      mode: intent.mode,
      workers: intent.workers,
      env_id: intent.env_id,
      synthetic_eye: true,
    });
    return `invoked start_omega(mode=${intent.mode}, workers=${intent.workers}, env_id=${intent.env_id})`;
  }
  if (intent.kind === "stop_omega") {
    await tauriInvoke("stop_omega");
    return "invoked stop_omega()";
  }
  if (intent.kind === "start_discovery") {
    await tauriInvoke("start_discovery", {
      env_id: intent.env_id,
      env_adapter: intent.env_adapter ?? null,
    });
    return `invoked start_discovery(env_id=${intent.env_id})`;
  }
  if (intent.kind === "stop_discovery") {
    await tauriInvoke("stop_discovery");
    return "invoked stop_discovery()";
  }
  if (intent.kind === "run_synthetic_eye_bench") {
    const res = await tauriInvoke<any>("run_synthetic_eye_bench", { id: intent.id ?? null });
    return `bench: ${JSON.stringify(res)}`;
  }
  return `no-op: ${intent.reason}`;
}

export default function SemanticCore({ envId, defaultWorkers = 1, onIntent }: SemanticCoreProps) {
  const [input, setInput] = useState("start omega mode train workers 1");
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<string>("");
  const intent = useMemo(() => parseSemantic(input, envId, defaultWorkers), [defaultWorkers, envId, input]);

  useEffect(() => {
    onIntent?.(intent);
  }, [intent, onIntent]);

  const preview = useMemo(() => {
    if (intent.kind === "unknown") return intent.reason;
    return JSON.stringify(intent, null, 2);
  }, [intent]);

  return (
    <div>
      <div className="card-header">
        <div>
          <h3>Semantic Interaction</h3>
          <p className="muted">Intent-first control surface. Type a command in plain language; the UI infers a callable intent.</p>
        </div>
      </div>
      <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={3}
          spellCheck={false}
          className="mono"
          style={{ width: "100%", resize: "vertical" }}
        />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <button
            className="btn"
            disabled={busy || intent.kind === "unknown"}
            onClick={async () => {
              setBusy(true);
              setLast("");
              try {
                const msg = await runIntent(intent);
                setLast(msg);
              } catch (e: any) {
                setLast(String(e?.message ?? e));
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? "running…" : "execute"}
          </button>
          <button className="btn btn-ghost" onClick={() => setInput("")}>
            clear
          </button>
          <div className="muted" style={{ fontSize: 12 }}>
            env: <span className="mono">{envId}</span>
          </div>
          <div className="muted" style={{ fontSize: 12 }}>
            runtime: <span className="mono">{isTauri() ? "tauri" : "browser"}</span>
          </div>
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          intent preview
        </div>
        <div className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 12, marginTop: -6 }}>
          {preview}
        </div>
        {last ? (
          <>
            <div className="muted" style={{ fontSize: 12 }}>
              last result
            </div>
            <div className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 12, marginTop: -6 }}>
              {last}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

