import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { fetchStatus, fetchWorkers } from "../api";
import { fmtFixed } from "../lib/format";
import Go2rtcWebRTC from "../components/Go2rtcWebRTC";
import Reasoning from "./Reasoning";
import useTauriEvent from "../hooks/useTauriEvent";

type OverlayEvent = {
  __meta_event?: string;
  kind?: string;
  instance_id?: string;
  ts?: number;
  payload?: any;
  overlay_png?: string;
  image_b64?: string;
  png_base64?: string;
  overlay_url?: string;
};

const fallbackGo2rtcBase = () => {
  if (typeof window === "undefined") return "";
  const host = window.location.hostname || "localhost";
  const protocol = window.location.protocol.startsWith("http") ? window.location.protocol : "http:";
  return `${protocol}//${host}:1984`;
};

export default function NeuralInterface() {
  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 2000 });
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 1000 });
  const workers = Object.values(workersQ.data ?? {});
  const [selectedId, setSelectedId] = useState<string>("");

  const runningWorker = workers.find((w) => w.status === "running") ?? null;
  const selectedWorker = selectedId ? workers.find((w) => String(w.instance_id ?? "") === selectedId) ?? null : null;
  const focused = selectedWorker ?? runningWorker ?? workers[0] ?? null;
  const instanceId = focused?.instance_id ?? null;
  const go2rtcBase = String((focused as any)?.go2rtc_base_url ?? "").trim() || fallbackGo2rtcBase();
  const go2rtcName = String((focused as any)?.go2rtc_stream_name ?? "").trim() || "metabonk";
  const embedUrl = go2rtcBase ? `${go2rtcBase.replace(/\/+$/, "")}/stream.html?src=${encodeURIComponent(go2rtcName)}` : "";

  const [showThoughts, setShowThoughts] = useState(true);
  const [showHUD, setShowHUD] = useState(true);
  const [overlayMode, setOverlayMode] = useState<"attention" | "trajectory" | "prediction" | "none">("none");
  const [overlayEvent, setOverlayEvent] = useState<OverlayEvent | null>(null);

  useTauriEvent<OverlayEvent>("omega-meta", (payload) => {
    const kind = String(payload?.__meta_event ?? payload?.kind ?? "");
    if (!kind) return;
    if (!kind.includes("world") && !kind.includes("overlay") && !kind.includes("trajectory")) return;
    if (payload?.instance_id && instanceId && String(payload.instance_id) !== String(instanceId)) return;
    setOverlayEvent(payload);
  });

  const liveLabel = useMemo(() => {
    if (statusQ.isError) return "OFFLINE";
    if (!focused) return "IDLE";
    return "LIVE";
  }, [statusQ.isError, focused]);

  const overlayImage = useMemo(() => {
    const raw = overlayEvent?.overlay_png || overlayEvent?.png_base64 || overlayEvent?.image_b64 || "";
    if (raw) {
      return raw.startsWith("data:") ? raw : `data:image/png;base64,${raw}`;
    }
    const url = overlayEvent?.overlay_url;
    return url || "";
  }, [overlayEvent]);

  const agentRows = useMemo(() => {
    const list = workers.slice();
    list.sort((a, b) => {
      const ar = a.status === "running" ? 1 : 0;
      const br = b.status === "running" ? 1 : 0;
      if (ar !== br) return br - ar;
      return String(a.display_name ?? a.instance_id ?? "").localeCompare(String(b.display_name ?? b.instance_id ?? ""));
    });
    return list.slice(0, 12);
  }, [workers]);

  return (
    <div className="page neural-page">
      <div className="neural-shell">
        <div className="neural-stream">
          <div className="neural-badge">
            <span className={`dot ${liveLabel === "LIVE" ? "live" : "idle"}`} />
            {liveLabel}
            {focused ? ` | ${focused.display_name ?? focused.instance_id}` : ""}
          </div>
          <Go2rtcWebRTC
            streamName={go2rtcName}
            baseUrl={go2rtcBase}
            embedUrl={embedUrl || undefined}
            className="neural-video"
          />
          {overlayMode !== "none" && overlayImage ? (
            <img className="neural-overlay" src={overlayImage} alt="world-model overlay" />
          ) : null}
          {showHUD ? (
            <div className="neural-hud">
              <div>
                <span>policy</span>
                <strong>{focused?.policy_name ?? "—"}</strong>
              </div>
              <div>
                <span>reward</span>
                <strong>{focused?.reward ?? focused?.steam_score ?? "—"}</strong>
              </div>
              <div>
                <span>obs fps</span>
                <strong>{focused?.obs_fps ? fmtFixed(Number(focused.obs_fps), 1) : "—"}</strong>
              </div>
              <div>
                <span>act hz</span>
                <strong>{focused?.act_hz ? fmtFixed(Number(focused.act_hz), 1) : "—"}</strong>
              </div>
            </div>
          ) : null}
        </div>

        <aside className="neural-side">
          <section className="card">
            <div className="card-header">
              <div>
                <h3>Agents</h3>
                <p className="muted">Select a live instance to focus.</p>
              </div>
            </div>
            <div className="card-body">
              <div className="lobby-worker-list" style={{ marginTop: 0 }}>
                {agentRows.length === 0 ? (
                  <div className="muted">No workers detected yet.</div>
                ) : (
                  agentRows.map((w) => {
                    const id = String(w.instance_id ?? "");
                    const name = String(w.display_name ?? id).trim() || "Worker";
                    const isSelected = Boolean(id) && (id === selectedId || (!selectedId && id === instanceId));
                    return (
                      <button
                        key={id || name}
                        type="button"
                        className={`lobby-worker lobby-worker-btn ${isSelected ? "active" : ""}`.trim()}
                        onClick={() => setSelectedId(id)}
                      >
                        <div>
                          <strong>{name}</strong>
                          <span className="muted">{w.policy_name ?? "policy"}</span>
                        </div>
                        <div className={`badge ${w.status === "running" ? "" : "warn"}`}>{w.status ?? "idle"}</div>
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>Agent Thoughts</h3>
                <p className="muted">Live intent and reasoning trace.</p>
              </div>
              <label className="toggle">
                <input type="checkbox" checked={showThoughts} onChange={(e) => setShowThoughts(e.target.checked)} />
                <span>show</span>
              </label>
            </div>
            <div className="card-body" style={{ minHeight: 240 }}>
              {showThoughts ? (
                <Reasoning embedded defaultInstanceId={instanceId ?? undefined} />
              ) : (
                <div className="muted">Thought stream hidden.</div>
              )}
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>Overlay Controls</h3>
                <p className="muted">Toggle HUD and future overlays.</p>
              </div>
            </div>
            <div className="card-body neural-controls">
              <label>
                <input type="checkbox" checked={showHUD} onChange={(e) => setShowHUD(e.target.checked)} />
                HUD metrics
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={overlayMode === "attention"}
                  onChange={(e) => setOverlayMode(e.target.checked ? "attention" : "none")}
                  disabled={!overlayImage}
                />
                Attention map {overlayImage ? "" : "(waiting)"}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={overlayMode === "trajectory"}
                  onChange={(e) => setOverlayMode(e.target.checked ? "trajectory" : "none")}
                  disabled={!overlayImage}
                />
                Predicted trajectory {overlayImage ? "" : "(waiting)"}
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={overlayMode === "prediction"}
                  onChange={(e) => setOverlayMode(e.target.checked ? "prediction" : "none")}
                  disabled={!overlayImage}
                />
                Imagination preview {overlayImage ? "" : "(waiting)"}
              </label>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>World Model Feed</h3>
                <p className="muted">Latest overlay payload from Omega.</p>
              </div>
            </div>
            <div className="card-body">
              {overlayEvent ? (
                <div className="mono" style={{ whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(
                    {
                      event: overlayEvent.__meta_event ?? overlayEvent.kind,
                      ts: overlayEvent.ts,
                      instance_id: overlayEvent.instance_id,
                      payload: overlayEvent.payload ?? null,
                    },
                    null,
                    2,
                  )}
                </div>
              ) : (
                <div className="muted">No world-model overlay packets yet.</div>
              )}
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h3>Session Health</h3>
                <p className="muted">Surface latency + backend state.</p>
              </div>
            </div>
            <div className="card-body neural-health">
              <div>
                <span>Status</span>
                <strong>{statusQ.isError ? "offline" : "online"}</strong>
              </div>
              <div>
                <span>Workers</span>
                <strong>{workers.length}</strong>
              </div>
              <div>
                <span>Streaming</span>
                <strong>{go2rtcBase ? "go2rtc" : "offline"}</strong>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
