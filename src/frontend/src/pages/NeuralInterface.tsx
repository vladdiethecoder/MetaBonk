import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { fetchStatus, fetchWorkers } from "../api";
import { fmtFixed } from "../lib/format";
import Go2rtcWebRTC from "../components/Go2rtcWebRTC";
import Reasoning from "./Reasoning";

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
  const focused = workers.find((w) => w.status === "running") ?? workers[0] ?? null;
  const instanceId = focused?.instance_id ?? null;
  const go2rtcBase = String((focused as any)?.go2rtc_base_url ?? "").trim() || fallbackGo2rtcBase();
  const go2rtcName = String((focused as any)?.go2rtc_stream_name ?? "").trim() || "metabonk";
  const embedUrl = go2rtcBase ? `${go2rtcBase.replace(/\/+$/, "")}/stream.html?src=${encodeURIComponent(go2rtcName)}` : "";

  const [showThoughts, setShowThoughts] = useState(true);
  const [showHUD, setShowHUD] = useState(true);

  const liveLabel = useMemo(() => {
    if (statusQ.isError) return "OFFLINE";
    if (!focused) return "IDLE";
    return "LIVE";
  }, [statusQ.isError, focused]);

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
                <input type="checkbox" disabled />
                Attention map (coming soon)
              </label>
              <label>
                <input type="checkbox" disabled />
                Predicted trajectory (coming soon)
              </label>
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
