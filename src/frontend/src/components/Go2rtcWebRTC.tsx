import { useEffect, useRef, useState } from "react";

type HudSample = {
  t: number;
  fps?: number | null;
  p95?: number | null;
  p99?: number | null;
  stalls100?: number | null;
  dropped?: number | null;
  total?: number | null;
  avgJitterMs?: number | null;
  framesDecoded?: number | null;
  framesDropped?: number | null;
  freezeCount?: number | null;
  packetsLost?: number | null;
};

function waitIceComplete(pc: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    let done = false;
    const onState = () => {
      if (pc.iceGatheringState === "complete" && !done) {
        done = true;
        pc.removeEventListener("icegatheringstatechange", onState);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", onState);
    window.setTimeout(() => {
      if (!done) {
        done = true;
        pc.removeEventListener("icegatheringstatechange", onState);
        resolve();
      }
    }, timeoutMs);
  });
}

function percentile(sortedVals: number[], pct: number): number | null {
  if (!sortedVals.length) return null;
  if (pct <= 0) return sortedVals[0];
  if (pct >= 100) return sortedVals[sortedVals.length - 1];
  const k = (pct / 100) * (sortedVals.length - 1);
  const lo = Math.floor(k);
  const hi = Math.ceil(k);
  if (lo === hi) return sortedVals[lo];
  const frac = k - lo;
  return sortedVals[lo] * (1 - frac) + sortedVals[hi] * frac;
}

export default function Go2rtcWebRTC({
  baseUrl,
  streamName,
  className,
  onVideoReady,
  debugHud = false,
  embedUrl,
}: {
  baseUrl: string;
  streamName: string;
  className?: string;
  onVideoReady?: (el: HTMLVideoElement | null) => void;
  debugHud?: boolean;
  embedUrl?: string;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const retryRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);
  const gapRef = useRef<number[]>([]);
  const lastFrameTsRef = useRef<number | null>(null);
  const hudTimerRef = useRef<number | null>(null);
  const rvfcIdRef = useRef<number | null>(null);
  const samplesRef = useRef<HudSample[]>([]);

  const [status, setStatus] = useState<"loading" | "playing" | "error">("loading");
  const [hud, setHud] = useState<{
    fps: number | null;
    p95: number | null;
    p99: number | null;
    stalls100: number | null;
    dropped: number | null;
    total: number | null;
    avgJitterMs: number | null;
    freezeCount: number | null;
  } | null>(null);
  const showEmbed = Boolean(embedUrl && status === "error");

  useEffect(() => {
    if (!onVideoReady) return;
    onVideoReady(videoRef.current);
    return () => onVideoReady(null);
  }, [onVideoReady]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    el.muted = true;
    el.playsInline = true;
    el.autoplay = true;
  }, []);

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    stoppedRef.current = false;

    const cleanupPc = () => {
      const pc = pcRef.current;
      pcRef.current = null;
      if (pc) {
        try {
          pc.ontrack = null;
          pc.onconnectionstatechange = null;
          pc.oniceconnectionstatechange = null;
        } catch {}
        try {
          pc.close();
        } catch {}
      }
      if (el.srcObject) {
        try {
          const ms = el.srcObject as MediaStream;
          ms.getTracks().forEach((t) => t.stop());
        } catch {}
      }
      el.srcObject = null;
    };

    const scheduleRetry = () => {
      if (retryRef.current != null || stoppedRef.current) return;
      retryRef.current = window.setTimeout(() => {
        retryRef.current = null;
        if (!stoppedRef.current) connect();
      }, 1500);
    };

    const connect = async () => {
      cleanupPc();
      setStatus("loading");
      const pc = new RTCPeerConnection();
      pcRef.current = pc;
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.ontrack = (ev) => {
        if (ev.streams && ev.streams[0]) {
          if (el.srcObject !== ev.streams[0]) {
            el.srcObject = ev.streams[0];
          }
          el.play().catch(() => {});
        }
      };
      pc.onconnectionstatechange = () => {
        const st = pc.connectionState;
        if (st === "connected") setStatus("playing");
        if (st === "failed" || st === "disconnected") {
          setStatus("error");
          scheduleRetry();
        }
      };
      pc.oniceconnectionstatechange = () => {
        const st = pc.iceConnectionState;
        if (st === "failed" || st === "disconnected") {
          setStatus("error");
          scheduleRetry();
        }
      };

      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitIceComplete(pc, 2000);
        const localSdp = pc.localDescription?.sdp || offer.sdp || "";
        const apiBase = baseUrl.replace(/\/+$/, "");
        const url = `${apiBase}/api/webrtc?src=${encodeURIComponent(streamName)}`;
        const resp = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/sdp",
            Accept: "application/sdp, application/json",
          },
          body: localSdp,
        });
        if (!resp.ok) throw new Error(`webrtc offer failed (${resp.status})`);
        const ct = resp.headers.get("content-type") || "";
        let answerSdp = "";
        if (ct.includes("application/json")) {
          const data = await resp.json();
          answerSdp = data?.sdp || "";
        } else {
          answerSdp = await resp.text();
        }
        if (!answerSdp) throw new Error("empty webrtc answer");
        await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
        setStatus("playing");
      } catch (e) {
        setStatus("error");
        scheduleRetry();
      }
    };

    connect();

    return () => {
      stoppedRef.current = true;
      if (retryRef.current) {
        window.clearTimeout(retryRef.current);
        retryRef.current = null;
      }
      cleanupPc();
    };
  }, [baseUrl, streamName]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || !debugHud) {
      setHud(null);
      gapRef.current = [];
      lastFrameTsRef.current = null;
      if (rvfcIdRef.current != null) {
        try {
          (el as any).cancelVideoFrameCallback?.(rvfcIdRef.current);
        } catch {}
        rvfcIdRef.current = null;
      }
      if (hudTimerRef.current) {
        window.clearInterval(hudTimerRef.current);
        hudTimerRef.current = null;
      }
      return;
    }

    const hasRvfc = typeof (el as any).requestVideoFrameCallback === "function";
    if (!hasRvfc) {
      setHud({ fps: null, p95: null, p99: null, stalls100: null, dropped: null, total: null, avgJitterMs: null, freezeCount: null });
      return;
    }

    gapRef.current = [];
    lastFrameTsRef.current = null;

    const onFrame = (now: number) => {
      const last = lastFrameTsRef.current;
      if (last != null) {
        const gap = now - last;
        if (Number.isFinite(gap) && gap > 0) {
          const gaps = gapRef.current;
          gaps.push(gap);
          if (gaps.length > 360) gaps.splice(0, gaps.length - 360);
        }
      }
      lastFrameTsRef.current = now;
      rvfcIdRef.current = (el as any).requestVideoFrameCallback(onFrame);
    };

    rvfcIdRef.current = (el as any).requestVideoFrameCallback(onFrame);

    const computeHud = async () => {
      const gaps = gapRef.current.slice().sort((a, b) => a - b);
      const avg = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
      const fps = avg > 0 ? 1000 / avg : null;
      const p95 = percentile(gaps, 95);
      const p99 = percentile(gaps, 99);
      const stalls100 = gaps.length ? gaps.filter((g) => g > 100).length : null;
      let dropped: number | null = null;
      let total: number | null = null;
      try {
        const q = (el as any).getVideoPlaybackQuality?.();
        if (q) {
          dropped = Number.isFinite(q.droppedVideoFrames) ? q.droppedVideoFrames : null;
          total = Number.isFinite(q.totalVideoFrames) ? q.totalVideoFrames : null;
        }
      } catch {}

      let avgJitterMs: number | null = null;
      let freezeCount: number | null = null;
      let framesDecoded: number | null = null;
      let framesDropped: number | null = null;
      let packetsLost: number | null = null;
      try {
        const pc = pcRef.current;
        if (pc) {
          const stats = await pc.getStats();
          stats.forEach((r) => {
            if (r.type === "inbound-rtp" && (r as any).kind === "video") {
              const jbDelay = (r as any).jitterBufferDelay;
              const jbCount = (r as any).jitterBufferEmittedCount;
              if (typeof jbDelay === "number" && typeof jbCount === "number" && jbCount > 0) {
                avgJitterMs = (jbDelay / jbCount) * 1000.0;
              }
              freezeCount = typeof (r as any).freezeCount === "number" ? (r as any).freezeCount : freezeCount;
              framesDecoded = typeof (r as any).framesDecoded === "number" ? (r as any).framesDecoded : framesDecoded;
              framesDropped = typeof (r as any).framesDropped === "number" ? (r as any).framesDropped : framesDropped;
              packetsLost = typeof (r as any).packetsLost === "number" ? (r as any).packetsLost : packetsLost;
            }
          });
        }
      } catch {}

      setHud({ fps, p95, p99, stalls100, dropped, total, avgJitterMs, freezeCount });

      const samples = samplesRef.current;
      samples.push({
        t: Date.now(),
        fps,
        p95,
        p99,
        stalls100,
        dropped,
        total,
        avgJitterMs,
        framesDecoded,
        framesDropped,
        freezeCount,
        packetsLost,
      });
      if (samples.length > 1800) samples.splice(0, samples.length - 1800);
    };

    hudTimerRef.current = window.setInterval(computeHud, 1000);
    return () => {
      if (rvfcIdRef.current != null) {
        try {
          (el as any).cancelVideoFrameCallback?.(rvfcIdRef.current);
        } catch {}
        rvfcIdRef.current = null;
      }
      if (hudTimerRef.current) {
        window.clearInterval(hudTimerRef.current);
        hudTimerRef.current = null;
      }
    };
  }, [debugHud]);

  const exportJson = () => {
    const payload = {
      stream: streamName,
      source: "webrtc",
      collectedAt: new Date().toISOString(),
      samples: samplesRef.current,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `stream_jank_${streamName}_${Date.now()}.json`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 3000);
  };

  return (
    <div className="mse-wrap">
      <video ref={videoRef} className={className} muted playsInline autoPlay />
      {showEmbed ? (
        <iframe
          className="mse-embed"
          src={embedUrl}
          title={`go2rtc-${streamName}`}
          allow="autoplay; fullscreen"
        />
      ) : null}
      {status !== "playing" && !showEmbed ? (
        <div className="mse-overlay">
          <div className="mse-overlay-inner">
            <div className="mse-overlay-title">{status === "error" ? "SIGNAL LOST" : "CONNECTING"}</div>
            <div className="mse-overlay-sub muted">{status === "error" ? "retrying…" : "negotiating…"}</div>
          </div>
        </div>
      ) : null}
      {debugHud ? (
        <div className="mse-hud">
          <div className="mse-hud-title">JANK HUD</div>
          {hud?.fps != null ? <div className="mse-hud-row">fps: {hud.fps.toFixed(1)}</div> : <div className="mse-hud-row">fps: n/a</div>}
          <div className="mse-hud-row">
            p95/p99: {hud?.p95 != null ? hud.p95.toFixed(1) : "n/a"} / {hud?.p99 != null ? hud.p99.toFixed(1) : "n/a"} ms
          </div>
          <div className="mse-hud-row">stalls&gt;100ms: {hud?.stalls100 ?? "n/a"}</div>
          <div className="mse-hud-row">dropped: {hud?.dropped ?? "n/a"} / {hud?.total ?? "n/a"}</div>
          <div className="mse-hud-row">jitter avg: {hud?.avgJitterMs != null ? hud.avgJitterMs.toFixed(1) : "n/a"} ms</div>
          <div className="mse-hud-row">freeze: {hud?.freezeCount ?? "n/a"}</div>
          <button className="mse-hud-btn" onClick={(e) => { e.preventDefault(); e.stopPropagation(); exportJson(); }}>
            Export JSON
          </button>
        </div>
      ) : null}
    </div>
  );
}
