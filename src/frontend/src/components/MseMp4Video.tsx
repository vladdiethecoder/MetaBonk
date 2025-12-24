import { useEffect, useRef, useState } from "react";

type Mp4Box = { type: string; data: Uint8Array };

type HudSample = {
  t: number;
  fps?: number | null;
  p95?: number | null;
  p99?: number | null;
  stalls100?: number | null;
  dropped?: number | null;
  total?: number | null;
};

function findAvc1Codec(init: Uint8Array): string | null {
  const needle = [0x61, 0x76, 0x63, 0x43]; // "avcC"
  for (let i = 0; i < init.length - 16; i++) {
    if (init[i] !== needle[0]) continue;
    if (init[i + 1] !== needle[1] || init[i + 2] !== needle[2] || init[i + 3] !== needle[3]) continue;
    const base = i + 4;
    const configurationVersion = init[base];
    if (configurationVersion !== 1) continue;
    const profile = init[base + 1];
    const compat = init[base + 2];
    const level = init[base + 3];
    const hex = (n: number) => n.toString(16).padStart(2, "0").toUpperCase();
    return `avc1.${hex(profile)}${hex(compat)}${hex(level)}`;
  }
  return null;
}

function _u32be(b: Uint8Array, o: number) {
  return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
}

function _ascii4(b: Uint8Array, o: number) {
  return String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
}

function tryParseMp4Box(buf: Uint8Array): { box: Mp4Box; rest: Uint8Array } | null {
  if (buf.length < 8) return null;
  const size32 = _u32be(buf, 0);
  const type = _ascii4(buf, 4);
  let header = 8;
  let size = size32;
  if (size32 === 1) {
    if (buf.length < 16) return null;
    const hi = _u32be(buf, 8);
    const lo = _u32be(buf, 12);
    size = hi * 2 ** 32 + lo;
    header = 16;
  } else if (size32 === 0) {
    return null;
  }
  if (!Number.isFinite(size) || size < header) return null;
  if (buf.length < size) return null;
  const data = buf.slice(0, size);
  const rest = buf.slice(size);
  return { box: { type, data }, rest };
}

function concatParts(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

export default function MseMp4Video({
  url,
  className,
  fallbackUrl,
  exclusiveKey,
  onVideoReady,
  debug = false,
  debugHud = false,
}: {
  url: string;
  className?: string;
  fallbackUrl?: string;
  exclusiveKey?: string;
  onVideoReady?: (el: HTMLVideoElement | null) => void;
  debug?: boolean;
  debugHud?: boolean;
}) {
  const ref = useRef<HTMLVideoElement | null>(null);
  const [status, setStatus] = useState<"loading" | "playing" | "error">("loading");
  const [errMsg, setErrMsg] = useState<string>("");
  const [stillTick, setStillTick] = useState(0);
  const [stillOk, setStillOk] = useState(false);
  const [usingSnapshot, setUsingSnapshot] = useState(false);
  const [overlayOn, setOverlayOn] = useState(true);
  const [retryTick, setRetryTick] = useState(0);
  const [hud, setHud] = useState<{
    fps: number | null;
    p50: number | null;
    p95: number | null;
    p99: number | null;
    stalls100: number | null;
    dropped: number | null;
    total: number | null;
  } | null>(null);

  const gapRef = useRef<number[]>([]);
  const lastFrameTsRef = useRef<number | null>(null);
  const hudTimerRef = useRef<number | null>(null);
  const hudActiveRef = useRef(false);
  const rvfcIdRef = useRef<number | null>(null);
  const samplesRef = useRef<HudSample[]>([]);

  useEffect(() => {
    if (!onVideoReady) return;
    const el = ref.current;
    onVideoReady(el);
    return () => onVideoReady(null);
  }, [onVideoReady]);

  const STREAM_LOCKS = (globalThis as any).__METABONK_STREAM_LOCKS__ as Map<string, number> | undefined;
  const LOCKS: Map<string, number> =
    STREAM_LOCKS ??
    (() => {
      const m = new Map<string, number>();
      (globalThis as any).__METABONK_STREAM_LOCKS__ = m;
      return m;
    })();

  useEffect(() => {
    const el = ref.current;
    if (!el || !url) return;
    if (onVideoReady) onVideoReady(el);
    setStatus("loading");
    setErrMsg("");
    setStillOk(false);
    setUsingSnapshot(false);
    const MediaSourceImpl = (window as any).MediaSource as typeof MediaSource | undefined;
    if (!MediaSourceImpl) {
      el.src = url;
      return;
    }

    const lockKey = String(exclusiveKey || "").trim();
    let lockToken: number | null = null;
    if (lockKey) {
      if (LOCKS.has(lockKey)) {
        setErrMsg("stream already in use by another tile");
        setStatus("error");
        try {
          el.removeAttribute("src");
          el.load();
        } catch {}
        return;
      }
      lockToken = Date.now() + Math.floor(Math.random() * 100000);
      LOCKS.set(lockKey, lockToken);
    }

    const ms = new MediaSourceImpl();
    const objUrl = URL.createObjectURL(ms);
    el.src = objUrl;
    el.muted = true;
    el.playsInline = true;
    el.autoplay = true;
    el.preload = "auto";

    const ac = new AbortController();
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let sb: SourceBuffer | null = null;
    let queue: Uint8Array[] = [];
    let ended = false;

    const cleanup = () => {
      try {
        ac.abort();
      } catch {}
      try {
        if (reader) reader.cancel().catch(() => {});
      } catch {}
      try {
        if (!ended && ms.readyState === "open") ms.endOfStream();
      } catch {}
      ended = true;
      if (lockKey && lockToken != null) {
        try {
          if (LOCKS.get(lockKey) === lockToken) LOCKS.delete(lockKey);
        } catch {}
      }
      try {
        URL.revokeObjectURL(objUrl);
      } catch {}
    };

    const fallbackToSnapshot = (msg?: string) => {
      if (msg) setErrMsg(msg);
      setStatus("error");
      cleanup();
      try {
        el.removeAttribute("src");
        el.load();
      } catch {
        // ignore
      }
    };

    const pump = () => {
      if (!sb || sb.updating) return;
      const next = queue.shift();
      if (!next) return;
      try {
        sb.appendBuffer(next);
      } catch {
        queue = [];
        fallbackToSnapshot("appendBuffer failed (bad segment boundary or buffer error)");
      }
    };

    const trim = () => {
      if (!sb || sb.updating) return;
      try {
        const t = el.currentTime;
        if (t > 15) sb.remove(0, t - 10);
      } catch {
        // ignore
      }
    };

    const onOpen = async () => {
      try {
        let r: Response | null = null;
        for (let attempt = 0; attempt < 8; attempt++) {
          r = await fetch(url, { signal: ac.signal, cache: "no-store" });
          if (r.status !== 429) break;
          try {
            await new Promise((res) => setTimeout(res, 350));
          } catch {
            break;
          }
        }
        if (!r || !r.ok || !r.body) throw new Error(`stream fetch failed: ${r?.status ?? "no-response"}`);
        reader = r.body.getReader();

        let mp4Buf = new Uint8Array(0);
        let initParts: Uint8Array[] = [];
        let initReady = false;
        let segParts: Uint8Array[] = [];
        let sawMoof = false;

        const feedBytes = (value: Uint8Array) => {
          const merged = new Uint8Array(mp4Buf.length + value.length);
          merged.set(mp4Buf, 0);
          merged.set(value, mp4Buf.length);
          mp4Buf = merged;

          while (true) {
            const parsed = tryParseMp4Box(mp4Buf);
            if (!parsed) break;
            const { box, rest } = parsed;
            mp4Buf = rest;

            if (!initReady) {
              initParts.push(box.data);
              if (box.type === "moov") {
                const initSeg = concatParts(initParts);
                const sniffed = findAvc1Codec(initSeg);
                const codecCandidates = sniffed
                  ? [sniffed]
                  : ["avc1.42E01E", "avc1.4D401E", "avc1.64001E"];
                let picked: string | null = null;
                for (const c of codecCandidates) {
                  const mime = `video/mp4; codecs="${c}"`;
                  try {
                    if ((MediaSourceImpl as any).isTypeSupported(mime)) {
                      picked = c;
                      break;
                    }
                  } catch {
                    // ignore
                  }
                }
                if (!picked) {
                  fallbackToSnapshot(`MSE codec unsupported (sniffed=${sniffed ?? "none"})`);
                  return;
                }
                sb = ms.addSourceBuffer(`video/mp4; codecs="${picked}"`);
                sb.mode = "segments";
                sb.addEventListener("updateend", () => {
                  pump();
                  trim();
                });
                queue.push(initSeg);
                initParts = [];
                initReady = true;
                pump();
              }
              continue;
            }

            if (!segParts.length) {
              if (box.type === "styp" || box.type === "moof") {
                segParts.push(box.data);
                sawMoof = box.type === "moof";
              }
              continue;
            }
            segParts.push(box.data);
            if (box.type === "moof") sawMoof = true;
            if (box.type === "mdat" && sawMoof) {
              queue.push(concatParts(segParts));
              segParts = [];
              sawMoof = false;
              pump();
            }
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value || !value.length) continue;
          feedBytes(value);
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e ?? "stream error");
        setErrMsg(msg);
        setStatus("error");
        fallbackToSnapshot(msg);
      } finally {
        try {
          if (!ended && ms.readyState === "open") ms.endOfStream();
        } catch {}
        ended = true;
      }
    };

    ms.addEventListener("sourceopen", onOpen, { once: true });
    el.play().catch(() => {});
    return cleanup;
  }, [url, exclusiveKey, retryTick]);

  useEffect(() => {
    if (!fallbackUrl) return;
    if (status !== "error") return;
    const id = window.setInterval(() => setStillTick((t) => (t + 1) % 10_000), 1500);
    return () => window.clearInterval(id);
  }, [fallbackUrl, status]);

  useEffect(() => {
    if (status !== "error") return;
    if (errMsg.includes("stream already in use")) return;
    const id = window.setTimeout(() => setRetryTick((t) => (t + 1) % 10_000), 3500);
    return () => window.clearTimeout(id);
  }, [status, errMsg]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onPlaying = () => setStatus("playing");
    const onError = () => {
      setStatus("error");
      try {
        const me = (el as any).error;
        if (me && typeof me.code === "number") setErrMsg(`media error (code ${me.code})`);
      } catch {}
    };
    const onTime = () => {
      try {
        if (!el.paused && Number.isFinite(el.currentTime) && el.currentTime > 0) setStatus("playing");
      } catch {}
    };
    el.addEventListener("playing", onPlaying);
    el.addEventListener("error", onError);
    el.addEventListener("timeupdate", onTime);
    return () => {
      el.removeEventListener("playing", onPlaying);
      el.removeEventListener("error", onError);
      el.removeEventListener("timeupdate", onTime);
    };
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || !debugHud) {
      setHud(null);
      gapRef.current = [];
      lastFrameTsRef.current = null;
      hudActiveRef.current = false;
      samplesRef.current = [];
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
      setHud({
        fps: null,
        p50: null,
        p95: null,
        p99: null,
        stalls100: null,
        dropped: null,
        total: null,
      });
      return;
    }

    hudActiveRef.current = true;
    gapRef.current = [];
    lastFrameTsRef.current = null;

    const onFrame = (now: number) => {
      if (!hudActiveRef.current) return;
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

    const computeHud = () => {
      if (!hudActiveRef.current) return;
      const gaps = gapRef.current.slice().sort((a, b) => a - b);
      const toPct = (p: number) => {
        if (!gaps.length) return null;
        const k = (p / 100) * (gaps.length - 1);
        const lo = Math.floor(k);
        const hi = Math.ceil(k);
        if (lo === hi) return gaps[lo];
        const frac = k - lo;
        return gaps[lo] * (1 - frac) + gaps[hi] * frac;
      };
      const avg = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
      const fps = avg > 0 ? 1000 / avg : null;
      const p50 = toPct(50);
      const p95 = toPct(95);
      const p99 = toPct(99);
      const stalls100 = gaps.length ? gaps.filter((g) => g > 100).length : null;
      let dropped: number | null = null;
      let total: number | null = null;
      try {
        const q = (el as any).getVideoPlaybackQuality?.();
        if (q) {
          dropped = Number.isFinite(q.droppedVideoFrames) ? q.droppedVideoFrames : null;
          total = Number.isFinite(q.totalVideoFrames) ? q.totalVideoFrames : null;
        }
      } catch {
        dropped = null;
        total = null;
      }
      setHud({
        fps,
        p50,
        p95,
        p99,
        stalls100,
        dropped,
        total,
      });
      const samples = samplesRef.current;
      samples.push({
        t: Date.now(),
        fps,
        p95,
        p99,
        stalls100,
        dropped,
        total,
      });
      if (samples.length > 1800) samples.splice(0, samples.length - 1800);
    };

    hudTimerRef.current = window.setInterval(computeHud, 1000);
    return () => {
      hudActiveRef.current = false;
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
      source: "mse",
      collectedAt: new Date().toISOString(),
      samples: samplesRef.current,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `stream_jank_mse_${Date.now()}.json`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 3000);
  };

  const showOverlayRaw = status !== "playing" && !(fallbackUrl && stillOk);

  useEffect(() => {
    const delay = showOverlayRaw ? 450 : 260;
    const id = window.setTimeout(() => setOverlayOn(showOverlayRaw), delay);
    return () => window.clearTimeout(id);
  }, [showOverlayRaw]);

  useEffect(() => {
    if (!fallbackUrl) {
      setUsingSnapshot(false);
      return;
    }
    setUsingSnapshot(status !== "playing" && Boolean(stillOk));
  }, [fallbackUrl, status, stillOk]);

  return (
    <div className="mse-wrap">
      {fallbackUrl ? (
        <img
          className={className}
          src={`${fallbackUrl}${fallbackUrl.includes("?") ? "&" : "?"}t=${stillTick}`}
          style={{ display: usingSnapshot ? "block" : "none" }}
          alt="latest frame"
          onLoad={() => setStillOk(true)}
          onError={() => setStillOk(false)}
        />
      ) : null}
      <video ref={ref} className={className} muted playsInline autoPlay />
      {overlayOn ? (
        <div className="mse-overlay">
          <div className="mse-overlay-inner">
            <div className="mse-overlay-title">{status === "error" ? "SIGNAL LOST" : "NO KEYFRAME"}</div>
            <div className="mse-overlay-sub muted">{status === "error" ? "reconnecting…" : "syncing…"}</div>
            {debug && status === "error" ? (
              <div className="mse-overlay-debug muted">
                {errMsg || "media decode error"} <span className="muted">(try `METABONK_STREAM_CODEC=h264` or check worker CORS)</span>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      {debug && usingSnapshot ? (
        <div className="mse-overlay" style={{ pointerEvents: "none", opacity: 0.65 }}>
          <div className="mse-overlay-inner">
            <div className="mse-overlay-title">SNAPSHOT</div>
            <div className="mse-overlay-sub muted">video not playing; using /frame.jpg</div>
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
          <div className="mse-hud-row">
            dropped: {hud?.dropped ?? "n/a"} / {hud?.total ?? "n/a"}
          </div>
          <button className="mse-hud-btn" onClick={(e) => { e.preventDefault(); e.stopPropagation(); exportJson(); }}>
            Export JSON
          </button>
        </div>
      ) : null}
    </div>
  );
}
