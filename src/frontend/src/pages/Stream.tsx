import { useQuery } from "@tanstack/react-query";
import { type SyntheticEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { Canvas, useFrame } from "@react-three/fiber";
import { EffectComposer, Bloom, Scanline, ChromaticAberration, Glitch } from "@react-three/postprocessing";
import { BlendFunction, GlitchMode } from "postprocessing";
import * as THREE from "three";
import type { AttractHighlights, Event, Heartbeat, HistoricLeaderboardEntry, HofEntry, TimelineFuse, TimelineState } from "../api";
import {
  fetchAttractHighlights,
  fetchBettingState,
  fetchFeatured,
  fetchHistoricLeaderboard,
  fetchHofTop,
  fetchPollState,
  fetchStatus,
  fetchTimeline,
  fetchTimelineFuse,
  fetchWorkers,
} from "../api";
import { useEventStream } from "../hooks";
import iconSheetUrl from "../assets/icon_sheet.png";
import { iconIndex, iconVariantClass, type IconKey, type IconVariant } from "../lib/icon_map";
import { PROGRESS_GOALS, type GoalTier, type ProgressGoal } from "../lib/megabonk_progress";
import { UI_TOKENS } from "../lib/ui_tokens";
import { timeAgo } from "../lib/format";
import MseMp4Video from "../components/MseMp4Video";
import Go2rtcWebRTC from "../components/Go2rtcWebRTC";
import RouteScope from "../components/RouteScope";
import { bumpWebglCount, reportWebglLost } from "../hooks/useWebglCounter";
import { useWebglResetNonce } from "../hooks/useWebglReset";

const HUD_W = 3840;
const HUD_H = 2160;
const DEBUG_ON = import.meta.env.DEV && new URLSearchParams(window.location.search).get("debug") === "1";
const DEBUG_HUD = new URLSearchParams(window.location.search).get("debugHud") === "1";
const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
const computeUiScale = (w: number, h: number) => {
  const scale = Math.min(w / HUD_W, h / HUD_H);
  return clamp(scale, 1, 3);
};
const computeTier = (w: number) => {
  if (w >= 3840) return "2160";
  if (w >= 2560) return "1440";
  if (w >= 1920) return "1080";
  return "720";
};
const hashString = (input: string) => {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

function CurvedScreen({
  texture,
  width = 1.6,
  height = 0.9,
  curvature = 0.22,
}: {
  texture: THREE.Texture | null;
  width?: number;
  height?: number;
  curvature?: number;
}) {
  const geom = useMemo(() => {
    const g = new THREE.PlaneGeometry(width, height, 64, 32);
    const pos = g.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = -Math.pow(Math.abs(x) / width, 2) * curvature;
      pos.setZ(i, z);
    }
    g.computeVertexNormals();
    return g;
  }, [width, height, curvature]);

  useFrame(() => {
    if (texture && "offset" in texture) {
      texture.needsUpdate = true;
    }
  });

  return (
    <mesh geometry={geom}>
      {texture ? (
        <meshStandardMaterial map={texture} emissive="#ffffff" emissiveMap={texture} emissiveIntensity={1.25} toneMapped={false} />
      ) : (
        <meshStandardMaterial emissive="#220000" emissiveIntensity={0.4} toneMapped={false} />
      )}
    </mesh>
  );
}

function HoloStreamCanvas({
  videoEl,
  fallbackUrl,
  surprise,
  glitch,
}: {
  videoEl: HTMLVideoElement | null;
  fallbackUrl?: string;
  surprise: number;
  glitch: boolean;
}) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const [lost, setLost] = useState(false);
  const resetNonce = useWebglResetNonce();
  useEffect(() => {
    bumpWebglCount(1);
    return () => bumpWebglCount(-1);
  }, []);

  useEffect(() => {
    if (!videoEl) return;
    const tex = new THREE.VideoTexture(videoEl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    setTexture(tex);
    return () => {
      tex.dispose();
    };
  }, [videoEl]);

  useEffect(() => {
    if (!fallbackUrl || videoEl) return;
    const loader = new THREE.TextureLoader();
    loader.load(fallbackUrl, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      setTexture(tex);
    });
  }, [fallbackUrl, videoEl]);

  const chromaOffset = 0.001 + surprise * 0.004;

  if (lost) {
    return <div className="canvas-placeholder">WebGL context lost</div>;
  }
  return (
    <Canvas
      className="holo-r3f-canvas"
      key={`stream-holo-${resetNonce}`}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      camera={{ position: [0, 0, 2.2], fov: 45 }}
      onCreated={({ gl }) => {
        gl.outputColorSpace = THREE.SRGBColorSpace;
        const onLost = (evt: Event) => {
          evt.preventDefault();
          setLost(true);
          reportWebglLost();
        };
        const onRestore = () => setLost(false);
        gl.domElement.addEventListener("webglcontextlost", onLost, { passive: false });
        gl.domElement.addEventListener("webglcontextrestored", onRestore);
      }}
    >
      <color attach="background" args={["#020405"]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[0.5, 0.4, 1.8]} intensity={0.35} color="#7ffff0" />
      <group>
        <CurvedScreen texture={texture} />
        <mesh position={[0, 0, 0.01]}>
          <planeGeometry args={[1.64, 0.94]} />
          <meshBasicMaterial color={new THREE.Color(0x6fffe6)} opacity={0.05} transparent />
        </mesh>
      </group>
      <EffectComposer>
        <Bloom intensity={1.1} luminanceThreshold={0.2} />
        <Scanline density={1.5} opacity={0.25} />
        <ChromaticAberration blendFunction={BlendFunction.NORMAL} offset={[chromaOffset, 0]} />
        <Glitch
          delay={[0.6, 2.2]}
          duration={[0.2, 0.45]}
          strength={[0.2, 0.6]}
          mode={GlitchMode.SPORADIC}
          active={glitch}
        />
      </EffectComposer>
    </Canvas>
  );
}

type RailTab = "moments" | "community" | "bucks" | "poll" | "progress" | "controls";

type SpriteIconProps = {
  idx: number;
  size?: number;
  className?: string;
  title?: string;
};

type SpriteAtlas = { cols: number; rows: number; tilePx: number; offsetXPx: number; offsetYPx: number; maxIdx: number };

const ICON_ATLAS: SpriteAtlas = { cols: 8, rows: 8, tilePx: 128, offsetXPx: 0, offsetYPx: 0, maxIdx: 63 };

function spriteXY(idx: number, atlas: SpriteAtlas = ICON_ATLAS) {
  const clamped = Math.max(0, Math.min(atlas.maxIdx, Math.floor(idx)));
  return { x: clamped % atlas.cols, y: Math.floor(clamped / atlas.cols) };
}

function SafeGuides() {
  const gs = UI_TOKENS.safe.graphics;
  const as = UI_TOKENS.safe.action;
  const w = UI_TOKENS.resolution.w;
  const h = UI_TOKENS.resolution.h;
  const legacyTitle = { x: Math.round(w * 0.1), y: Math.round(h * 0.1) };
  const legacyAction = { x: Math.round(w * 0.05), y: Math.round(h * 0.05) };
  const hdTitle = { x: Math.round(w * 0.05), y: Math.round(h * 0.05) };
  const hdAction = { x: Math.round(w * 0.035), y: Math.round(h * 0.035) };
  return (
    <div className="safe-guides" aria-hidden="true">
      <div className="safe-rect title legacy" style={{ inset: `${legacyTitle.y}px ${legacyTitle.x}px` }}>
        <span className="safe-label">legacy title-safe (80%)</span>
      </div>
      <div className="safe-rect action90 legacy" style={{ inset: `${legacyAction.y}px ${legacyAction.x}px` }}>
        <span className="safe-label">legacy action-safe (90%)</span>
      </div>
      <div className="safe-rect title hd" style={{ inset: `${hdTitle.y}px ${hdTitle.x}px` }}>
        <span className="safe-label">hd title-safe (90%)</span>
      </div>
      <div className="safe-rect action90 hd" style={{ inset: `${hdAction.y}px ${hdAction.x}px` }}>
        <span className="safe-label">hd action-safe (93%)</span>
      </div>
      <div className="safe-rect graphics" style={{ inset: `${gs.y}px ${gs.x}px` }}>
        <span className="safe-label">graphics-safe</span>
      </div>
      <div className="safe-rect action" style={{ inset: `${as.y}px ${as.x}px` }}>
        <span className="safe-label">action-safe</span>
      </div>
      <div className="safe-ruler-x" style={{ top: gs.y - 18 }}>
        <span>0</span>
        <span>{UI_TOKENS.resolution.w}</span>
      </div>
      <div className="safe-ruler-y" style={{ left: gs.x - 28 }}>
        <span>0</span>
        <span>{UI_TOKENS.resolution.h}</span>
      </div>
    </RouteScope>
  );
}

function capCount(n: number, cap = 99) {
  const v = Math.max(0, Math.floor(Number(n) || 0));
  if (v > cap) return `${cap}+`;
  return String(v);
}

function SpriteIcon({ idx, size = 16, className, title, variant = "normal" as IconVariant }: SpriteIconProps & { variant?: IconVariant }) {
  const { x, y } = spriteXY(idx);
  return (
    <span
      className={`mb-sprite ${iconVariantClass(variant)} ${className ?? ""}`}
      style={
        {
          width: size,
          height: size,
          ["--sx" as any]: x,
          ["--sy" as any]: y,
          ["--ox" as any]: `${ICON_ATLAS.offsetXPx}px`,
          ["--oy" as any]: `${ICON_ATLAS.offsetYPx}px`,
        } as any
      }
      title={title}
      aria-hidden="true"
    />
  );
}

function sheetIcon(key: IconKey) {
  return iconIndex(key, 63);
}

function eventIconIdx(eventType: string) {
  const t = String(eventType ?? "");
  if (t === "Overcrit" || t === "NewMaxHit") return sheetIcon("overcrit");
  if (t === "OverrunStart") return sheetIcon("warning");
  if (t === "Disaster") return sheetIcon("warning");
  if (t === "LootDrop" || t === "BountyClaimed" || t === "BountyCreated") return sheetIcon("loot_chest");
  if (t === "WeirdBuild") return sheetIcon("loot_chest");
  if (t === "Heal") return sheetIcon("health");
  if (t === "Eureka") return sheetIcon("info");
  if (t === "EpisodeEnd") return sheetIcon("heartbreak");
  if (t === "EpisodeStart") return sheetIcon("time");
  if (t === "Clutch" || t === "ClutchClip") return sheetIcon("moment");
  if (t === "ChatSpike" || t === "CommunityPin") return sheetIcon("moment");
  if (t === "WorkerOnline") return sheetIcon("training");
  return sheetIcon("unknown");
}

function scoreOf(w: Heartbeat | null | undefined) {
  return (w?.steam_score ?? w?.reward ?? 0) as number;
}

function scoreDisplay(w: Heartbeat | null | undefined, digits = 2) {
  if (!w) return "—";
  if (w.steam_score == null && w.reward == null) return "—";
  return fmt(scoreOf(w), digits);
}

function fmt(n: number | null | undefined, digits = 2) {
  if (n === null || n === undefined) return "—";
  if (!Number.isFinite(n)) return "—";
  return Number(n).toFixed(digits);
}

function fmtCompact(n: number | null | undefined) {
  if (n === null || n === undefined) return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}m`;
  if (abs >= 1_000) return `${(v / 1_000).toFixed(1)}k`;
  return `${Math.round(v)}`;
}

function fmtDelta(v: number | null | undefined, digits = 2) {
  if (v == null || !Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : v < 0 ? "−" : "";
  return `${sign}${Math.abs(v).toFixed(digits)}`;
}

function sparkPath(values: number[], w = 60, h = 16) {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1e-6, max - min);
  const step = w / Math.max(1, values.length - 1);
  return values
    .map((v, i) => {
      const x = i * step;
      const y = h - ((v - min) / range) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

type Chip = {
  label: string;
  icon: IconKey;
  ts: number;
  priority: number;
  value?: string;
};

const CHIP_TTL_MS = 8000;
const CHIP_COOLDOWN_MS = 2000;

function trimHistory<T extends { ts: number }>(arr: T[], windowMs: number, now: number) {
  let i = 0;
  while (i < arr.length && arr[i].ts < now - windowMs) i++;
  return i > 0 ? arr.slice(i) : arr;
}

function deltaInWindow(history: Array<{ ts: number; v: number }>, windowMs: number, now: number) {
  if (!history.length) return null;
  const latest = history[history.length - 1];
  const cutoff = now - windowMs;
  let base = history[0];
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].ts <= cutoff) {
      base = history[i];
      break;
    }
  }
  return latest.v - base.v;
}

function ratePerMinute(history: Array<{ ts: number; v: number }>, windowMs: number, now: number) {
  const delta = deltaInWindow(history, windowMs, now);
  if (delta == null) return null;
  const elapsed = Math.min(windowMs, Math.max(1, now - history[0].ts));
  return (delta / elapsed) * 60000;
}

function pickInterestingEvents(events: Event[], limit = 6) {
  const deny = new Set(["Telemetry"]);
  const allow = new Set([
    "WorkerOnline",
    "EpisodeStart",
    "EpisodeEnd",
    "OverrunStart",
    "Disaster",
    "Overcrit",
    "NewMaxHit",
    "LootDrop",
    "Heal",
    "Eureka",
    "Clutch",
    "ClutchClip",
    "WeirdBuild",
    "ChatSpike",
    "BountyCreated",
    "BountyClaimed",
    "CommunityPin",
  ]);

  const out: Event[] = [];
  const seen = new Set<string>();
  const newestFirst = [...events].reverse();
  for (const e of newestFirst) {
    if (!e?.event_type) continue;
    if (deny.has(e.event_type)) continue;
    if (!allow.has(e.event_type)) continue;
    const key = `${e.event_type}:${e.instance_id ?? ""}:${String(e.message ?? "").slice(0, 60)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(e);
    if (out.length >= limit) break;
  }
  return out;
}

function fuseBars(fuse: TimelineFuse | null | undefined) {
  if (!fuse?.segments?.length) return null;
  const segs = fuse.segments.slice(0, 220);
  const total = segs.reduce((a, s) => a + Math.max(0, Number((s as any).duration_s ?? 0)), 0);
  if (!(total > 0)) return { w: 1280, h: 86, bars: [] as any[], xWr: 0, xNow: 1270, eraLines: [] as any[], nowLabelX: 1152, wrLabelX: 128, hot: false };

  const w = 1280;
  const h = 86;
  const baseH = 118;
  const scaleY = (v: number) => Math.round((v / baseH) * h);
  const padL = 14;
  const padR = 18;
  const innerW = w - padL - padR;

  let cum = 0;
  const bars = segs.map((s: any) => {
    const dur = Math.max(0, Number(s.duration_s ?? 0));
    const x = padL + (cum / total) * innerW;
    const bw = Math.max(1, (dur / total) * innerW);
    cum += dur;
    const res = String(s.result ?? "").toLowerCase();
    const cls = res.includes("death") || res.includes("fail") ? "fuse-bar-death" : res.includes("victory") || res.includes("win") ? "fuse-bar-victory" : "fuse-bar-neutral";
    const title = `${s.instance_id ?? "—"} · ${dur.toFixed(1)}s · ${String(s.result ?? "")}`;
    return { x, w: bw, cls, title };
  });

  const wr = Number(fuse.world_record_duration_s ?? 0);
  const xWr = wr > 0 ? padL + Math.min(1, wr / total) * innerW : padL + innerW * 0.85;

  const eraLines: Array<{ x: number; label: string; color: string }> = [];
  try {
    const eras = fuse.eras ?? [];
    for (const e of eras.slice(0, 8)) {
      const ts = Number((e as any).ts ?? 0);
      if (!(ts > 0)) continue;
      const idx = segs.findIndex((s: any) => Number((s as any).ts ?? 0) <= ts);
      if (idx < 0) continue;
      let dd = 0;
      for (let i = 0; i < idx; i++) dd += Math.max(0, Number((segs[i] as any).duration_s ?? 0));
      eraLines.push({ x: padL + (dd / total) * innerW, label: String((e as any).label ?? ""), color: String((e as any).color ?? "#ff88ff") });
    }
  } catch {
    // ignore
  }

  const xNow = padL + innerW;
  const titleSafeX = Math.round(w * 0.1);
  const nowLabelX = w - titleSafeX;
  const wrLabelX = clamp(xWr + 6, titleSafeX, nowLabelX - 24);

  let best = Number.POSITIVE_INFINITY;
  for (const s of segs) {
    const d = Math.max(0, Number((s as any).duration_s ?? 0));
    if (d > 0 && d < best) best = d;
  }
  const hot = Number.isFinite(best) && wr > 0 ? best <= wr * 1.08 : false;

  return {
    w,
    h,
    bars,
    xWr,
    xNow,
    padL,
    eraLines,
    nowLabelX,
    wrLabelX,
    hot,
    yCenter: scaleY(68),
    barY: scaleY(48),
    barH: Math.max(6, scaleY(24)),
    wrY1: scaleY(34),
    wrY2: scaleY(92),
    nowY1: scaleY(10),
    nowY2: scaleY(110),
    nowDotY: scaleY(14),
    nowLabelY: scaleY(18),
    wrLabelY: scaleY(34),
  };
}

function AttractOverlay({
  hof,
  shame,
  clips,
  variant = "stage",
}: {
  hof: HofEntry[];
  shame: Array<{ instance_id: string; duration_s: number; score: number; ts: number }>;
  clips: AttractHighlights | null;
  variant?: "stage" | "fullscreen";
}) {
  const fameClip = clips?.fame?.clip_url ? `/api${clips.fame.clip_url}` : null;
  const shameClip = clips?.shame?.clip_url ? `/api${clips.shame.clip_url}` : null;
  return (
    <div className={`attract-root ${variant === "fullscreen" ? "attract-root-fullscreen" : ""}`}>
      <div className="attract-ui">
        <div className="attract-crt" />
        <div className="attract-scanlines" />
        <div className="attract-title">METABONK</div>
        <div className="attract-sub">ARCADE ATTRACT MODE</div>
        <div className="insert-coin">INSERT COIN</div>
        <div className="attract-panels">
          <div className="attract-panel">
            <div className="attract-panel-head">HALL OF FAME</div>
            {fameClip ? (
              <>
                <video className="attract-clip" src={fameClip} autoPlay muted playsInline loop />
                <div className="attract-clip-meta">
                  <span className="badge">BEST</span>
                  <span className="muted">{clips?.fame?.instance_id ?? "—"}</span>
                  <span className="muted">{(clips?.fame?.duration_s ?? 0).toFixed(0)}s</span>
                  <span className="muted">{(clips?.fame?.score ?? 0).toFixed(2)}</span>
                </div>
              </>
            ) : !hof.length ? (
              <div className="muted">waiting for hot runs…</div>
            ) : (
              <div className="attract-list">
                {hof.slice(0, 6).map((x, i) => (
                  <div key={i} className="attract-row">
                    <span className="muted">#{i + 1}</span>
                    <span className="attract-row-title">{x.instance_id ?? "—"}</span>
                    <span className="attract-row-score">{(x.score ?? 0).toFixed(0)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="attract-panel">
            <div className="attract-panel-head">WALL OF SHAME</div>
            {shameClip ? (
              <>
                <video className="attract-clip" src={shameClip} autoPlay muted playsInline loop />
                <div className="attract-clip-meta">
                  <span className="badge">WORST (LONG)</span>
                  <span className="muted">{clips?.shame?.instance_id ?? "—"}</span>
                  <span className="muted">{(clips?.shame?.duration_s ?? 0).toFixed(0)}s</span>
                  <span className="muted">{(clips?.shame?.score ?? 0).toFixed(2)}</span>
                </div>
              </>
            ) : !shame.length ? (
              <div className="muted">waiting for disasters…</div>
            ) : (
              <div className="attract-list">
                {shame.slice(0, 6).map((x, i) => (
                  <div key={i} className="attract-row">
                    <span className="muted">#{i + 1}</span>
                    <span className="attract-row-title">{x.instance_id ?? "—"}</span>
                    <span className="attract-row-score">{x.duration_s.toFixed(0)}s</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StreamTile({
  w,
  slotLabel,
  cornerTag,
  className,
  onClick,
  impact,
  focused,
  rankUp,
  scoreDelta30s,
  stepRate,
  resets5m,
}: {
  w: Heartbeat | null;
  slotLabel: string;
  cornerTag?: string;
  className?: string;
  onClick?: () => void;
  impact?: boolean;
  focused?: boolean;
  rankUp?: number | null;
  scoreDelta30s?: number | null;
  stepRate?: number | null;
  resets5m?: number;
}) {
  const nowMs = Date.now();
  const overrun = Boolean(w?.overrun);
  const name = w?.display_name ?? w?.instance_id ?? "—";
  const streamUrl = w?.stream_url ?? null;
  const isVideo = (w?.stream_type ?? "").toLowerCase() === "mp4";
  const controlUrl = String((w as any)?.control_url ?? "").trim();
  const frameUrl = controlUrl ? `${controlUrl.replace(/\/+$/, "")}/frame.jpg` : "";
  const go2rtcBase = String((w as any)?.go2rtc_base_url ?? "").trim();
  const go2rtcName = String((w as any)?.go2rtc_stream_name ?? "").trim();
  const useGo2rtc = Boolean(go2rtcBase && go2rtcName);
  const featuredRole = String((w as any)?.featured_role ?? "").toLowerCase();

  const danger = Math.max(0, Math.min(1, Number(w?.danger_level ?? (w?.survival_prob != null ? 1 - Number(w.survival_prob) : 0)) || 0));
  const dangerPct = Math.round(danger * 100);
  const policy = String(w?.policy_name ?? "");
  const hype = String((w as any)?.hype_label ?? "");
  const shame = String((w as any)?.shame_label ?? "");
  const showLabels = Boolean(focused);
  const noFeed = !streamUrl || !isVideo;
  let lastFrameAgeS: number | null = null;
  try {
    const raw = Number(w?.stream_last_frame_ts ?? 0);
    if (raw > 0) {
      const frameMs = raw > 10_000_000_000 ? raw : raw * 1000;
      const ageS = (nowMs - frameMs) / 1000;
      if (Number.isFinite(ageS) && ageS >= 0) lastFrameAgeS = ageS;
    }
  } catch {
    lastFrameAgeS = null;
  }
  const [standbyStable, setStandbyStable] = useState(true);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (noFeed || lastFrameAgeS == null) {
      if (!standbyStable) setStandbyStable(true);
      return;
    }
    if (standbyStable) {
      if (lastFrameAgeS <= 2.0) setStandbyStable(false);
    } else {
      if (lastFrameAgeS >= 6.0) setStandbyStable(true);
    }
  }, [noFeed, lastFrameAgeS, standbyStable]);

  const standby = standbyStable;
  const offlineDetail = (() => {
    if (featuredRole === "background") return "not featured";
    const backend = String(w?.stream_backend ?? "").toLowerCase();
    if (backend.includes("cpu")) return "CPU fallback active";
    const err = String((w as any)?.fifo_stream_last_error ?? w?.streamer_last_error ?? w?.stream_error ?? "").trim();
    if (err) return err.slice(0, 80);
    return "syncing keyframes";
  })();

  const score = scoreOf(w);
  const dps = w?.clearing_dps ?? null;
  const step = w?.step ?? 0;
  const epTime = Number(w?.episode_t ?? 0);
  const progressRaw = epTime > 0 ? epTime / 900 : Number(step) / 10000;
  const progress = Math.max(0, Math.min(1, Number.isFinite(progressRaw) ? progressRaw : 0));
  const progressPct = Math.round(progress * 100);

  const drops = Number((w as any)?.luck_drop_count ?? 0);
  const legs = Number((w as any)?.luck_legendary_count ?? 0);
  const streakCur = Number((w as any)?.streak_current ?? 0) || null;
  const streakBest = Number((w as any)?.streak_best ?? 0) || null;
  const status = String(w?.status ?? "").toLowerCase();
  const menuStuck = status.includes("menu") || status.includes("stuck");
  const actionsPerMin = stepRate != null ? Math.max(0, stepRate) : null;
  const surprise = clamp(Number(w?.action_entropy ?? danger), 0, 1);
  const glitch = surprise > 0.75 || danger > 0.6;
  const overlayBoxes = useMemo(() => {
    const seed = hashString(String(w?.instance_id ?? slotLabel));
    const boxes = Array.from({ length: 3 }, (_, i) => {
      const r1 = ((seed >> (i * 4)) & 0xf) / 15;
      const r2 = ((seed >> (i * 6 + 3)) & 0xf) / 15;
      const r3 = ((seed >> (i * 5 + 7)) & 0xf) / 15;
      const wPct = 18 + r1 * 32;
      const hPct = 14 + r2 * 28;
      const xPct = 6 + r3 * (100 - wPct - 12);
      const yPct = 10 + ((r1 + r2) * 0.5) * (100 - hPct - 18);
      return { xPct, yPct, wPct, hPct };
    });
    return boxes;
  }, [w?.instance_id, slotLabel]);

  const policyKey: IconKey = "policy";
  const hypeKey: IconKey = "hype";
  const shameKey: IconKey = "shame";

  return (
    <div
      className={`stream-tile ${className ?? ""} ${overrun ? "stream-tile-overrun" : ""} ${impact ? "stream-tile-impact" : ""} ${
        focused ? "stream-tile-focused" : ""
      } ${noFeed ? "stream-tile-nofeed" : ""}`}
      style={{ cursor: onClick ? "pointer" : "default" }}
      onClick={onClick}
    >
      <div className={`stream-tile-feed holo-feed ${glitch ? "holo-glitch" : ""}`} style={{ ["--surprise" as any]: surprise }}>
        {cornerTag ? <div className="stream-tile-role">{cornerTag}</div> : null}
        {rankUp && rankUp > 0 ? (
          <div className="stream-rank-burst" aria-label={`rank up ${rankUp}`}>
            ▲ +{rankUp}
          </div>
        ) : null}
        {focused ? (
          <div className="holo-r3f">
            <HoloStreamCanvas videoEl={videoEl} fallbackUrl={frameUrl || undefined} surprise={surprise} glitch={glitch} />
          </div>
        ) : null}
        <div className="holo-frame" />
        <div className="holo-ghost" />
        <div className="holo-fovea" />
        <div className="holo-scan" />
        <div className="holo-boxes" aria-hidden="true">
          {overlayBoxes.map((b, i) => (
            <div
              key={`box-${i}`}
              className="holo-box"
              style={{ left: `${b.xPct}%`, top: `${b.yPct}%`, width: `${b.wPct}%`, height: `${b.hPct}%` }}
            />
          ))}
        </div>
        {useGo2rtc ? (
          <Go2rtcWebRTC
            baseUrl={go2rtcBase}
            streamName={go2rtcName}
            className={`stream-img ${focused ? "holo-fallback" : ""}`}
            onVideoReady={focused ? setVideoEl : undefined}
            debugHud={DEBUG_HUD}
          />
        ) : streamUrl && isVideo ? (
          <MseMp4Video
            className={`stream-img ${focused ? "holo-fallback" : ""}`}
            url={streamUrl}
            fallbackUrl={frameUrl || undefined}
            exclusiveKey={String(w?.instance_id ?? streamUrl)}
            onVideoReady={focused ? setVideoEl : undefined}
            debug={DEBUG_ON}
            debugHud={DEBUG_HUD}
          />
        ) : null}

        {noFeed && !useGo2rtc ? (
          <div className="mse-overlay" aria-label="no feed">
            <div className="mse-overlay-inner">
              <div className="mse-overlay-title">NO FEED</div>
              <div className="mse-overlay-sub muted">syncing keyframes</div>
              <div className="mse-overlay-debug muted">{offlineDetail}</div>
              {lastFrameAgeS != null ? <div className="mse-overlay-debug muted numeric">last frame: {lastFrameAgeS.toFixed(1)}s</div> : null}
            </div>
          </div>
        ) : null}

        <div className="stream-overlay-top">
          <div className="stream-ov-left">
            <div className="stream-ov-name">
              <span className="stream-ov-persona" title={policy || "policy"}>
                <SpriteIcon idx={sheetIcon(policyKey)} size={20} />
              </span>
              <span className="stream-ov-name-text">{name}</span>
            </div>
            <div className="stream-ov-sub">
              {!standby ? <span className="stream-ov-pill">{slotLabel}</span> : null}
              {!standby && policy ? <span className="stream-ov-pill">{policy}</span> : null}
            </div>
            {standby ? (
              <div className="stream-ov-nofeed">
                <span className="stream-ov-statuschip" title="no GPU feed">
                  <SpriteIcon idx={sheetIcon("no_gpu_feed")} size={16} />
                  <span className="stream-ov-statuschip-text">{noFeed ? "NO FEED" : "NO KEYFRAME"} • syncing keyframes</span>
                  {lastFrameAgeS != null ? <span className="muted numeric">last frame {lastFrameAgeS.toFixed(1)}s</span> : null}
                </span>
              </div>
            ) : (
              <div className="stream-ov-strip">
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="score">
                <SpriteIcon idx={sheetIcon("score")} size={18} />
                <span className="stream-ov-label">{fmt(score, 2)}</span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="rank delta (30s)">
                <SpriteIcon idx={sheetIcon("rank_up")} size={18} />
                <span className="stream-ov-label">{rankUp ? `+${rankUp}` : "0"}</span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="Δscore (30s)">
                <SpriteIcon idx={sheetIcon("speed")} size={18} />
                <span className={`stream-ov-label ${scoreDelta30s != null ? (scoreDelta30s > 0 ? "delta-pos" : scoreDelta30s < 0 ? "delta-neg" : "") : ""}`}>
                  {fmtDelta(scoreDelta30s, 2)}
                </span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="episode time">
                <SpriteIcon idx={sheetIcon("time")} size={18} />
                <span className="stream-ov-label">
                  {epTime > 0 ? `${Math.floor(epTime / 60)}:${String(Math.floor(epTime % 60)).padStart(2, "0")}` : "—"}
                </span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="actions / min">
                <SpriteIcon idx={sheetIcon("training")} size={18} />
                <span className="stream-ov-label">{actionsPerMin == null ? "—" : fmtCompact(actionsPerMin)}</span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="tome progress">
                <SpriteIcon idx={sheetIcon("tome_mastery")} size={18} />
                <span className="stream-ov-label">{drops > 0 ? capCount(drops, 999) : "—"}</span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="streak">
                <SpriteIcon idx={sheetIcon("rarity_star")} size={18} />
                <span className="stream-ov-label">{streakCur != null ? `${streakCur}${streakBest ? `/${streakBest}` : ""}` : "—"}</span>
              </span>
              <span className={`stream-ov-iconpill ${showLabels ? "" : "compact"}`} title="resets (5m)">
                <SpriteIcon idx={sheetIcon("refresh")} size={18} />
                <span className="stream-ov-label">{resets5m != null ? capCount(resets5m, 99) : "—"}</span>
              </span>
              </div>
            )}
            {!standby && (overrun || menuStuck || (showLabels && (hype || shame || drops > 0 || legs > 0))) ? (
              <div className="stream-ov-alerts">
                {showLabels && hype ? (
                  <span className="stream-ov-flag" title={`hype ${hype}`}>
                    <SpriteIcon idx={sheetIcon(hypeKey)} size={16} />
                    <span className="stream-ov-flag-label">{hype}</span>
                  </span>
                ) : null}
                {showLabels && shame ? (
                  <span className="stream-ov-flag" title={`shame ${shame}`}>
                    <SpriteIcon idx={sheetIcon(shameKey)} size={16} />
                    <span className="stream-ov-flag-label">{shame}</span>
                  </span>
                ) : null}
                {showLabels && (drops > 0 || legs > 0) ? (
                  <span className="stream-ov-flag" title="drops / legendaries">
                    <span className="mb-badge-wrap">
                      <SpriteIcon idx={sheetIcon("loot_chest")} size={16} />
                      {drops > 0 ? <span className="mb-badge">{capCount(drops, 99)}</span> : null}
                    </span>
                    <span className="mb-badge-wrap">
                      <SpriteIcon idx={sheetIcon("rarity_star")} size={16} />
                      {legs > 0 ? <span className="mb-badge">{capCount(legs, 99)}</span> : null}
                    </span>
                  </span>
                ) : null}
                {menuStuck ? (
                  <span className="stream-ov-flag danger" title="menu stuck">
                    <SpriteIcon idx={sheetIcon("menu_stuck")} size={16} />
                    <span className="stream-ov-flag-label">MENU STUCK</span>
                  </span>
                ) : null}
                {overrun ? (
                  <span className="stream-ov-flag danger" title="overrun">
                    <SpriteIcon idx={sheetIcon("warning")} size={16} />
                    <span className="stream-ov-flag-label">OVERRUN</span>
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>
          {!standby ? (
            <div className="stream-ov-right">
              <div className="stream-ov-score numeric">{scoreDisplay(w, 2)}</div>
              <div className="stream-ov-kpi">
                step {w?.step ?? 0} • ep {progressPct}%
              </div>
            </div>
          ) : null}
        </div>

        {!standby ? (
          <div className="stream-overlay-bottom">
            <div className="stream-progress">
              <div className="stream-progress-track">
                <div
                  className="stream-progress-fill"
                  style={{
                    width: `${progressPct}%`,
                    background: dangerPct >= 75 ? "rgba(255,68,68,.9)" : dangerPct >= 40 ? "rgba(255,214,10,.85)" : "rgba(34,211,238,.9)",
                  }}
                />
              </div>
              <div className="stream-progress-meta">
                EP {progressPct}% • danger {dangerPct}% • {w?.clearing_dps != null ? `dps ${fmtCompact(w.clearing_dps)}` : "dps —"}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function IdleStreamTile({
  slotLabel,
  cornerTag,
  className,
  children,
}: {
  slotLabel: string;
  cornerTag?: string;
  className?: string;
  children?: any;
}) {
  return (
    <div className={`stream-tile stream-tile-idle ${className ?? ""}`}>
      <div className="stream-tile-feed">
        {cornerTag ? <div className="stream-tile-role">{cornerTag}</div> : null}
        {children ? <div className="stream-idle-content">{children}</div> : <div className="stream-idle-content muted">waiting…</div>}
        <div className="stream-idle-overlay">
          <div className="stream-ov-left">
            <div className="stream-ov-name">
              <span className="stream-ov-persona" title="idle">
                <SpriteIcon idx={sheetIcon("focus")} size={20} />
              </span>
              <span className="stream-ov-name-text">METABONK</span>
            </div>
            <div className="stream-ov-sub">
              <span className="stream-ov-pill">{slotLabel}</span>
            </div>
            <div className="stream-ov-nofeed">
              <span className="stream-ov-statuschip" title="waiting for feed">
                <SpriteIcon idx={sheetIcon("no_gpu_feed")} size={16} />
                <span className="stream-ov-statuschip-text">NO FEED • syncing keyframes</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function HighlightPanel({
  clips,
  leader,
  runnerUp,
  pulse,
}: {
  clips: AttractHighlights | null;
  leader: Heartbeat | null;
  runnerUp: Heartbeat | null;
  pulse?: boolean;
}) {
  const fame = clips?.fame ?? null;
  const clipUrl = fame?.clip_url ? `/api${fame.clip_url}` : null;
  const name = leader?.display_name ?? leader?.instance_id ?? fame?.instance_id ?? "—";
  const score = leader ? scoreOf(leader) : fame?.score ?? null;
  const duration = fame?.duration_s ?? null;
  const runnerScore = runnerUp ? scoreOf(runnerUp) : null;
  const deltaVs2 = score != null && runnerScore != null ? score - runnerScore : null;
  const onVideoReady = (e: SyntheticEvent<HTMLVideoElement>) => {
    try {
      e.currentTarget.playbackRate = 2.0;
    } catch {
      // ignore
    }
  };
  return (
    <div className={`stream-card stream-highlight ${pulse ? "stream-highlight-pulse" : ""}`}>
      <div className="stream-card-title">#1 Replay</div>
      <div className="stream-highlight-media">
        {clipUrl ? (
          <video className="stream-highlight-video" src={clipUrl} autoPlay muted playsInline loop onLoadedMetadata={onVideoReady} />
        ) : (
          <div className="stream-highlight-placeholder">waiting for hot runs…</div>
        )}
        <div className="stream-replay-overlay">
          <div className="stream-replay-top">
            <span className="stream-replay-name">#1 {name} • {score == null ? "—" : fmt(score, 2)}</span>
          </div>
          <div className="stream-replay-bottom">
            <span className="stream-replay-left">x2.0 • {duration == null ? "—" : `${Number(duration).toFixed(0)}s`}</span>
            <span className="stream-replay-right">
              {deltaVs2 != null ? `Δ vs #2 ${fmt(deltaVs2, 2)}` : <SpriteIcon idx={sheetIcon("moment")} size={16} />}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

const PROGRESS_STORAGE_KEY = "mb_progress_tracker_v1";

function useProgressTrackerState() {
  const [state, setState] = useState<Record<string, boolean>>(() => {
    if (typeof window === "undefined") return {};
    try {
      const raw = window.localStorage.getItem(PROGRESS_STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw) as Record<string, boolean>;
      if (parsed && typeof parsed === "object") return parsed;
    } catch {
      // ignore storage parse errors
    }
    return {};
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(state));
    } catch {
      // ignore storage write errors
    }
  }, [state]);

  return [state, setState] as const;
}

function ProgressTrackerPanel() {
  const [goalState, setGoalState] = useProgressTrackerState();
  const [tier, setTier] = useState<GoalTier>("minor");
  const [query, setQuery] = useState("");
  const [incompleteOnly, setIncompleteOnly] = useState(false);
  const [flash, setFlash] = useState("");

  const goalsByTier = useMemo(() => {
    const minor: ProgressGoal[] = [];
    const major: ProgressGoal[] = [];
    for (const g of PROGRESS_GOALS) (g.tier === "minor" ? minor : major).push(g);
    return { minor, major };
  }, []);

  const countDone = useCallback(
    (list: ProgressGoal[]) => list.reduce((acc, g) => acc + (goalState[g.id] ? 1 : 0), 0),
    [goalState],
  );

  const minorDone = countDone(goalsByTier.minor);
  const majorDone = countDone(goalsByTier.major);
  const totalDone = minorDone + majorDone;

  const filterText = query.trim().toLowerCase();
  const filtered = (tier === "minor" ? goalsByTier.minor : goalsByTier.major).filter((g) => {
    if (incompleteOnly && goalState[g.id]) return false;
    if (!filterText) return true;
    const hay = `${g.title} ${g.detail ?? ""}`.toLowerCase();
    return hay.includes(filterText);
  });

  const onToggle = (id: string) => {
    setGoalState((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const onReset = () => {
    if (!window.confirm("Clear all progress?")) return;
    setGoalState({});
  };

  const onCopy = async () => {
    const payload = {
      updatedAt: new Date().toISOString(),
      completed: Object.keys(goalState).filter((k) => goalState[k]),
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setFlash("Copied progress JSON");
    } catch {
      setFlash("Copy failed");
    }
  };

  useEffect(() => {
    if (!flash) return;
    const t = window.setTimeout(() => setFlash(""), 1800);
    return () => window.clearTimeout(t);
  }, [flash]);

  const minorPct = Math.round((minorDone / Math.max(1, goalsByTier.minor.length)) * 100);
  const majorPct = Math.round((majorDone / Math.max(1, goalsByTier.major.length)) * 100);

  return (
    <div className="stream-card progress-card">
      <div className="stream-card-title">Progress Tracker</div>
      <div className="progress-summary">
        <div className="progress-summary-row">
          <span className="progress-pill minor">Minor</span>
          <span className="progress-count">
            {minorDone}/{goalsByTier.minor.length}
          </span>
          <div className="progress-bar">
            <div className="progress-bar-fill minor" style={{ width: `${minorPct}%` }} />
          </div>
          <span className="progress-pct">{minorPct}%</span>
        </div>
        <div className="progress-summary-row">
          <span className="progress-pill major">Major</span>
          <span className="progress-count">
            {majorDone}/{goalsByTier.major.length}
          </span>
          <div className="progress-bar">
            <div className="progress-bar-fill major" style={{ width: `${majorPct}%` }} />
          </div>
          <span className="progress-pct">{majorPct}%</span>
        </div>
        <div className="progress-total">
          <span className="badge">Total</span>
          <span className="progress-total-num">
            {totalDone}/{PROGRESS_GOALS.length}
          </span>
        </div>
      </div>

      <div className="progress-controls">
        <div className="progress-toggle">
          <button className={`progress-btn ${tier === "minor" ? "active" : ""}`} onClick={() => setTier("minor")}>
            Minor
          </button>
          <button className={`progress-btn ${tier === "major" ? "active" : ""}`} onClick={() => setTier("major")}>
            Major
          </button>
        </div>
        <label className="progress-checkbox">
          <input type="checkbox" checked={incompleteOnly} onChange={(e) => setIncompleteOnly(e.target.checked)} />
          <span>Incomplete only</span>
        </label>
      </div>

      <div className="progress-search">
        <input
          className="progress-input"
          type="text"
          placeholder="Search goals..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="progress-action" onClick={onCopy}>
          Copy
        </button>
        <button className="progress-action danger" onClick={onReset}>
          Reset
        </button>
      </div>

      <div className="progress-flash">{flash}</div>

      <div className="progress-list">
        {filtered.map((g) => (
          <label key={g.id} className={`progress-item ${goalState[g.id] ? "done" : ""}`}>
            <input type="checkbox" checked={!!goalState[g.id]} onChange={() => onToggle(g.id)} />
            <span className="progress-title">{g.title}</span>
            {g.detail ? <span className="progress-detail">{g.detail}</span> : null}
          </label>
        ))}
        {!filtered.length ? <div className="muted">no goals match</div> : null}
      </div>
    </div>
  );
}

function ReplayPiP({
  clips,
  leader,
  runnerUp,
  why,
}: {
  clips: AttractHighlights | null;
  leader: Heartbeat | null;
  runnerUp: Heartbeat | null;
  why?: string;
}) {
  const fame = clips?.fame ?? null;
  const clipUrl = fame?.clip_url ? `/api${fame.clip_url}` : null;
  const name = leader?.display_name ?? leader?.instance_id ?? fame?.instance_id ?? "—";
  const score = leader ? scoreOf(leader) : fame?.score ?? null;
  const duration = fame?.duration_s ?? null;
  const runnerScore = runnerUp ? scoreOf(runnerUp) : null;
  const deltaVs2 = score != null && runnerScore != null ? score - runnerScore : null;
  const onVideoReady = (e: SyntheticEvent<HTMLVideoElement>) => {
    try {
      e.currentTarget.playbackRate = 2.0;
    } catch {
      // ignore
    }
  };
  if (!clipUrl) return null;
  return (
    <div className="replay-pip" aria-label="replay picture in picture">
      <div className="replay-pip-head">
        <span className="badge">NEW #1</span>
        <span className="replay-pip-title">
          {name} • {score == null ? "—" : fmt(score, 2)}
        </span>
        <span className="muted">{duration == null ? "—" : `${Number(duration).toFixed(0)}s`}</span>
      </div>
      <div className="replay-pip-media">
        <video className="replay-pip-video" src={clipUrl} autoPlay muted playsInline loop onLoadedMetadata={onVideoReady} />
        <div className="replay-pip-overlay">
          <span className="replay-pip-why">{why ?? (deltaVs2 != null ? `Δ vs #2 ${fmt(deltaVs2, 2)}` : "new leader replay")}</span>
        </div>
      </div>
    </div>
  );
}

export default function Stream() {
  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 2000 });
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 750 });
  const featuredQ = useQuery({ queryKey: ["featured"], queryFn: fetchFeatured, refetchInterval: 750 });
  const fuseQ = useQuery({ queryKey: ["fuse"], queryFn: () => fetchTimelineFuse(2000), refetchInterval: 2500 });
  const hofQ = useQuery({ queryKey: ["hofTop"], queryFn: () => fetchHofTop(5, "score"), refetchInterval: 3500 });
  const timelineQ = useQuery({ queryKey: ["timeline"], queryFn: () => fetchTimeline(120), refetchInterval: 3500 });
  const bettingQ = useQuery({ queryKey: ["betting"], queryFn: fetchBettingState, refetchInterval: 2500 });
  const pollQ = useQuery({ queryKey: ["poll"], queryFn: fetchPollState, refetchInterval: 2500 });
  const attractQ = useQuery({ queryKey: ["attractHighlights"], queryFn: fetchAttractHighlights, refetchInterval: 4000 });
  const histQ = useQuery({ queryKey: ["historicLeaderboard"], queryFn: () => fetchHistoricLeaderboard(200, "best_score"), refetchInterval: 5000 });
  const events = useEventStream(250);

  const canvasRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);

  const [railTab, setRailTab] = useState<RailTab>("moments");
  const [railAuto, setRailAuto] = useState(true);
  const [replayQueue, setReplayQueue] = useState<Array<{ clipUrl: string; ts: number; label: string }>>([]);
  const lastReplayKeyRef = useRef<string>("");

  const [directorOn, setDirectorOn] = useState(true);
  const [directorId, setDirectorId] = useState<string | null>(null);
  const [directorUntil, setDirectorUntil] = useState(0);
  const directorLastSwitchRef = useRef(0);
  const loc = useLocation();
  const [layoutMode, setLayoutMode] = useState<"broadcast" | "dense">("broadcast");
  const qs = useMemo(() => new URLSearchParams(loc.search), [loc.search]);
  const debugParamOn = useMemo(() => qs.get("debug") === "1", [qs]);
  const safeOn = useMemo(() => {
    const safe = qs.get("safe") === "1";
    if (!safe) return false;
    return !import.meta.env.PROD || debugParamOn;
  }, [qs, debugParamOn]);
  const [lowVision, setLowVision] = useState(false);
  useEffect(() => {
    const visionParam = qs.get("vision");
    if (visionParam === "1") {
      try {
        window.localStorage.setItem("mb:vision", "1");
      } catch {}
      setLowVision(true);
      return;
    }
    if (visionParam === "0") {
      try {
        window.localStorage.setItem("mb:vision", "0");
      } catch {}
      setLowVision(false);
      return;
    }
    try {
      setLowVision(window.localStorage.getItem("mb:vision") === "1");
    } catch {
      setLowVision(false);
    }
  }, [qs]);
  const setQueryParams = useCallback((patch: Record<string, string | null>) => {
    const next = new URLSearchParams(window.location.search);
    for (const [k, v] of Object.entries(patch)) {
      if (v == null) next.delete(k);
      else next.set(k, v);
    }
    const s = next.toString();
    window.location.search = s ? `?${s}` : "";
  }, []);

  const [tickerIdx, setTickerIdx] = useState(0);
  const tickerLines = useMemo(() => {
    const lines = [
      "GOAL: survive longer, score higher. Watch for clutch moments + overtake bursts.",
      "WINNERS: score is the main rank; step is progress/pace. Historic best/peak are persistent.",
    ];
    if (debugParamOn) {
      lines.push("RACE BAR: each segment is a run; WR marker shows record pace; NOW is current time.");
    }
    return lines;
  }, [debugParamOn]);

  const [momentCard, setMomentCard] = useState<{ until: number; title: string; subtitle?: string; iid?: string } | null>(null);
  const safeX = Math.round(HUD_W * 0.05);
  const safeY = Math.round(HUD_H * 0.05);
  const actionSafeX = Math.round(HUD_W * 0.035);
  const actionSafeY = Math.round(HUD_H * 0.035);
  const scopeRef = useRef<HTMLDivElement | null>(null);
  const [uiScale, setUiScale] = useState(1);
  const [uiTier, setUiTier] = useState("720");

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return;
    const ro = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect();
      const s = Math.min(r.width / HUD_W, r.height / HUD_H);
      stage.style.transform = `scale(${s})`;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const scope = scopeRef.current;
    if (!scope) return;
    let raf = 0;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        setUiScale(computeUiScale(width, height));
        setUiTier(computeTier(width));
      });
    });
    ro.observe(scope);
    return () => {
      ro.disconnect();
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const workers = useMemo(() => {
    const arr = Object.values(workersQ.data ?? {});
    arr.sort((a, b) => scoreOf(b) - scoreOf(a));
    return arr;
  }, [workersQ.data]);
  const pipewireDown = useMemo(() => {
    return workers.filter(
      (w) =>
        (w as any)?.pipewire_ok === false ||
        (w as any)?.pipewire_session_ok === false ||
        (w as any)?.pipewire_node_ok === false ||
        String((w as any)?.status ?? "").includes("no_pipewire"),
    );
  }, [workers]);

  const memeStats = useMemo(() => {
    const now = Date.now() / 1000;
    const recent = events.filter((e) => (e.ts ?? 0) >= now - 120);
    const bonks = recent.filter((e) => String(e.event_type ?? "").toLowerCase().includes("bonk") || String(e.message ?? "").toLowerCase().includes("bonk"));
    const deaths = recent.filter((e) => String(e.event_type ?? "").toLowerCase().includes("death") || String(e.message ?? "").toLowerCase().includes("death"));
    const borgars = workers.reduce((sum, w) => sum + Number((w as any)?.borgar_count ?? 0), 0);
    const hype = workers.reduce((sum, w) => sum + Number((w as any)?.hype_score ?? 0), 0);
    const hypeAvg = workers.length ? hype / workers.length : 0;
    const confVals = workers.map((w) => Number((w as any)?.bonk_confidence)).filter((v) => Number.isFinite(v));
    const doomVals = workers.map((w) => Number((w as any)?.menu_doom_spiral)).filter((v) => Number.isFinite(v));
    const chatVals = workers.map((w) => Number((w as any)?.chat_influence)).filter((v) => Number.isFinite(v));
    const conf = confVals.length ? confVals.reduce((a, b) => a + b, 0) / confVals.length : 0;
    const doom = doomVals.length ? doomVals.reduce((a, b) => a + b, 0) / doomVals.length : 0;
    const chat = chatVals.length ? chatVals.reduce((a, b) => a + b, 0) / chatVals.length : 0;
    return {
      bonksPerMin: bonks.length / 2,
      deaths2m: deaths.length,
      borgars,
      hype: Math.round(hypeAvg * 100),
      bonkConfidence: Math.round(conf * 100),
      menuDoom: Math.round(doom),
      chatInfluence: Math.round(chat * 100),
    };
  }, [events, workers]);

  useEffect(() => {
    const latest = events[events.length - 1];
    if (!latest) return;
    const payload = (latest as any)?.payload ?? {};
    const rawClip = payload?.clip_url ?? payload?.clipUrl ?? null;
    if (!rawClip) return;
    const clipUrl = String(rawClip).startsWith("/api") ? String(rawClip) : `/api${rawClip}`;
    const key = `${clipUrl}|${latest.event_id}`;
    if (key === lastReplayKeyRef.current) return;
    lastReplayKeyRef.current = key;
    const label = latest.message || latest.event_type || "highlight";
    setReplayQueue((prev) => [{ clipUrl, ts: latest.ts ?? Date.now() / 1000, label }, ...prev].slice(0, 3));
  }, [events]);

  const workersById = useMemo(() => {
    const m: Record<string, Heartbeat> = {};
    for (const w of Object.values(workersQ.data ?? {})) m[String(w.instance_id)] = w;
    return m;
  }, [workersQ.data]);
  const workersByIdRef = useRef<Record<string, Heartbeat>>({});
  useEffect(() => {
    workersByIdRef.current = workersById;
  }, [workersById]);

  const slots = featuredQ.data?.slots ?? {};
  const hype0 = slots["hype0"] ? workersById[String(slots["hype0"])] ?? null : null;
  const hype1 = slots["hype1"] ? workersById[String(slots["hype1"])] ?? null : null;
  const hype2 = slots["hype2"] ? workersById[String(slots["hype2"])] ?? null : null;
  const shame0 = slots["shame0"] ? workersById[String(slots["shame0"])] ?? null : null;
  const shame1 = slots["shame1"] ? workersById[String(slots["shame1"])] ?? null : null;

  const [focusId, setFocusId] = useState<string | null>(null);
  const focus = useMemo(() => {
    if (focusId) return workersById[String(focusId)] ?? null;
    if (directorOn && directorId && directorUntil > Date.now()) return workersById[String(directorId)] ?? null;
    return hype0 ?? hype1 ?? hype2 ?? shame0 ?? shame1 ?? null;
  }, [focusId, directorOn, directorId, directorUntil, workersById, hype0, hype1, hype2, shame0, shame1]);

  const manualFocusActive = Boolean(focusId && focus);
  const directorActive = Boolean(!focusId && directorOn && directorId && directorUntil > Date.now() && focus);
  const applyDirectorCut = useCallback((iid: string, holdMs: number) => {
    const now = Date.now();
    setDirectorId(iid);
    setDirectorUntil(now + holdMs);
    directorLastSwitchRef.current = now;
  }, []);

  const interesting = useMemo(() => pickInterestingEvents(events, 6), [events]);

  const leaderTop = useMemo(() => workers.slice(0, 10), [workers]);
  const leaderRows = useMemo(() => {
    return leaderTop.length
      ? leaderTop
      : ((histQ.data ?? []) as HistoricLeaderboardEntry[]).slice(0, 10).map((h) => ({
          instance_id: h.instance_id,
          display_name: h.display_name ?? h.instance_id,
          policy_name: h.policy_name ?? null,
          step: h.last_step ?? 0,
          steam_score: h.last_score ?? 0,
        } as any));
  }, [leaderTop, histQ.data]);
  const histById = useMemo(() => {
    const m: Record<string, HistoricLeaderboardEntry> = {};
    for (const r of (histQ.data ?? []) as HistoricLeaderboardEntry[]) {
      if (!r?.instance_id) continue;
      m[String(r.instance_id)] = r;
    }
    return m;
  }, [histQ.data]);

  const overrunAny = Boolean(hype0?.overrun || hype1?.overrun || hype2?.overrun || shame0?.overrun || shame1?.overrun);

  const fuseViz = useMemo(() => fuseBars(fuseQ.data as any), [fuseQ.data]);

  // Rank-change bursts: tiered overtakes + batch mode throttle.
  const prevRanksRef = useRef<Record<string, number>>({});
  const [rankBursts, setRankBursts] = useState<Record<string, { delta: number; until: number; size: "small" | "big" }>>({});
  const [heatMap, setHeatMap] = useState<Record<string, { until: number; kind: "up" | "down" }>>({});
  const [swapUntil, setSwapUntil] = useState(0);
  const [batchModeUntil, setBatchModeUntil] = useState(0);
  const [overtakeActiveUntil, setOvertakeActiveUntil] = useState(0);
  const [overtakeTicker, setOvertakeTicker] = useState<Array<{ iid: string; delta: number; ts: number }>>([]);
  const overtakeWindowRef = useRef<number[]>([]);
  useEffect(() => {
    if (!leaderTop.length) return;
    const now = Date.now();
    const prev = prevRanksRef.current;
    const next: Record<string, number> = {};
    const overtakes: Array<{ iid: string; delta: number; rank: number }> = [];
    for (let i = 0; i < leaderTop.length; i++) {
      const iid = String((leaderTop[i] as any)?.instance_id ?? "");
      if (!iid) continue;
      const rank = i + 1;
      next[iid] = rank;
      const prevRank = prev[iid];
      if (prevRank && prevRank > rank) {
        const delta = prevRank - rank;
        overtakes.push({ iid, delta, rank });
      }
    }
    if (overtakes.length) {
      setSwapUntil(now + 900);
      const heat: Array<{ iid: string; kind: "up" | "down" }> = [];
      const window = overtakeWindowRef.current.filter((t) => t > now - 4000);
      window.push(...overtakes.map(() => now));
      overtakeWindowRef.current = window;
      if (window.length >= 3) setBatchModeUntil(now + 10000);
      const batchActive = batchModeUntil > now;
      const maxDelta = Math.max(...overtakes.map((o) => o.delta));
      const winner = overtakes.find((o) => o.delta === maxDelta) ?? overtakes[0];
      for (const o of overtakes) {
        heat.push({ iid: o.iid, kind: "up" });
        const victim = Object.keys(prev).find((id) => prev[id] === o.rank && next[id] !== o.rank);
        if (victim) heat.push({ iid: victim, kind: "down" });
        const isBig = o.delta >= 2 || o.rank === 1;
        const size: "small" | "big" = isBig ? "big" : "small";
        const duration = isBig ? 5600 : 3200;
        if (batchActive && o.iid !== winner.iid) {
          setOvertakeTicker((cur) => [...cur.slice(-5), { iid: o.iid, delta: o.delta, ts: now }]);
          continue;
        }
        setRankBursts((cur) => ({ ...cur, [o.iid]: { delta: o.delta, until: now + duration, size } }));
        setOvertakeActiveUntil((cur) => Math.max(cur, now + duration));
        setOvertakeTicker((cur) => [...cur.slice(-5), { iid: o.iid, delta: o.delta, ts: now }]);
      }
      if (heat.length) {
        setHeatMap((cur) => {
          const pruned: typeof cur = {};
          for (const [k, v] of Object.entries(cur)) if (v.until > now) pruned[k] = v;
          for (const h of heat) pruned[h.iid] = { until: now + 1800, kind: h.kind };
          return pruned;
        });
      }
    }
    prevRanksRef.current = next;
  }, [leaderTop, batchModeUntil]);

  useEffect(() => {
    if (!overtakeTicker.length) return;
    const t = window.setTimeout(() => {
      const cutoff = Date.now() - 12000;
      setOvertakeTicker((cur) => cur.filter((x) => x.ts >= cutoff));
    }, 1200);
    return () => window.clearTimeout(t);
  }, [overtakeTicker]);

  const lastHighlightKeyRef = useRef<string>("");
  const [highlightPulseUntil, setHighlightPulseUntil] = useState(0);
  const [replayPromoteUntil, setReplayPromoteUntil] = useState(0);
  useEffect(() => {
    const fame = (attractQ.data as AttractHighlights | null)?.fame ?? null;
    const key = fame?.clip_url ? `${fame.clip_url}|${fame.instance_id ?? ""}|${fame.score ?? ""}` : "";
    if (!key || key === lastHighlightKeyRef.current) return;
    lastHighlightKeyRef.current = key;
    setHighlightPulseUntil(Date.now() + 1600);
    setReplayPromoteUntil(Date.now() + 9000);
  }, [attractQ.data]);

  useEffect(() => {
    if (!highlightPulseUntil) return;
    const timeout = Math.max(0, highlightPulseUntil - Date.now());
    const t = window.setTimeout(() => setHighlightPulseUntil(0), timeout);
    return () => window.clearTimeout(t);
  }, [highlightPulseUntil]);

  const dominantMode = useMemo(() => {
    if (!hype0) return false;
    const s0 = scoreOf(hype0);
    const others = [hype1, hype2, shame0, shame1].filter(Boolean) as Heartbeat[];
    if (!others.length) return true;
    const bestOther = Math.max(...others.map((w) => scoreOf(w)));
    const delta = s0 - bestOther;
    const ratio = bestOther !== 0 ? s0 / bestOther : s0 > 0 ? Number.POSITIVE_INFINITY : 0;
    // "Absolutely dominating" heuristic (tuned to be conservative).
    // Requires meaningful absolute lead AND a clear relative lead.
    return s0 >= 5 && delta >= 2 && ratio >= 1.35;
  }, [hype0, hype1, hype2, shame0, shame1]);

  // Announce dominance once per "dominance streak" (old arcade-style callout).
  const prevDominantRef = useRef(false);
  useEffect(() => {
    const prev = prevDominantRef.current;
    if (!prev && dominantMode) {
      const now = Date.now();
      setPop((cur) => {
        // Don't stomp an active callout (prevents spam/overwrites).
        if (cur && cur.until > now) return cur;
        return { text: "DOMINATING!", until: now + 1300 };
      });
    }
    prevDominantRef.current = dominantMode;
  }, [dominantMode]);

  const shameList = useMemo(() => {
    const segs = (fuseQ.data as any)?.segments ?? [];
    const rows = segs
      .map((s: any) => ({
        instance_id: String(s.instance_id ?? "—"),
        duration_s: Math.max(0, Number(s.duration_s ?? 0)),
        score: Number(s.score ?? 0),
        ts: Number(s.ts ?? 0),
      }))
      .filter((x: any) => x.duration_s > 0 && x.instance_id !== "—");
    rows.sort((a: any, b: any) => a.duration_s - b.duration_s);
    return rows.slice(0, 6);
  }, [fuseQ.data]);

  const attractMode = useMemo(() => {
    // Only show attract mode when *no workers* are connected. If workers exist
    // but feeds are down, keep the tiles visible so "NO GPU FEED" is explicit.
    return workers.length === 0;
  }, [workers.length]);

  // Event-driven juice / FX.
  const [impactUntil, setImpactUntil] = useState(0);
  const [invertUntil, setInvertUntil] = useState(0);
  const [linesUntil, setLinesUntil] = useState(0);
  const [shakeUntil, setShakeUntil] = useState(0);
  const [pop, setPop] = useState<{ text: string; until: number } | null>(null);
  const [impactMap, setImpactMap] = useState<Record<string, number>>({});

  // Ticker: rotate every ~7s. Pauses when manual focus is active.
  useEffect(() => {
    if (manualFocusActive) return;
    const t = window.setInterval(() => setTickerIdx((i) => (i + 1) % tickerLines.length), 7000);
    return () => window.clearInterval(t);
  }, [manualFocusActive, tickerLines.length]);

  useEffect(() => {
    if (!railAuto) return;
    const order: RailTab[] = ["moments", "community", "bucks", "poll", "progress"];
    const t = window.setInterval(() => {
      setRailTab((cur) => {
        const idx = order.indexOf(cur);
        return order[(idx >= 0 ? idx : 0) + 1 < order.length ? idx + 1 : 0];
      });
    }, 9500);
    return () => window.clearInterval(t);
  }, [railAuto]);

  const prevScoreRef = useRef<Record<string, number>>({});
  const scoreHistoryRef = useRef<Record<string, Array<{ ts: number; v: number }>>>({});
  const stepHistoryRef = useRef<Record<string, Array<{ ts: number; v: number }>>>({});
  const lastScoreChangeRef = useRef<Record<string, number>>({});
  useEffect(() => {
    const now = Date.now();
    const next: Record<string, number> = {};
    for (const w of workers) {
      const iid = String((w as any)?.instance_id ?? "");
      if (!iid) continue;
      const score = scoreOf(w);
      const prev = prevScoreRef.current[iid];
      if (prev == null || score !== prev) lastScoreChangeRef.current[iid] = now;
      next[iid] = score;
      const step = Number(w?.step ?? 0);
      const scoreHist = scoreHistoryRef.current[iid] ?? [];
      const stepHist = stepHistoryRef.current[iid] ?? [];
      scoreHist.push({ ts: now, v: score });
      stepHist.push({ ts: now, v: step });
      scoreHistoryRef.current[iid] = trimHistory(scoreHist, 90000, now);
      stepHistoryRef.current[iid] = trimHistory(stepHist, 90000, now);
    }
    prevScoreRef.current = next;
  }, [workers]);

  const sparkSeriesRef = useRef<Record<string, number[]>>({});
  const [sparkTick, setSparkTick] = useState(0);
  const chipStateRef = useRef<Record<string, Chip[]>>({});
  const chipCooldownRef = useRef<Record<string, number>>({});
  const lastChipEventTsRef = useRef(0);
  const [chipTick, setChipTick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => {
      const now = Date.now();
      for (const w of workersById ? Object.values(workersById) : []) {
        const iid = String((w as any)?.instance_id ?? "");
        if (!iid) continue;
        const hist = scoreHistoryRef.current[iid] ?? [];
        const d30 = deltaInWindow(hist, 30000, now);
        const series = sparkSeriesRef.current[iid] ?? [];
        series.push(Number.isFinite(d30 as any) ? Number(d30) : 0);
        if (series.length > 36) series.splice(0, series.length - 36);
        sparkSeriesRef.current[iid] = series;
      }
      setSparkTick((v) => v + 1);
    }, 2000);
    return () => window.clearInterval(t);
  }, [workersById]);

  useEffect(() => {
    if (!events.length) return;
    const now = Date.now();
    const batchActive = batchModeUntil > now;
    const highOnly = batchActive;
    const allowHigh = new Set(["RECORD", "PB", "#1"]);

    const toChip = (e: Event): Chip | null => {
      const t = String(e.event_type ?? "");
      const upper = t.toUpperCase();
      const map: Record<string, { label: string; icon: IconKey; priority: number }> = {
        RECORD: { label: "RECORD", icon: "record_wr", priority: 100 },
        NEWRECORD: { label: "RECORD", icon: "record_wr", priority: 100 },
        PB: { label: "PB", icon: "rarity_star", priority: 80 },
        PERSONALBEST: { label: "PB", icon: "rarity_star", priority: 80 },
        OVERTAKE: { label: "OVERTAKE", icon: "rank_up", priority: 70 },
        RANKSWAP: { label: "OVERTAKE", icon: "rank_up", priority: 70 },
        BOSS: { label: "BOSS", icon: "boss", priority: 60 },
        BIGBONK: { label: "BIG BONK", icon: "big_bonk", priority: 55 },
        OVERCRIT: { label: "BIG BONK", icon: "big_bonk", priority: 55 },
        TOME: { label: "TOME+", icon: "tome_mastery", priority: 50 },
        EUREKA: { label: "TOME+", icon: "tome_mastery", priority: 50 },
        CLUTCH: { label: "CLUTCH", icon: "moment", priority: 45 },
        DISASTER: { label: "DISASTER", icon: "warning", priority: 60 },
        WEIRDBUILD: { label: "WEIRD BUILD", icon: "loot_chest", priority: 45 },
        SKIP: { label: "SKIP", icon: "time", priority: 40 },
      };
      const key = map[upper] ? upper : upper.includes("OVERTAKE") ? "OVERTAKE" : upper.includes("BOSS") ? "BOSS" : "";
      const def = key ? map[key] : null;
      if (!def) return null;
      if (highOnly && !allowHigh.has(def.label)) return null;
      return { ...def, ts: Number(e.ts ?? now), value: undefined };
    };

    let updated = false;
    for (const e of events) {
      if (!e?.instance_id) continue;
      if (Number(e.ts ?? 0) <= lastChipEventTsRef.current) continue;
      const iid = String(e.instance_id);
      const chip = toChip(e);
      if (!chip) continue;
      const lastTs = chipCooldownRef.current[iid] ?? 0;
      const existing = (chipStateRef.current[iid] ?? []).filter((c) => c.ts + CHIP_TTL_MS > now);
      if (now - lastTs < CHIP_COOLDOWN_MS) {
        const top = existing[0];
        if (!top || chip.priority > top.priority) {
          chipStateRef.current[iid] = [chip];
          updated = true;
        }
        continue;
      }
      chipCooldownRef.current[iid] = now;
      chipStateRef.current[iid] = [chip, ...existing].slice(0, 2);
      updated = true;
    }
    lastChipEventTsRef.current = Number(events[events.length - 1].ts ?? now);
    if (updated) setChipTick((v) => v + 1);
  }, [events, batchModeUntil]);

  useEffect(() => {
    if (!events.length) return;
    const last = events[events.length - 1];
    const now = Date.now();
    const et = String(last.event_type ?? "");
    const iid = String(last.instance_id ?? "");
    if (iid) {
      setImpactMap((prev) => ({ ...prev, [iid]: now + 1800 }));
    }
    if (et === "Overcrit" || et === "NewMaxHit") {
      setImpactUntil(now + 140);
      setLinesUntil(now + 520);
      setPop({ text: et === "Overcrit" ? "OVERRIDE CRIT!" : "NEW MAX HIT!", until: now + 950 });
    } else if (et === "LootDrop" || et === "BountyClaimed") {
      setInvertUntil(now + 160);
      setPop({ text: "LOOT!", until: now + 900 });
    } else if (et === "WeirdBuild") {
      setInvertUntil(now + 220);
      setPop({ text: "WEIRD BUILD!", until: now + 1100 });
    } else if (et === "Disaster") {
      setShakeUntil(now + 520);
      setLinesUntil(now + 620);
      setPop({ text: "DISASTER!", until: now + 1200 });
    } else if (et === "EpisodeEnd") {
      setShakeUntil(now + 360);
    }

    // Director: chase interesting moments (without stealing manual focus).
    if (!focusId && directorOn && iid) {
      const chase = new Set(["Overcrit", "NewMaxHit", "LootDrop", "BountyClaimed", "Eureka", "OverrunStart", "ChatSpike", "Heal", "Disaster", "WeirdBuild", "Clutch"]);
      if (chase.has(et)) {
        applyDirectorCut(iid, 8000);
      }
    }

    // Moment card: short “what just happened” broadcast cue.
    if (iid && et && et !== "Telemetry") {
      const show = new Set(["Overcrit", "NewMaxHit", "LootDrop", "BountyClaimed", "Eureka", "OverrunStart", "Disaster", "WeirdBuild", "Clutch"]);
      if (show.has(et)) {
        const cur = workersByIdRef.current[iid];
        const name = (cur?.display_name ?? iid) as string;
        const scoreNow = scoreOf(cur);
        const scorePrev = prevScoreRef.current[iid];
        const delta = Number.isFinite(scoreNow) && Number.isFinite(scorePrev) ? scoreNow - scorePrev : null;
        const subtitle = delta != null && Math.abs(delta) > 0.01 ? `${name} · Δscore ${fmt(delta, 2)}` : `${name}`;
        setMomentCard({ until: now + 4200, title: et.toUpperCase(), subtitle, iid });
      }
    }
  }, [events, focusId, directorOn, applyDirectorCut]);

  const nowMs = Date.now();
  const impactOn = impactUntil > nowMs;
  const invertOn = invertUntil > nowMs;
  const linesOn = linesUntil > nowMs;
  const shakeOn = shakeUntil > nowMs;

  const rankUpFor = (iid: string | null | undefined) => {
    if (!iid) return null;
    const b = rankBursts[String(iid)];
    if (!b || !(b.until > nowMs) || !(b.delta > 0)) return null;
    return b.delta;
  };

  const rankBurstFor = (iid: string | null | undefined) => {
    if (!iid) return null;
    const b = rankBursts[String(iid)];
    if (!b || !(b.until > nowMs) || !(b.delta > 0)) return null;
    return b;
  };

  const betting = bettingQ.data as any;
  const poll = pollQ.data as any;
  const timeline = timelineQ.data as TimelineState | undefined;

  const focusTitle = focus?.display_name ?? focus?.instance_id ?? "—";

  const scoreDelta30sById = useMemo(() => {
    const now = Date.now();
    const m: Record<string, number | null> = {};
    for (const w of workers) {
      const iid = String((w as any)?.instance_id ?? "");
      if (!iid) continue;
      const hist = scoreHistoryRef.current[iid] ?? [];
      m[iid] = deltaInWindow(hist, 30000, now);
    }
    return m;
  }, [workers]);

  const stepRateById = useMemo(() => {
    const now = Date.now();
    const m: Record<string, number | null> = {};
    for (const w of workers) {
      const iid = String((w as any)?.instance_id ?? "");
      if (!iid) continue;
      const hist = stepHistoryRef.current[iid] ?? [];
      m[iid] = ratePerMinute(hist, 30000, now);
    }
    return m;
  }, [workers]);

  const resetsById = useMemo(() => {
    const now = Date.now();
    const windowMs = 5 * 60 * 1000;
    const m: Record<string, number> = {};
    for (const e of events) {
      if (!e?.instance_id || e.event_type !== "EpisodeEnd") continue;
      if (e.ts < now - windowMs) continue;
      const iid = String(e.instance_id);
      m[iid] = (m[iid] ?? 0) + 1;
    }
    return m;
  }, [events]);

  const hotRun = useMemo(() => {
    let best: { iid: string; delta: number } | null = null;
    for (const w of workers) {
      const iid = String((w as any)?.instance_id ?? "");
      if (!iid) continue;
      const d = scoreDelta30sById[iid];
      if (d == null || !(d > 0)) continue;
      if (!best || d > best.delta) best = { iid, delta: d };
    }
    return best;
  }, [workers, scoreDelta30sById]);

  const pbChaser = useMemo(() => {
    let best: { iid: string; ratio: number } | null = null;
    for (const w of workers) {
      const iid = String((w as any)?.instance_id ?? "");
      if (!iid) continue;
      const hist = histById[iid];
      const bestScore = Number(hist?.best_score ?? 0);
      const cur = scoreOf(w);
      if (!(bestScore > 0) || !(cur > 0)) continue;
      const ratio = cur / bestScore;
      if (ratio >= 1) continue;
      if (!best || ratio > best.ratio) best = { iid, ratio };
    }
    return best;
  }, [workers, histById]);

  const stageMosaic = useMemo(() => {
    const out: Array<{ hb: Heartbeat; label: string }> = [];
    const seen = new Set<string>();
    const push = (hb: Heartbeat | null, label: string) => {
      if (!hb) return;
      const iid = String((hb as any)?.instance_id ?? "");
      if (!iid || seen.has(iid)) return;
      if (focus?.instance_id && String(focus.instance_id) === iid) return;
      seen.add(iid);
      out.push({ hb, label });
    };
    push(leaderTop[0] ?? null, "LEADER");
    if (hotRun?.iid) push(workersById[hotRun.iid] ?? null, "HOT RUN");
    if (pbChaser?.iid) push(workersById[pbChaser.iid] ?? null, "PB CHASE");
    push(hype1 ?? null, "HYPE #2");
    push(hype2 ?? null, "HYPE #3");
    push(shame0 ?? null, "SHAME #1");
    push(shame1 ?? null, "SHAME #2");
    return out.slice(0, 4);
  }, [leaderTop, hotRun, pbChaser, workersById, focus, hype1, hype2, shame0, shame1]);

  const leaderId = leaderTop[0]?.instance_id ? String(leaderTop[0].instance_id) : "";
  const prevLeaderRef = useRef<string>("");
  const [hardOverride, setHardOverride] = useState<{ iid: string; until: number } | null>(null);
  useEffect(() => {
    if (!leaderId) return;
    if (prevLeaderRef.current && prevLeaderRef.current !== leaderId) {
      setHardOverride({ iid: leaderId, until: Date.now() + 2000 });
    }
    prevLeaderRef.current = leaderId;
  }, [leaderId]);

  const pendingCutRef = useRef<string | null>(null);

  useEffect(() => {
    if (!directorOn || focusId) return;
    const tick = window.setInterval(() => {
      const now = Date.now();
      if (hardOverride && hardOverride.until > now) {
        if (overtakeActiveUntil > now) {
          pendingCutRef.current = hardOverride.iid;
          return;
        }
        applyDirectorCut(hardOverride.iid, 12000);
        return;
      }

      const w1 = 2.2;
      const w2 = 1.6;
      const w3 = 1.1;
      const w4 = 1.8;
      const w5 = 1.4;
      const w6 = 3.0;
      let best: { iid: string; score: number } | null = null;
      for (const w of workers) {
        const iid = String((w as any)?.instance_id ?? "");
        if (!iid) continue;
        const combat = Number(w?.enemy_count ?? 0) > 0 || Boolean(w?.overrun);
        const delta = Number(scoreDelta30sById[iid] ?? 0);
        const novelty = Number((w as any)?.luck_drop_count ?? 0) > 0 ? 1 : 0;
        const danger = Math.max(0, Math.min(1, Number(w?.danger_level ?? (w?.survival_prob != null ? 1 - Number(w.survival_prob) : 0)) || 0));
        const recordChase =
          leaderTop.length > 1 && iid !== leaderId ? Math.max(0, 1 - scoreOf(w) / Math.max(1, scoreOf(leaderTop[0]))) : 0;
        const status = String(w?.status ?? "").toLowerCase();
        const menu = status.includes("menu") || status.includes("stuck") ? 1 : 0;
        const score = w1 * (combat ? 1 : 0) + w2 * delta + w3 * novelty + w4 * danger + w5 * recordChase - w6 * menu;
        if (!best || score > best.score) best = { iid, score };
      }
      if (!best) return;

      const minDwellMs = 12000;
      const softOverride =
        Boolean(workersById[best.iid]?.overrun ?? false) || Number(workersById[best.iid]?.incoming_dps ?? 0) > 0;
      const lastSwitch = directorLastSwitchRef.current || 0;
      const canCut = now - lastSwitch > minDwellMs || softOverride;
      if (!canCut) return;
      if (overtakeActiveUntil > now) {
        pendingCutRef.current = best.iid;
        return;
      }
      applyDirectorCut(best.iid, 12000);
    }, 1000);
    return () => window.clearInterval(tick);
  }, [directorOn, focusId, workers, workersById, leaderTop, leaderId, scoreDelta30sById, hardOverride, overtakeActiveUntil, applyDirectorCut]);

  useEffect(() => {
    const now = Date.now();
    if (overtakeActiveUntil > now) return;
    if (!directorOn || focusId) return;
    if (!pendingCutRef.current) return;
    applyDirectorCut(pendingCutRef.current, 12000);
    pendingCutRef.current = null;
  }, [overtakeActiveUntil, directorOn, focusId, applyDirectorCut]);

  return (
    <RouteScope ref={scopeRef} className="route-scope stream-scope" uiScale={uiScale} tier={uiTier}>
      <div
        className="hud"
        style={
          {
            ["--mb-sprite-url" as any]: `url(${iconSheetUrl})`,
            ["--hud-w" as any]: `${HUD_W}px`,
            ["--hud-h" as any]: `${HUD_H}px`,
            ["--safe-x-base" as any]: `${safeX}px`,
            ["--safe-y-base" as any]: `${safeY}px`,
            ["--action-safe-x-base" as any]: `${actionSafeX}px`,
            ["--action-safe-y-base" as any]: `${actionSafeY}px`,
          } as any
        }
      >
      <div className="hud-fx">
        <div className="hud-chroma" />
        <div className="hud-vignette" />
        <div className="hud-noise" />
        <div className="hud-scanlines" />
      </div>
      <div ref={canvasRef} className="hud-canvas">
        <div ref={stageRef} className={`hud-stage ${overrunAny || shakeOn ? "juice-shake" : ""} ${overrunAny ? "stream-root-overrun" : ""}`}>
          {safeOn ? <SafeGuides /> : null}
          {impactOn ? <div className="impact-frame" /> : null}
          {invertOn ? <div className="juice-invert" /> : null}
          {linesOn ? <div className="juice-lines" /> : null}
          {pop && pop.until > nowMs ? <div className="bonk-pop">{pop.text}</div> : null}
          {momentCard && momentCard.until > nowMs ? (
            <div className="moment-card" aria-label="moment card">
              <div className="moment-head">
                <span className="badge">MOMENT</span>
                <span className="moment-title">{momentCard.title}</span>
              </div>
              <div className="moment-sub muted">{momentCard.subtitle ?? ""}</div>
            </div>
          ) : null}
          <div className={`stream-root layout-${layoutMode} ${lowVision ? "vision-on" : ""}`}>
            <div className="stream-layout">
                <div className="stream-top">
                <div>
                  <div className="stream-title">MetaBonk • Spectator Cam</div>
                  <div className="muted" style={{ marginTop: 2 }}>
                    {workers.length} instances • featured = 3 hype + 2 shame
                  </div>
                  <div className="stream-ticker" aria-label="what am i watching">
                    <span className="stream-ticker-dot" />
                    <span className="stream-ticker-text">{tickerLines[tickerIdx] ?? ""}</span>
                  </div>
                  {pipewireDown.length ? (
                    <div className="stream-alert stream-alert-danger" role="alert">
                      PipeWire missing on {pipewireDown.length} instance{pipewireDown.length === 1 ? "" : "s"} • audio graph detached
                    </div>
                  ) : null}
                </div>
                <div className="stream-kpis">
                  <div className="stream-kpi">
                    <div className="muted">FOCUS</div>
                    <div className="stream-kpi-value">{focusTitle}</div>
                  </div>
                  <div className="stream-kpi">
                    <div className="muted">SCORE</div>
                    <div className="stream-kpi-value numeric">{scoreDisplay(focus, 2)}</div>
                  </div>
                  <div className="stream-kpi" style={{ minWidth: 150 }}>
                    <div className="row-between" style={{ alignItems: "baseline" }}>
                      <div className="muted">DIRECTOR</div>
                      <span className={`status-pill ${directorOn ? "on" : "off"}`}>{directorOn ? "ON" : "OFF"}</span>
                    </div>
                    <div className="muted" style={{ marginTop: 2 }}>
                      {manualFocusActive ? "manual focus" : directorActive ? "chasing moments" : "idle"}
                    </div>
                  </div>
                  <div className="stream-kpi" style={{ minWidth: 150 }}>
                    <div className="row-between" style={{ alignItems: "baseline" }}>
                      <div className="muted">LAYOUT</div>
                      <span className={`status-pill ${layoutMode === "broadcast" ? "on" : "off"}`}>
                        {layoutMode === "broadcast" ? "720P SAFE" : "DENSE"}
                      </span>
                    </div>
                    <div className="muted" style={{ marginTop: 2 }}>
                      {layoutMode === "broadcast" ? "720p safe" : "dev compact"}
                    </div>
                  </div>
                </div>
              </div>
              <div className="stream-meme">
                <div className="meme-pill">Bonks/min {memeStats.bonksPerMin.toFixed(1)}</div>
                <div className="meme-pill">Borgars {memeStats.borgars}</div>
                <div className="meme-pill">Deaths/2m {memeStats.deaths2m}</div>
                <div className="meme-pill">Hype {memeStats.hype}</div>
                <div className="meme-pill">Bonk Conf {memeStats.bonkConfidence}%</div>
                <div className="meme-pill">Menu Doom {memeStats.menuDoom}</div>
                <div className="meme-pill">Chat Influence {memeStats.chatInfluence}%</div>
              </div>

              {fuseViz ? (
                <div className="fuse-wrap">
                  <svg className={`fuse ${fuseViz.hot ? "fuse-hot" : "fuse-cool"}`} viewBox={`0 0 ${fuseViz.w} ${fuseViz.h}`} preserveAspectRatio="none">
                    <rect x="0" y="0" width={fuseViz.w} height={fuseViz.h} className="fuse-bg" />
                    <line x1="0" y1={fuseViz.yCenter} x2={fuseViz.w} y2={fuseViz.yCenter} className="fuse-centerline" />
                    {fuseViz.eraLines.map((e: any, i: number) => (
                      <g key={i}>
                        <line x1={e.x} y1={fuseViz.nowY1} x2={e.x} y2={fuseViz.nowY2} stroke={e.color} strokeWidth="1" opacity="0.55" />
                        <text x={e.x + 4} y={fuseViz.nowLabelY} fill={e.color} fontSize="10">
                          {e.label}
                        </text>
                      </g>
                    ))}
                    {(fuseViz.bars as any[]).map((b: any, i: number) => (
                      <rect key={i} x={b.x} y={fuseViz.barY} width={b.w} height={fuseViz.barH} className={`fuse-bar ${b.cls}`}>
                        <title>{b.title}</title>
                      </rect>
                    ))}
                    <line x1={fuseViz.xWr} y1={fuseViz.wrY1} x2={fuseViz.xWr} y2={fuseViz.wrY2} className="fuse-wr" />
                    <text x={fuseViz.wrLabelX ?? fuseViz.xWr + 6} y={fuseViz.wrLabelY} className="fuse-wr-label">
                      WR
                    </text>
                    <line x1={fuseViz.xNow} y1={fuseViz.nowY1} x2={fuseViz.xNow} y2={fuseViz.nowY2} className="fuse-now" />
                    <circle cx={fuseViz.xNow} cy={fuseViz.nowDotY} r="6" className="fuse-now-dot" />
                    <text x={fuseViz.nowLabelX ?? fuseViz.xNow - 34} y={fuseViz.nowLabelY} className="fuse-now-label" textAnchor="end">
                      NOW
                    </text>
                  </svg>
                </div>
              ) : (
                <div className="fuse-wrap">
                  <svg className="fuse fuse-cool" viewBox="0 0 1280 86" preserveAspectRatio="none">
                    <rect x="0" y="0" width="1280" height="86" className="fuse-bg" />
                    <line x1="0" y1="49" x2="1280" y2="49" className="fuse-centerline" />
                  </svg>
                </div>
              )}

              <div className="stream-body">
                <div className="stream-main">
                  <div className="stream-stage-col">
                    {attractMode ? (
                      <div className="run-stage run-stage-idle">
                        <IdleStreamTile slotLabel="LIVE" className="run-stage-focus stream-tile-focused">
                          <AttractOverlay hof={(hofQ.data as any) ?? []} shame={shameList} clips={(attractQ.data as any) ?? null} />
                        </IdleStreamTile>
                        <div className="run-stage-mosaic" aria-label="run mosaic">
                          <IdleStreamTile slotLabel="HOT" cornerTag="HOT" className="run-stage-mosaic-tile" />
                          <IdleStreamTile slotLabel="CHASER" cornerTag="CHASER" className="run-stage-mosaic-tile" />
                          <IdleStreamTile slotLabel="COMMUNITY" cornerTag="COMMUNITY" className="run-stage-mosaic-tile" />
                          <IdleStreamTile slotLabel="#2" cornerTag="#2" className="run-stage-mosaic-tile" />
                        </div>
                      </div>
                    ) : (
                      <div className="run-stage">
                        <StreamTile
                          w={focus}
                          slotLabel={manualFocusActive ? "LIVE (MANUAL)" : directorActive ? "LIVE (DIRECTOR)" : "LIVE"}
                          className="run-stage-focus"
                          onClick={() => setFocusId(focus?.instance_id ?? null)}
                          impact={Boolean(focus?.instance_id && impactMap[String(focus.instance_id)] > nowMs)}
                          focused
                          rankUp={rankUpFor(focus?.instance_id)}
                          scoreDelta30s={focus?.instance_id ? scoreDelta30sById[String(focus.instance_id)] : null}
                          stepRate={focus?.instance_id ? stepRateById[String(focus.instance_id)] : null}
                          resets5m={focus?.instance_id ? resetsById[String(focus.instance_id)] : 0}
                        />
                        <div className="run-stage-mosaic" aria-label="run mosaic">
                          {stageMosaic.map((s) => {
                            const iid = String(s.hb.instance_id ?? "");
                            return (
                              <StreamTile
                                key={`mosaic-${iid}`}
                                w={s.hb}
                                slotLabel={s.label}
                                cornerTag={s.label}
                                className="run-stage-mosaic-tile"
                                onClick={() => setFocusId(s.hb.instance_id ?? null)}
                                impact={Boolean(iid && impactMap[iid] > nowMs)}
                                focused={String(iid) === String(focus?.instance_id)}
                                rankUp={rankUpFor(iid)}
                                scoreDelta30s={scoreDelta30sById[iid]}
                                stepRate={stepRateById[iid]}
                                resets5m={resetsById[iid]}
                              />
                            );
                          })}
                          {stageMosaic.length < 4 ? (
                            Array.from({ length: 4 - stageMosaic.length }).map((_, i) => (
                              <div key={`mosaic-empty-${i}`} className="run-stage-mosaic-empty">
                                <div className="muted">waiting…</div>
                              </div>
                            ))
                          ) : null}
                        </div>
                        {hotRun ? (
                          <div className="hotruns" aria-label="hot runs ticker">
                            <span className="badge">HOT</span>
                            <span className="hotruns-text">
                              {workersById[hotRun.iid]?.display_name ?? hotRun.iid} +{fmt(hotRun.delta, 2)} in 30s
                            </span>
                          </div>
                        ) : null}
                      </div>
                    )}

                    {!attractMode && replayPromoteUntil > nowMs ? (
                      <ReplayPiP
                        clips={(attractQ.data as any) ?? null}
                        leader={leaderTop[0] ?? null}
                        runnerUp={leaderTop[1] ?? null}
                        why={hotRun?.iid && hotRun.iid === String(leaderTop[0]?.instance_id ?? "") ? `surging: +${fmt(hotRun.delta, 2)} / 30s` : undefined}
                      />
                    ) : null}
                  </div>

                  <div className="stream-rail">
                    <div className="stream-card stream-leaderboard-card">
                      <div className="stream-card-title">Leaderboard</div>
                      {batchModeUntil > Date.now() ? (
                        <div className="overtake-ticker">
                          <span className="badge">BATCH</span>
                          {overtakeTicker.map((x, i) => (
                            <span key={`${x.iid}-${i}`} className="overtake-chip">
                              {x.iid} ▲ +{x.delta}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      <div className="lb-table" style={{ marginTop: 10 }}>
                        <div className="lb-row lb-head">
                          <div>#</div>
                          <div className="lb-lane-head" />
                          <div>Agent</div>
                          <div>Score</div>
                          <div>Δ30s</div>
                          <div className="lb-tag-head">Tag</div>
                          <div className="lb-extra">Pace</div>
                          <div className="lb-extra">Stall</div>
                          <div className="lb-spark lb-extra">Trend</div>
                        </div>
                        <div className="lb-list" style={{ ["--lb-row-count" as any]: leaderRows.length || 6 } as any}>
                          {(leaderRows.length ? leaderRows : Array.from({ length: 6 }).map((_, i) => ({ __skel: true, instance_id: `skel-${i}` }))).map(
                            (w: any, i: number) => {
                            const iid = String(w.instance_id);
                            const isSkel = Boolean(w.__skel);
                            const hist = histById[iid];
                            const bestScore = hist?.best_score ?? null;
                            const rankUp = rankUpFor(iid);
                            const burst = rankBurstFor(iid);
                            const heat = heatMap[iid];
                            const isLive = String(iid) === String(focus?.instance_id ?? "");
                            const isChaser = i === 1;
                            const danger = Math.max(
                              0,
                              Math.min(1, Number(w?.danger_level ?? (w?.survival_prob != null ? 1 - Number(w.survival_prob) : 0)) || 0),
                            );
                            const dangerPct = Math.round(danger * 100);
                            const delta30 = scoreDelta30sById[iid];
                            const pace = delta30 != null ? delta30 * 2 : null;
                            const stallTs = lastScoreChangeRef.current[iid] ?? Date.now();
                            const stall = Math.max(0, Math.round((Date.now() - stallTs) / 1000));
                            const spark = sparkSeriesRef.current[iid] ?? [];
                            const chips = (chipStateRef.current[iid] ?? []).filter((c) => c.ts + CHIP_TTL_MS > Date.now());
                            const prev = i > 0 ? leaderRows[i - 1] : null;
                            const targetScore = prev ? scoreOf(prev) : null;
                            const curScore = scoreOf(w);
                            const toPass = targetScore != null ? Math.max(0, targetScore - curScore + 0.01) : null;
                            const pacePerSec = delta30 != null ? delta30 / 30 : null;
                            const eta = toPass != null && pacePerSec != null && pacePerSec > 0 ? toPass / pacePerSec : null;
                            const etaNorm = eta != null ? clamp(eta / 30, 0, 1) : null;
                            let tag: IconKey | null = null;
                            if (w?.overrun || Number(w?.enemy_count ?? 0) > 0) tag = "boss";
                            else if (dangerPct >= 75) tag = "warning";
                            else if (Number((w as any)?.borgar_count ?? 0) > 0) tag = "borgar";
                            else if (Number((w as any)?.luck_drop_count ?? 0) > 0) tag = "tome_mastery";
                            else if (delta30 != null && delta30 > 1) tag = "big_bonk";
                            return (
                              <div
                                key={iid}
                                className={`lb-row lb-item ${isSkel ? "lb-skel" : ""} ${String(iid) === String(focus?.instance_id) ? "active" : ""} ${burst ? "rank-up" : ""} ${
                                  burst?.size ? `rank-${burst.size}` : ""
                                } ${swapUntil > nowMs ? "swap-active" : ""} ${heat && heat.until > nowMs ? (heat.kind === "up" ? "heat-up" : "heat-down") : ""}`}
                                style={{ ["--lb-row-i" as any]: i } as any}
                              >
                                <div className="muted">{isSkel ? "—" : i + 1}</div>
                                <div className="lb-lane">
                                  {isSkel ? <span className="lb-skel-dot" /> : burst ? <span className={`lb-lane-mark ${burst.size}`}>▲</span> : <span className="lb-lane-mark idle">•</span>}
                                  {!isSkel && etaNorm != null ? (
                                    <span className="lb-pace-mark" style={{ transform: `translateX(${etaNorm * 14}px)` }} />
                                  ) : null}
                                </div>
                                <div className="lb-agent">
                                  {isSkel ? (
                                    <div className="lb-skel-name" />
                                  ) : (
                                    <>
                                      <SpriteIcon idx={sheetIcon("policy")} size={16} title={String(w.policy_name ?? "")} />
                                      <span className="lb-agent-name">{w.display_name ?? iid}</span>
                                      {isLive ? <span className="lb-afford lb-live">LIVE</span> : null}
                                      {isChaser ? <span className="lb-afford lb-hunt">HUNTING</span> : null}
                                      {dangerPct >= 75 ? <SpriteIcon idx={sheetIcon("danger")} size={16} title={`danger ${dangerPct}%`} /> : null}
                                    </>
                                  )}
                                </div>
                                <div className="lb-score numeric">
                                  {isSkel ? (
                                    <>
                                      <div className="lb-skel-bar" />
                                      <div className="lb-skel-bar sub" />
                                    </>
                                  ) : (
                                    <>
                                      <div className="lb-score-main">{scoreDisplay(w, 2)}</div>
                                      <div className="lb-score-sub muted">best {bestScore == null ? "—" : fmt(bestScore, 2)}</div>
                                    </>
                                  )}
                                </div>
                                <div
                                  className={`lb-delta numeric ${delta30 != null ? (delta30 > 0 ? "delta-pos" : delta30 < 0 ? "delta-neg" : "") : ""}`}
                                >
                                  {isSkel ? (
                                    <>
                                      <div className="lb-skel-bar" />
                                      <div className="lb-skel-bar sub" />
                                    </>
                                  ) : (
                                    <>
                                      <div className="lb-delta-main">{fmtDelta(delta30, 2)}</div>
                                      <div className="lb-delta-sub">
                                        {rankUp ? <span className="rank-callout">▲ +{rankUp}</span> : null}
                                        {toPass != null && toPass > 0 ? <span className="lb-pace-text">+{fmt(toPass, 2)} to pass</span> : null}
                                      </div>
                                    </>
                                  )}
                                </div>
                                <div className="lb-tag">
                                  {isSkel ? <span className="lb-skel-dot" /> : tag ? <SpriteIcon idx={sheetIcon(tag)} size={16} title={tag} /> : <span className="muted">—</span>}
                                </div>
                                <div className="lb-extra numeric">{isSkel ? "—" : pace == null ? "—" : fmtDelta(pace, 1)}</div>
                                <div className="lb-extra numeric">{isSkel ? "—" : stall ? `${stall}s` : "—"}</div>
                                <div className="lb-spark lb-extra">
                                  {!isSkel ? (
                                    <svg viewBox="0 0 60 16" preserveAspectRatio="none">
                                      <path d={sparkPath(spark)} />
                                    </svg>
                                  ) : (
                                    <div className="lb-skel-bar" />
                                  )}
                                </div>
                                <div className="lbChipRail">
                                  {!isSkel
                                    ? chips.slice(0, 2).map((chip, idx) => (
                                        <span key={`${chip.label}-${idx}`} className={`chip isOn ${idx > 0 ? "chip-secondary" : ""}`}>
                                          <SpriteIcon idx={sheetIcon(chip.icon)} size={16} />
                                          <span className="chip-label">{chip.label}</span>
                                          {chip.value ? <span className="chip-value numeric">{chip.value}</span> : null}
                                        </span>
                                      ))
                                    : null}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        {!leaderTop.length && !(histQ.data ?? []).length ? <div className="muted">waiting for instances…</div> : null}
                      </div>
                    </div>

                    <HighlightPanel
                      clips={(attractQ.data as any) ?? null}
                      leader={leaderTop[0] ?? null}
                      runnerUp={leaderTop[1] ?? null}
                      pulse={highlightPulseUntil > Date.now()}
                    />

                    <div className="stream-card stream-highlight" style={{ marginTop: 12 }}>
                      <div className="stream-card-title">Replay Buffer</div>
                      <div className="stream-highlight-media">
                        {replayQueue.length ? (
                          <div className="replay-queue">
                            {replayQueue.map((r) => (
                              <div key={r.clipUrl} className="replay-queue-item">
                                <video className="replay-queue-video" src={r.clipUrl} autoPlay muted playsInline loop />
                                <div className="replay-queue-label">
                                  <span>{r.label}</span>
                                  <span className="muted">{timeAgo(r.ts)}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="stream-highlight-placeholder">waiting for event-driven clips…</div>
                        )}
                      </div>
                    </div>

                  <div className="stream-tabs">
                    <div className="stream-tabbar">
                      {debugParamOn ? (
                        <>
                          <button
                            className={`stream-tabbtn ${railTab === "moments" ? "active" : ""}`}
                            onClick={() => {
                              setRailAuto(false);
                              setRailTab("moments");
                            }}
                          >
                            <SpriteIcon idx={sheetIcon("moment")} size={16} className="tab-ico" /> Moments
                          </button>
                          <button
                            className={`stream-tabbtn ${railTab === "community" ? "active" : ""}`}
                            onClick={() => {
                              setRailAuto(false);
                              setRailTab("community");
                            }}
                          >
                            <SpriteIcon idx={sheetIcon("bonk_bucks")} size={16} className="tab-ico" /> Community
                          </button>
                          <button
                            className={`stream-tabbtn ${railTab === "bucks" ? "active" : ""}`}
                            onClick={() => {
                              setRailAuto(false);
                              setRailTab("bucks");
                            }}
                          >
                            <SpriteIcon idx={sheetIcon("bonk_bucks")} size={16} className="tab-ico" /> Bonk Bucks
                          </button>
                          <button
                            className={`stream-tabbtn ${railTab === "poll" ? "active" : ""}`}
                            onClick={() => {
                              setRailAuto(false);
                              setRailTab("poll");
                            }}
                          >
                            <SpriteIcon idx={sheetIcon("tome_random")} size={16} className="tab-ico" /> Vote
                          </button>
                          <button
                            className={`stream-tabbtn ${railTab === "progress" ? "active" : ""}`}
                            onClick={() => {
                              setRailAuto(false);
                              setRailTab("progress");
                            }}
                          >
                            <SpriteIcon idx={sheetIcon("tome_mastery")} size={16} className="tab-ico" /> Progress
                          </button>
                          <button
                            className={`stream-tabbtn ${railTab === "controls" ? "active" : ""}`}
                            onClick={() => {
                              setRailAuto(false);
                              setRailTab("controls");
                            }}
                            style={{ marginLeft: "auto" }}
                          >
                            <SpriteIcon idx={sheetIcon("settings")} size={16} className="tab-ico" /> Controls
                          </button>
                        </>
                      ) : (
                        <>
                          <span className={`stream-tablabel ${railTab === "moments" ? "active" : ""}`}>
                            <SpriteIcon idx={sheetIcon("moment")} size={16} className="tab-ico" title="Moments" />
                          </span>
                          <span className={`stream-tablabel ${railTab === "community" ? "active" : ""}`}>
                            <SpriteIcon idx={sheetIcon("bonk_bucks")} size={16} className="tab-ico" title="Community" />
                          </span>
                          <span className={`stream-tablabel ${railTab === "bucks" ? "active" : ""}`}>
                            <SpriteIcon idx={sheetIcon("bonk_bucks")} size={16} className="tab-ico" title="Bonk Bucks" />
                          </span>
                          <span className={`stream-tablabel ${railTab === "poll" ? "active" : ""}`}>
                            <SpriteIcon idx={sheetIcon("tome_random")} size={16} className="tab-ico" title="Vote" />
                          </span>
                          <span className={`stream-tablabel ${railTab === "progress" ? "active" : ""}`}>
                            <SpriteIcon idx={sheetIcon("tome_mastery")} size={16} className="tab-ico" title="Progress" />
                          </span>
                        </>
                      )}
                    </div>

                      <div className="stream-tabpanel">
                        {railTab === "moments" ? (
                          <div className="stream-card">
                            <div className="stream-card-title">Moments</div>
                            <div className="stream-events" style={{ marginTop: 8 }}>
                              {interesting.map((e) => (
                                <div key={e.event_id} className="stream-event">
                                  <span className="stream-event-ico">
                                    <SpriteIcon idx={eventIconIdx(e.event_type)} size={16} />
                                  </span>
                                  <span className="badge">{e.event_type}</span>
                                  <span className="stream-event-msg">
                                    {e.instance_id ? `${e.instance_id}: ` : ""}
                                    {e.message}
                                  </span>
                                  <span className="muted">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
                                </div>
                              ))}
                              {!interesting.length ? <div className="muted">waiting for moments…</div> : null}
                            </div>
                          </div>
                        ) : null}

                        {railTab === "community" ? (
                          <div className="stream-card">
                            <div className="stream-card-title">Community</div>
                            <div className="muted">Pins + bounties (chat + Bonk Bucks).</div>
                            <div className="timeline-pins" style={{ marginTop: 8 }}>
                              {(timeline?.pins ?? [])
                                .slice(-6)
                                .reverse()
                                .map((p: any) => (
                                  <div
                                    key={String(p.pin_id)}
                                    className={`timeline-pin ${p.kind === "system" ? "timeline-pin-system" : ""}`}
                                  >
                                    <span className="timeline-icon">{String(p.kind ?? "PIN").toUpperCase().slice(0, 3)}</span>
                                    <span style={{ fontWeight: 900 }}>{String(p.title ?? "Pin")}</span>
                                    <span className="muted">{String(p.instance_id ?? "")}</span>
                                  </div>
                                ))}
                              {!timeline?.pins?.length ? <div className="muted">no pins yet</div> : null}
                            </div>
                          </div>
                        ) : null}

                        {railTab === "bucks" ? (
                          <div className="stream-card">
                            <div className="stream-card-title">Bonk Bucks</div>
                            {!betting ? (
                              <div className="muted">loading…</div>
                            ) : betting.round?.active ? (
                              <>
                                <div className="muted">{betting.round.question ?? ""}</div>
                                <div className="bet-opts" style={{ marginTop: 8 }}>
                                  {(betting.round.options ?? []).slice(0, 2).map((o: any) => (
                                    <div key={String(o.name)} className="bet-opt">
                                      <span className="badge">{String(o.name)}</span>
                                      <span className="muted">{String(o.odds ?? "")}</span>
                                      <span className="muted">pool {fmtCompact(Number(o.pool ?? 0))}</span>
                                    </div>
                                  ))}
                                </div>
                              </>
                            ) : (
                              <div className="muted">no active bet</div>
                            )}
                          </div>
                        ) : null}

                        {railTab === "poll" ? (
                          <div className="stream-card">
                            <div className="stream-card-title">Blessing vs Curse</div>
                            {!poll ? (
                              <div className="muted">loading…</div>
                            ) : poll.active ? (
                              <>
                                <div className="muted">{poll.question ?? ""}</div>
                                {(() => {
                                  const v0 = Number(poll.votes?.[0] ?? 0);
                                  const v1 = Number(poll.votes?.[1] ?? 0);
                                  const tot = Math.max(1, v0 + v1);
                                  const pct = (v0 / tot) * 100;
                                  return (
                                    <div className="tug" style={{ marginTop: 8 }}>
                                      <div className="tug-bar">
                                        <div className="tug-left" style={{ width: `${pct}%` }} />
                                        <div className="tug-right" style={{ width: `${100 - pct}%` }} />
                                      </div>
                                      <div className="tug-labels">
                                        <span className="badge">{poll.options?.[0] ?? "A"}</span>
                                        <span className="muted">{fmtCompact(v0)}</span>
                                        <span className="badge">{poll.options?.[1] ?? "B"}</span>
                                        <span className="muted">{fmtCompact(v1)}</span>
                                      </div>
                                    </div>
                                  );
                                })()}
                              </>
                            ) : (
                              <div className="muted">no active poll</div>
                            )}
                          </div>
                        ) : null}

                        {railTab === "progress" ? (
                          debugParamOn ? (
                            <ProgressTrackerPanel />
                          ) : (
                            <div className="stream-card">
                              <div className="stream-card-title">Progress</div>
                              <div className="muted">progress controls are available in dev mode</div>
                            </div>
                          )
                        ) : null}
                        {railTab === "controls" && debugParamOn ? (
                          <div className="stream-card">
                            <div className="stream-card-title">Controls</div>
                            <div className="row" style={{ gap: 10, flexWrap: "wrap", marginTop: 8 }}>
                              <button
                                className={`btn btn-ghost ${safeOn ? "active" : ""}`}
                                onClick={() => setQueryParams({ safe: safeOn ? null : "1" })}
                                title={import.meta.env.PROD ? "requires ?debug=1&safe=1" : "toggle ?safe=1"}
                              >
                                Safe Guides
                              </button>
                              <button
                                className={`btn btn-ghost ${lowVision ? "active" : ""}`}
                                onClick={() => {
                                  const next = !lowVision;
                                  try {
                                    window.localStorage.setItem("mb:vision", next ? "1" : "0");
                                  } catch {}
                                  setQueryParams({ vision: next ? "1" : "0" });
                                }}
                                title="toggle ?vision=1/0 (persists to localStorage)"
                              >
                                Vision
                              </button>
                              <button
                                className={`btn btn-ghost ${directorOn ? "active" : ""}`}
                                onClick={() => setDirectorOn((v) => !v)}
                                title="auto-focus on interesting moments"
                              >
                                Director {directorOn ? "ON" : "OFF"}
                              </button>
                              <button
                                className="btn btn-ghost"
                                onClick={() => setLayoutMode((v) => (v === "broadcast" ? "dense" : "broadcast"))}
                                title="toggle layout density"
                              >
                                Layout {layoutMode === "broadcast" ? "BROADCAST" : "DENSE"}
                              </button>
                              <button
                                className={`btn btn-ghost ${railAuto ? "active" : ""}`}
                                onClick={() => setRailAuto((v) => !v)}
                                title="auto-rotate panels"
                              >
                                Rail Auto {railAuto ? "ON" : "OFF"}
                              </button>
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
