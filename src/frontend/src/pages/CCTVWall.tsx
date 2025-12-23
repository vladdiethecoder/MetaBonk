import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Canvas, useFrame } from "@react-three/fiber";
import { EffectComposer, Bloom, Noise, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import { fetchWorkers, type Heartbeat } from "../api";
import { useEventStream } from "../hooks";
import { useMseVideoTexture } from "../hooks/useMseVideoTexture";
import type { MonitorEffect } from "../components/CctvMonitor";
import "./CCTVWall.css";

type DirectorState = {
  heroId: string | null;
  heroReason: string;
  ambientIds: string[];
  interstitial: MonitorEffect;
};

const CRT_TINT = new THREE.Color("#7cffb4");

let _audioCtx: AudioContext | null = null;
const playClick = () => {
  try {
    if (!_audioCtx) _audioCtx = new AudioContext();
    const ctx = _audioCtx;
    if (ctx.state === "suspended") ctx.resume();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 900 + Math.random() * 280;
    gain.gain.value = 0.05;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.06);
    osc.stop(ctx.currentTime + 0.07);
  } catch {
    // ignore autoplay restrictions
  }
};

const pickInterstitial = () => {
  const r = Math.random();
  if (r < 0.7) return "static";
  if (r < 0.92) return "bars";
  return "dvd";
};

const scoreOf = (w: Heartbeat | null | undefined) => (w?.steam_score ?? w?.reward ?? 0) as number;

const eventBoostMap = (eventType: string) => {
  const t = eventType.toUpperCase();
  if (["RECORD", "NEWRECORD"].includes(t)) return { bonus: 6, reason: "RECORD" };
  if (["PB", "PERSONALBEST"].includes(t)) return { bonus: 5, reason: "PB" };
  if (["DISASTER", "WEIRDBUILD", "CLUTCH"].includes(t)) return { bonus: 4, reason: t };
  if (t.includes("OVERTAKE") || t.includes("RANK")) return { bonus: 3.5, reason: "RANK SWAP" };
  if (t.includes("BOSS")) return { bonus: 3.5, reason: "BOSS" };
  if (t.includes("CHAT") || t.includes("SPIKE")) return { bonus: 2.4, reason: "CHAT SPIKE" };
  if (["OVERCRIT", "NEWMAXHIT", "LOOTDROP", "BOUNTYCLAIMED", "EUREKA"].includes(t)) return { bonus: 3, reason: t };
  return { bonus: 0, reason: "" };
};

function buildCrtScreenGeometry(width: number, height: number, curvature: number) {
  const geom = new THREE.PlaneGeometry(width, height, 32, 24);
  const pos = geom.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const xNorm = Math.abs(x / (width * 0.5));
    const yNorm = Math.abs(y / (height * 0.5));
    const bulge = (1 - Math.min(1, (xNorm * xNorm + yNorm * yNorm) * 0.9)) * curvature;
    pos.setZ(i, bulge);
  }
  geom.computeVertexNormals();
  return geom;
}

function useInterstitialTexture(effect: MonitorEffect) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textureRef = useRef<THREE.CanvasTexture | null>(null);
  const dvdState = useRef({ x: 40, y: 60, vx: 3.2, vy: 2.6, colorIdx: 0 });

  if (!canvasRef.current) {
    const cvs = document.createElement("canvas");
    cvs.width = 640;
    cvs.height = 360;
    canvasRef.current = cvs;
  }
  if (!textureRef.current) {
    textureRef.current = new THREE.CanvasTexture(canvasRef.current);
    textureRef.current.colorSpace = THREE.SRGBColorSpace;
  }

  useFrame(() => {
    if (!effect) return;
    const cvs = canvasRef.current;
    const ctx = cvs?.getContext("2d");
    if (!cvs || !ctx) return;
    const w = cvs.width;
    const h = cvs.height;
    ctx.clearRect(0, 0, w, h);
    if (effect === "static") {
      const img = ctx.createImageData(w, h);
      for (let i = 0; i < img.data.length; i += 4) {
        const v = (Math.random() * 255) | 0;
        img.data[i] = v;
        img.data[i + 1] = v;
        img.data[i + 2] = v;
        img.data[i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#d8ffe9";
      ctx.font = "bold 48px 'Space Mono', ui-monospace, monospace";
      ctx.fillText("NO SIGNAL", w * 0.28, h * 0.55);
    } else if (effect === "bars") {
      const colors = ["#c0c0c0", "#c0c000", "#00c0c0", "#00c000", "#c000c0", "#c00000", "#0000c0"];
      const bw = w / colors.length;
      colors.forEach((c, i) => {
        ctx.fillStyle = c;
        ctx.fillRect(i * bw, 0, bw + 1, h);
      });
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fillRect(0, h * 0.68, w, h * 0.06);
    } else if (effect === "dvd") {
      const colors = ["#ff4b4b", "#5eff8a", "#68b5ff", "#ffe86a", "#ff8ad6"];
      const st = dvdState.current;
      st.x += st.vx;
      st.y += st.vy;
      if (st.x <= 18 || st.x >= w - 90) {
        st.vx *= -1;
        st.colorIdx = (st.colorIdx + 1) % colors.length;
      }
      if (st.y <= 30 || st.y >= h - 30) {
        st.vy *= -1;
        st.colorIdx = (st.colorIdx + 1) % colors.length;
      }
      ctx.fillStyle = "rgba(0,0,0,0.85)";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = colors[st.colorIdx];
      ctx.font = "bold 46px 'Space Mono', ui-monospace, monospace";
      ctx.fillText("DVD", st.x, st.y);
    }
    if (textureRef.current) textureRef.current.needsUpdate = true;
  });

  return effect ? textureRef.current : null;
}

function CctvScreen3D({
  agent,
  label,
  isHero,
  effect,
  position,
  rotation,
  scale,
}: {
  agent: Heartbeat | null;
  label: string;
  isHero?: boolean;
  effect: MonitorEffect;
  position: [number, number, number];
  rotation?: [number, number, number];
  scale?: number;
}) {
  const streamUrl = String(agent?.stream_url ?? "");
  const texture = useMseVideoTexture(streamUrl || undefined);
  const interstitialTex = useInterstitialTexture(effect);
  const screenGeom = useMemo(() => buildCrtScreenGeometry(1.4, 0.9, 0.04), []);
  const jitter = useMemo(() => {
    const h = label.split("").reduce((acc, c) => acc + c.charCodeAt(0), 0);
    const rand = (n: number) => ((Math.sin(n + h) + 1) / 2) * 0.08 - 0.04;
    return { rx: rand(1), ry: rand(2), rz: rand(3) };
  }, [label]);

  const map = effect ? interstitialTex : texture;
  const emissiveIntensity = isHero ? 1.55 : 1.2;

  return (
    <group position={position} rotation={rotation} scale={scale ?? 1}>
      <mesh position={[0, 0, -0.04]} rotation={[jitter.rx, jitter.ry, jitter.rz]}>
        <boxGeometry args={[1.7, 1.1, 0.35]} />
        <meshStandardMaterial color="#0d1112" roughness={0.7} metalness={0.2} />
      </mesh>
      <mesh position={[0, 0, 0.12]} geometry={screenGeom} rotation={[jitter.rx, jitter.ry, jitter.rz]}>
        {map ? (
          <meshStandardMaterial
            map={map}
            emissive={CRT_TINT}
            emissiveMap={map}
            emissiveIntensity={emissiveIntensity}
            color={CRT_TINT}
            toneMapped={false}
          />
        ) : (
          <meshStandardMaterial emissive="#153522" emissiveIntensity={0.5} toneMapped={false} />
        )}
      </mesh>
      <pointLight position={[0, 0, 0.35]} intensity={isHero ? 1.6 : 0.9} color="#7dffd0" distance={4} />
    </group>
  );
}

function CctvRig3D({
  hero,
  ambients,
  effect,
}: {
  hero: Heartbeat | null;
  ambients: Array<Heartbeat | null>;
  effect: MonitorEffect;
}) {
  return (
    <group>
      <CctvScreen3D agent={hero} label="MASTER" isHero effect={effect} position={[0, 0.35, 0]} scale={1.35} />
      <group position={[-1.55, 0.15, 0.25]} rotation={[0, 0.28, 0]}>
        <CctvScreen3D agent={ambients[0] ?? null} label="CAM 01" effect={null} position={[0, 0.42, 0]} scale={0.72} />
        <CctvScreen3D agent={ambients[1] ?? null} label="CAM 02" effect={null} position={[0, -0.38, 0]} scale={0.72} />
      </group>
      <group position={[1.55, 0.15, 0.25]} rotation={[0, -0.28, 0]}>
        <CctvScreen3D agent={ambients[2] ?? null} label="CAM 03" effect={null} position={[0, 0.42, 0]} scale={0.72} />
        <CctvScreen3D agent={ambients[3] ?? null} label="CAM 04" effect={null} position={[0, -0.38, 0]} scale={0.72} />
      </group>
    </group>
  );
}

export default function CCTVWall() {
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 1000 });
  const events = useEventStream(300);
  const workers = useMemo(() => Object.values(workersQ.data ?? {}), [workersQ.data]);

  const [state, setState] = useState<DirectorState>({
    heroId: null,
    heroReason: "BOOT",
    ambientIds: [],
    interstitial: null,
  });

  const lastCutRef = useRef(0);
  const cooldownRef = useRef<Record<string, number>>({});

  const recentEventBoosts = useMemo(() => {
    const now = Date.now();
    const windowMs = 20000;
    const m: Record<string, { bonus: number; reason: string }> = {};
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const e = events[i];
      if (!e?.instance_id) continue;
      if ((e.ts ?? 0) < now - windowMs) break;
      const iid = String(e.instance_id);
      if (m[iid]) continue;
      const boost = eventBoostMap(String(e.event_type ?? ""));
      if (boost.bonus > 0) m[iid] = boost;
    }
    return m;
  }, [events]);

  useEffect(() => {
    if (!workers.length) return;
    const tick = window.setInterval(() => {
      const now = Date.now();
      const minDwell = 12000;
      const cooldownMs = 30000;

      if (now - lastCutRef.current < minDwell) return;

      const sorted = [...workers].sort((a, b) => {
        const aBoost = recentEventBoosts[String((a as any)?.instance_id ?? "")]?.bonus ?? 0;
        const bBoost = recentEventBoosts[String((b as any)?.instance_id ?? "")]?.bonus ?? 0;
        const aDanger = Number(a?.danger_level ?? (a?.survival_prob != null ? 1 - Number(a.survival_prob) : 0)) || 0;
        const bDanger = Number(b?.danger_level ?? (b?.survival_prob != null ? 1 - Number(b.survival_prob) : 0)) || 0;
        const aCombat = Number(a?.enemy_count ?? 0) > 0 ? 1 : 0;
        const bCombat = Number(b?.enemy_count ?? 0) > 0 ? 1 : 0;
        const aOverrun = Boolean(a?.overrun ?? false) ? 1 : 0;
        const bOverrun = Boolean(b?.overrun ?? false) ? 1 : 0;
        const aScore = Math.log1p(scoreOf(a)) * 0.7 + aBoost * 3 + aCombat * 1.6 + aOverrun * 2 + aDanger * 2.5;
        const bScore = Math.log1p(scoreOf(b)) * 0.7 + bBoost * 3 + bCombat * 1.6 + bOverrun * 2 + bDanger * 2.5;
        return bScore - aScore;
      });

      const chaosPick = Math.random() < 0.035;
      const chaosPool = chaosPick ? sorted.slice(-Math.min(3, sorted.length)) : null;
      const chaosChoice = chaosPool ? chaosPool[Math.floor(Math.random() * chaosPool.length)] : null;
      const candidateList = chaosChoice ? [chaosChoice, ...sorted] : sorted;

      const nextHero = candidateList.find((w) => {
        const id = String((w as any)?.instance_id ?? "");
        if (!id) return false;
        const cooldown = cooldownRef.current[id] ?? 0;
        if (state.heroId && id !== state.heroId && cooldown > now) return false;
        return true;
      });

      if (!nextHero) return;
      const nextHeroId = String((nextHero as any)?.instance_id ?? "");
      const boost = recentEventBoosts[nextHeroId];
      const chaosId = chaosChoice ? String((chaosChoice as any)?.instance_id ?? "") : "";
      const reason = chaosId && chaosId === nextHeroId ? "CHAOS PICK" : boost?.reason || (nextHero?.overrun ? "OVERRUN" : "SCORE LEAD");

      if (nextHeroId && nextHeroId !== state.heroId) {
        const fx = pickInterstitial();
        setState({
          heroId: nextHeroId,
          heroReason: reason,
          ambientIds: sorted.filter((w) => String((w as any)?.instance_id ?? "") !== nextHeroId).slice(0, 4).map((w) => String((w as any)?.instance_id ?? "")),
          interstitial: fx,
        });
        lastCutRef.current = now;
        cooldownRef.current[nextHeroId] = now + cooldownMs;
        playClick();
        const hold = fx === "dvd" ? 1800 : fx === "bars" ? 850 : 400;
        window.setTimeout(() => {
          setState((prev) => ({ ...prev, interstitial: null }));
        }, hold);
        return;
      }

      setState((prev) => ({
        ...prev,
        ambientIds: sorted.filter((w) => String((w as any)?.instance_id ?? "") !== nextHeroId).slice(0, 4).map((w) => String((w as any)?.instance_id ?? "")),
      }));
    }, 1000);

    return () => window.clearInterval(tick);
  }, [workers, recentEventBoosts, state.heroId]);

  const heroAgent = workers.find((w) => String((w as any)?.instance_id ?? "") === state.heroId) || null;
  const ambientAgents = state.ambientIds.map((id) => workers.find((w) => String((w as any)?.instance_id ?? "") === id) || null);

  return (
    <div className="cctv-wall">
      <Canvas
        className="cctv-canvas"
        shadows
        camera={{ position: [0, 0.15, 2.2], fov: 38 }}
        onCreated={({ gl }) => {
          gl.outputColorSpace = THREE.SRGBColorSpace;
          gl.setClearColor(new THREE.Color("#050707"), 1);
        }}
      >
        <fog attach="fog" args={["#050707", 1.2, 5.4]} />
        <ambientLight intensity={0.25} />
        <pointLight position={[0, 1.8, 2.2]} intensity={0.6} color="#ffd7a6" />
        <pointLight position={[-2, 1.1, 1.6]} intensity={0.45} color="#7fffd2" />
        <CctvRig3D hero={heroAgent} ambients={ambientAgents} effect={state.interstitial} />
        <EffectComposer>
          <Bloom intensity={0.8} luminanceThreshold={0.2} luminanceSmoothing={0.75} mipmapBlur />
          <Noise opacity={0.12} />
          <Vignette eskil={false} offset={0.35} darkness={0.75} />
        </EffectComposer>
      </Canvas>
      <div className="cctv-hud">
        <div className="hud-title">META BONK // SURVEILLANCE</div>
        <div className="hud-sub">DIRECTOR_AI: ONLINE</div>
        <div className="hud-row">
          <span className="hud-label">HERO</span>
          <span className="hud-value">{heroAgent?.display_name ?? state.heroId ?? "SEARCHING"}</span>
        </div>
        <div className="hud-row">
          <span className="hud-label">REASON</span>
          <span className="hud-value">{state.heroReason.toUpperCase()}</span>
        </div>
      </div>
      <div className="wall-overlay" />
    </div>
  );
}
