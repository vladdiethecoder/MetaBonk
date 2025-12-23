import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { EffectComposer, Bloom, Vignette, Noise, ChromaticAberration } from "@react-three/postprocessing";
import { useQuery } from "@tanstack/react-query";
import * as THREE from "three";
import { fetchWorkers, type Heartbeat } from "../api";
import { useEventStream } from "../hooks";
import { CrtModel, type MonitorEffect } from "../components/CrtModel";
import "./CCTVOverlay.css";

type DirectorState = {
  heroId: string | null;
  heroReason: string;
  ambientIds: string[];
  interstitial: MonitorEffect;
};

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

const jitterFromId = (id: string, idx: number) => {
  const seed = id.split("").reduce((acc, c) => acc + c.charCodeAt(0) * (idx + 1), 0);
  const r = (n: number) => ((Math.sin(seed + n) + 1) / 2) * 0.1 - 0.05;
  return [r(1), r(2), r(3)] as [number, number, number];
};

export default function CCTV3D() {
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

  const sorted = useMemo(() => [...workers].sort((a, b) => scoreOf(b) - scoreOf(a)), [workers]);
  const hero = sorted.find((w) => String((w as any)?.instance_id ?? "") === state.heroId) || sorted[0] || null;
  const others = sorted.filter((w) => String((w as any)?.instance_id ?? "") !== String(hero?.instance_id ?? "")).slice(0, 4);

  const COL_HERO = "#AEEFFF";
  const COL_SIDE = "#5588AA";

  return (
    <div className="cctv-container">
      <div className="cctv-canvas-layer">
        <Canvas
          shadows
          gl={{ antialias: false, stencil: false, depth: true }}
          camera={{ position: [0, 0.15, 2.2], fov: 38 }}
          onCreated={({ gl }) => {
            gl.outputColorSpace = THREE.SRGBColorSpace;
          }}
        >
          <color attach="background" args={["#030507"]} />
          <fog attach="fog" args={["#030507", 2, 6]} />
          <Suspense fallback={null}>
            <group position={[0, -0.2, 0]}>
              <CrtModel
                position={[0, 0.35, 0]}
                scale={1.6}
                streamUrl={hero?.stream_url}
                label={hero ? `TARGET: ${String(hero.instance_id).slice(0, 6)}` : "NO SIGNAL"}
                color={COL_HERO}
                isHero
                effect={state.interstitial}
              />

              <group position={[-1.55, 0.15, 0.4]} rotation={[0, 0.28, 0]}>
                <CrtModel
                  position={[0, 0.55, 0]}
                  streamUrl={others[0]?.stream_url}
                  label={others[0] ? "CAM-01" : "--"}
                  color={COL_SIDE}
                  rotation={others[0] ? jitterFromId(String(others[0].instance_id ?? "cam1"), 1) : [0, 0, 0]}
                />
                <CrtModel
                  position={[0, -0.55, 0]}
                  streamUrl={others[1]?.stream_url}
                  label={others[1] ? "CAM-02" : "--"}
                  color={COL_SIDE}
                  rotation={others[1] ? jitterFromId(String(others[1].instance_id ?? "cam2"), 2) : [0, 0, 0]}
                />
              </group>

              <group position={[1.55, 0.15, 0.4]} rotation={[0, -0.28, 0]}>
                <CrtModel
                  position={[0, 0.55, 0]}
                  streamUrl={others[2]?.stream_url}
                  label={others[2] ? "CAM-03" : "--"}
                  color={COL_SIDE}
                  rotation={others[2] ? jitterFromId(String(others[2].instance_id ?? "cam3"), 3) : [0, 0, 0]}
                />
                <CrtModel
                  position={[0, -0.55, 0]}
                  streamUrl={others[3]?.stream_url}
                  label={others[3] ? "CAM-04" : "--"}
                  color={COL_SIDE}
                  rotation={others[3] ? jitterFromId(String(others[3].instance_id ?? "cam4"), 4) : [0, 0, 0]}
                />
              </group>

              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.0, 1]}>
                <planeGeometry args={[10, 5]} />
                <meshStandardMaterial color="#05080a" roughness={0.1} metalness={0.6} />
              </mesh>
            </group>

            <ambientLight intensity={0.2} color="#ccddff" />
            <spotLight position={[0, 5, -2]} intensity={4} color="#aaccff" angle={1} penumbra={1} />

            <EffectComposer disableNormalPass>
              <Bloom luminanceThreshold={1.1} luminanceSmoothing={0.3} intensity={0.6} mipmapBlur />
              <ChromaticAberration offset={[0.002, 0.001]} />
              <Noise opacity={0.08} />
              <Vignette eskil={false} offset={0.1} darkness={0.8} />
            </EffectComposer>
          </Suspense>
        </Canvas>
      </div>

      <div className="pip-overlay">
        <div className="pip-scanlines" />
        <div className="pip-hud-top">
          <h1>METABONK // SINGULARITY</h1>
          <div className="pip-tag">
            HIVE_MIND: <span className="blink">CONNECTED</span>
          </div>
        </div>
        <div className="pip-hud-bottom">
          <div className="pip-stat">
            <label>SUBJECT</label>
            <span>{hero?.instance_id || "SCANNING..."}</span>
          </div>
          <div className="pip-stat">
            <label>SCORE</label>
            <span>{Math.floor(hero?.reward ?? 0)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
