import { useMemo, useRef, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchOverviewHealth, fetchOverviewIssues, fetchStatus, fetchWorkers } from "../api";
import { fmtFixed, timeAgo } from "../lib/format";

const NAV = [
  { to: "/", label: "Neuro" },
  { to: "/runs", label: "Runs" },
  { to: "/instances", label: "Instances" },
  { to: "/build", label: "Build Lab" },
  { to: "/skills", label: "Skills" },
  { to: "/spy", label: "Spy" },
  { to: "/stream", label: "Stream" },
];

function useCanvasLoop(draw: (ctx: CanvasRenderingContext2D, t: number, w: number, h: number) => void) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;
    let raf = 0;
    let start = performance.now();
    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const onResize = () => resize();
    window.addEventListener("resize", onResize);

    const loop = () => {
      const t = (performance.now() - start) / 1000;
      draw(ctx, t, canvas.clientWidth, canvas.clientHeight);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, [draw]);
  return ref;
}

function FluidField({ className, hue = 190, energy = 0.4 }: { className?: string; hue?: number; energy?: number }) {
  const particles = useRef(
    Array.from({ length: 960 }, () => ({
      x: Math.random(),
      y: Math.random(),
      v: 0.4 + Math.random() * 1.1,
      life: Math.random(),
    }))
  );
  const ref = useCanvasLoop((ctx, t, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(6, 8, 10, 0.18)";
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 1 + energy * 0.8;
    for (const p of particles.current) {
      const px = p.x * w;
      const py = p.y * h;
      const angle =
        Math.sin(px * 0.006 + t * (0.6 + energy)) +
        Math.cos(py * 0.007 - t * 0.4) +
        Math.sin((px + py) * 0.002 + t * 0.2);
      const speed = p.v * (0.3 + 0.7 * Math.sin(t + p.life * Math.PI));
      const nx = Math.cos(angle) * speed * (0.0018 + energy * 0.001);
      const ny = Math.sin(angle) * speed * (0.0018 + energy * 0.001);
      p.x += nx;
      p.y += ny;
      p.life += 0.002 + energy * 0.003;
      if (p.x < -0.1 || p.x > 1.1 || p.y < -0.1 || p.y > 1.1) {
        p.x = Math.random();
        p.y = Math.random();
        p.life = 0;
      }
      ctx.strokeStyle = `hsla(${hue}, 85%, 65%, ${0.25 + energy * 0.35})`;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - nx * w * 6, py - ny * h * 6);
      ctx.stroke();
    }
  });
  return <canvas ref={ref} className={className} />;
}

function VectorField({ className, intensity = 0.5 }: { className?: string; intensity?: number }) {
  const particles = useRef(
    Array.from({ length: 720 }, () => ({
      x: Math.random(),
      y: Math.random(),
      v: 0.8 + Math.random() * 1.3,
      life: Math.random(),
    }))
  );
  const ref = useCanvasLoop((ctx, t, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(5, 7, 9, 0.22)";
    ctx.fillRect(0, 0, w, h);
    for (const p of particles.current) {
      const px = p.x * w;
      const py = p.y * h;
      const curl = Math.sin((py + t * 120) * 0.004) * 2 + Math.cos((px - t * 80) * 0.003);
      const drift = Math.cos((px + py) * 0.002 - t * (0.3 + intensity * 0.4));
      const angle = curl + drift;
      const nx = Math.cos(angle) * p.v * (0.002 + intensity * 0.002);
      const ny = Math.sin(angle) * p.v * (0.0016 + intensity * 0.002);
      p.x += nx;
      p.y += ny;
      p.life += 0.01 + intensity * 0.02;
      if (p.life > 1 || p.x < -0.2 || p.x > 1.2 || p.y < -0.2 || p.y > 1.2) {
        p.x = Math.random();
        p.y = Math.random();
        p.life = 0;
      }
      ctx.strokeStyle = `rgba(110, 255, 206, ${0.14 + 0.5 * (1 - p.life)})`;
      ctx.lineWidth = 1 + intensity * 0.4;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - nx * w * 10, py - ny * h * 10);
      ctx.stroke();
    }
  });
  return <canvas ref={ref} className={className} />;
}

function LatentField({ className, drift = 0.2 }: { className?: string; drift?: number }) {
  const points = useRef(
    Array.from({ length: 520 }, () => ({
      x: Math.random() * 2 - 1,
      y: Math.random() * 2 - 1,
      z: Math.random() * 2 - 1,
      hue: 160 + Math.random() * 80,
    }))
  );
  const ref = useCanvasLoop((ctx, t, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(8, 10, 12, 0.24)";
    ctx.fillRect(0, 0, w, h);
    const cx = w * 0.5;
    const cy = h * 0.55;
    const rot = t * (0.2 + drift * 0.4);
    for (const p of points.current) {
      const x = p.x * Math.cos(rot) - p.z * Math.sin(rot);
      const z = p.x * Math.sin(rot) + p.z * Math.cos(rot);
      const y = p.y * Math.cos(rot * 0.7) - z * Math.sin(rot * 0.7);
      const depth = 1.5 + z;
      const px = cx + (x / depth) * (w * 0.28);
      const py = cy + (y / depth) * (h * 0.3);
      const r = Math.max(1, 3 - z * 1.4);
      ctx.fillStyle = `hsla(${p.hue}, 80%, 68%, ${0.35 + z * 0.3})`;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(160, 255, 230, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.min(w, h) * 0.32, 0, Math.PI * 2);
    ctx.stroke();
  });
  return <canvas ref={ref} className={className} />;
}

function NeuroNebula() {
  const geo = useMemo(() => {
    const count = 800;
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 0] = (Math.random() - 0.5) * 3.5;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 2.2;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 2.5;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, []);

  const NebulaPoints = ({ geometry }: { geometry: THREE.BufferGeometry }) => {
    const ref = useRef<THREE.Points | null>(null);
    useFrame((state) => {
      if (!ref.current) return;
      ref.current.rotation.y = state.clock.getElapsedTime() * 0.06;
    });
    return (
      <points ref={ref} geometry={geometry}>
        <pointsMaterial color="#6fffe6" size={0.02} sizeAttenuation />
      </points>
    );
  };
  return (
    <Canvas className="syn-r3f" dpr={[1, 2]} camera={{ position: [0, 0, 3.2], fov: 60 }}>
      <color attach="background" args={["#020405"]} />
      <ambientLight intensity={0.6} />
      <NebulaPoints geometry={geo} />
    </Canvas>
  );
}

function ForgeField({ className, heat = 0.5 }: { className?: string; heat?: number }) {
  const particles = useRef(
    Array.from({ length: 460 }, (_, i) => ({
      x: i % 2 === 0 ? Math.random() * 0.4 : 0.6 + Math.random() * 0.4,
      y: Math.random(),
      side: i % 2 === 0 ? -1 : 1,
      v: 0.4 + Math.random() * 0.8,
      life: Math.random(),
    }))
  );
  const ref = useCanvasLoop((ctx, t, w, h) => {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(7, 9, 11, 0.22)";
    ctx.fillRect(0, 0, w, h);
    const cx = w * 0.5;
    const cy = h * 0.5;
    for (const p of particles.current) {
      const px = p.x * w;
      const py = p.y * h;
      const dx = cx - px;
      const dy = cy - py;
      const dist = Math.max(20, Math.hypot(dx, dy));
      const pull = (1 / dist) * p.v * 200;
      p.x += (dx / dist) * pull * 0.001;
      p.y += (dy / dist) * pull * 0.001;
      p.life += 0.006 + heat * 0.004;
      if (dist < 18 || p.life > 1.1) {
        p.x = p.side < 0 ? Math.random() * 0.4 : 0.6 + Math.random() * 0.4;
        p.y = Math.random();
        p.life = 0;
      }
      ctx.strokeStyle =
        p.side < 0
          ? `rgba(84, 180, 255, ${0.4 + heat * 0.4})`
          : `rgba(255, 140, 80, ${0.35 + heat * 0.4})`;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + (dx / dist) * 12, py + (dy / dist) * 12);
      ctx.stroke();
    }
    ctx.fillStyle = `rgba(255, 255, 255, ${0.55 + heat * 0.4})`;
    ctx.beginPath();
    ctx.arc(cx, cy, 10 + Math.sin(t * 2) * 2, 0, Math.PI * 2);
    ctx.fill();
  });
  return <canvas ref={ref} className={className} />;
}

function WorldProjection({ className, intensity = 0.4 }: { className?: string; intensity?: number }) {
  const traces = useRef(
    Array.from({ length: 140 }, (_, i) => ({
      x: Math.random(),
      y: Math.random(),
      drift: 0.1 + Math.random() * 0.6,
      phase: i * 0.2,
    }))
  );
  const ref = useCanvasLoop((ctx, t, w, h) => {
    ctx.clearRect(0, 0, w, h);
    const gradient = ctx.createLinearGradient(0, 0, w, h);
    gradient.addColorStop(0, "rgba(6, 10, 12, 0.6)");
    gradient.addColorStop(1, "rgba(4, 6, 8, 0.85)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = `rgba(120, 255, 215, ${0.08 + intensity * 0.18})`;
    ctx.lineWidth = 1;
    for (let i = 0; i < 16; i += 1) {
      const y = (i / 16) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y + Math.sin(t * 0.4 + i) * 6 * intensity);
      ctx.stroke();
    }

    ctx.strokeStyle = `rgba(255, 180, 120, ${0.12 + intensity * 0.3})`;
    ctx.lineWidth = 1.2;
    for (const p of traces.current) {
      const px = p.x * w;
      const py = p.y * h;
      const sweep = Math.sin(t * (0.4 + intensity) + p.phase) * 40;
      ctx.beginPath();
      ctx.moveTo(px - 40, py + sweep * 0.3);
      ctx.quadraticCurveTo(px, py + sweep, px + 120, py + sweep * 0.6);
      ctx.stroke();
      p.x += 0.001 * p.drift;
      if (p.x > 1.2) {
        p.x = -0.2;
        p.y = Math.random();
      }
    }

    ctx.fillStyle = `rgba(120, 255, 215, ${0.2 + intensity * 0.4})`;
    for (let i = 0; i < 20; i += 1) {
      const x = (i / 20) * w + Math.sin(t * 0.6 + i) * 12;
      const y = h * 0.65 + Math.cos(t * 0.4 + i) * 22;
      ctx.beginPath();
      ctx.arc(x, y, 1.5 + intensity * 1.2, 0, Math.PI * 2);
      ctx.fill();
    }
  });
  return <canvas ref={ref} className={className} />;
}

function ReasoningTree({ intensity = 0 }: { intensity?: number }) {
  const nodes = useMemo(() => {
    const levels = 5;
    const root = { x: 0.5, y: 0.12, depth: 0 };
    const all = [root];
    for (let d = 1; d <= levels; d += 1) {
      const count = 2 + d * 2;
      for (let i = 0; i < count; i += 1) {
        all.push({
          x: 0.16 + (i / (count - 1)) * 0.68 + (Math.random() - 0.5) * 0.05,
          y: 0.12 + (d / levels) * 0.78 + (Math.random() - 0.5) * 0.04,
          depth: d,
        });
      }
    }
    return all;
  }, []);

  return (
    <svg className="syn-tree" viewBox="0 0 100 100" preserveAspectRatio="none">
      {nodes.map((n, i) => {
        if (n.depth === 0) return null;
        const parent = nodes[Math.max(0, Math.floor((i - 1) / 2))];
        return (
          <line
            key={`l-${i}`}
            x1={parent.x * 100}
            y1={parent.y * 100}
            x2={n.x * 100}
            y2={n.y * 100}
            className={i % 5 < intensity ? "branch pruned" : "branch"}
          />
        );
      })}
      {nodes.map((n, i) => (
        <circle
          key={`n-${i}`}
          cx={n.x * 100}
          cy={n.y * 100}
          r={i === 0 ? 3.4 : 1.6}
          className={i % 5 < intensity ? "node pruned" : "node"}
        />
      ))}
    </svg>
  );
}

function RoutingConstellation({ focusIndex = 0 }: { focusIndex?: number }) {
  const experts = ["Scout", "Speed", "Killer", "Tank", "Synth", "Oracle"];
  return (
    <div className="syn-constellation">
      <svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid meet">
        <circle cx="100" cy="100" r="58" className="orbit" />
        <circle cx="100" cy="100" r="12" className="core" />
        {experts.map((label, i) => {
          const angle = (Math.PI * 2 * i) / experts.length;
          const x = 100 + Math.cos(angle) * 70;
          const y = 100 + Math.sin(angle) * 70;
          const active = i === focusIndex;
          return (
            <g key={label} className={active ? "active" : ""}>
              <line x1="100" y1="100" x2={x} y2={y} className="beam" />
              <circle cx={x} cy={y} r={active ? 9.5 : 8} className="node" />
              <text x={x} y={y + 18} textAnchor="middle">
                {label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function DivergenceRing({ level = 0 }: { level?: number }) {
  const spikes = useMemo(() => {
    const list: { a: number; r: number }[] = [];
    for (let i = 0; i < 64; i += 1) {
      list.push({
        a: (i / 64) * Math.PI * 2,
        r: 40 + Math.sin(i * 0.6) * level * 6 + Math.random() * level * 4,
      });
    }
    return list;
  }, [level]);
  const path = spikes
    .map((s, i) => {
      const x = 50 + Math.cos(s.a) * s.r;
      const y = 50 + Math.sin(s.a) * s.r;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg className="syn-divergence" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
      <circle cx="50" cy="50" r="28" className="plan" />
      <path d={`${path} Z`} className="spike" />
    </svg>
  );
}

function SpecStream({ lines }: { lines: string[] }) {
  const repeated = useMemo(() => [...lines, ...lines], [lines]);
  return (
    <div className="spec-stream">
      <div className="spec-marquee">
        {repeated.map((line, i) => (
          <div key={`${line}-${i}`} className="spec-line">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function NeuroSynaptic() {
  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 3000 });
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 3000 });
  const healthQ = useQuery({
    queryKey: ["overviewHealth"],
    queryFn: () => fetchOverviewHealth(240),
    refetchInterval: 5000,
  });
  const issuesQ = useQuery({
    queryKey: ["overviewIssues"],
    queryFn: () => fetchOverviewIssues(600),
    refetchInterval: 6000,
  });

  const workers = workersQ.data ?? {};
  const workerList = Object.values(workers);
  const activeCount = workerList.length;
  const onlineCount = workerList.filter((w) => w.status === "running").length;
  const avgReward =
    workerList.reduce((acc, w) => acc + Number(w.steam_score ?? w.reward ?? 0), 0) /
    Math.max(1, workerList.length);
  const avgEntropy =
    workerList.reduce((acc, w) => acc + Number(w.action_entropy ?? 0), 0) /
    Math.max(1, workerList.length);
  const avgActHz =
    workerList.reduce((acc, w) => acc + Number(w.act_hz ?? 0), 0) /
    Math.max(1, workerList.length);
  const avgObsFps =
    workerList.reduce((acc, w) => acc + Number(w.obs_fps ?? 0), 0) /
    Math.max(1, workerList.length);
  const lastSeen = workerList
    .map((w) => w.ts)
    .filter(Boolean)
    .sort((a, b) => b - a)[0];
  const issues = issuesQ.data ?? [];
  const health = healthQ.data;
  const divergenceLevel = Math.min(1, (issues.length + (health?.heartbeat?.late ?? 0) / 2) / 6);
  const streamTotal = health ? health.stream.ok + health.stream.stale + health.stream.missing : 0;
  const streamOkPct = health && streamTotal ? Math.round((health.stream.ok / streamTotal) * 100) : 0;

  const turbulence = avgEntropy > 0.6 ? "volatile" : avgEntropy > 0.3 ? "turbulent" : "laminar";
  const gatingWeight = Math.max(0.2, Math.min(0.88, 0.7 - avgEntropy * 0.22));
  const focusIndex = onlineCount % 6;

  const nudges = useMemo(
    () => [
      {
        title: "Reward plateau detected",
        detail: "Scout + Speed merge projected +15% efficiency.",
        action: "Simulate TIES merge",
      },
      {
        title: "Liquid instability rising",
        detail: "System 1 time-constant variance spiking in Instance A14.",
        action: "Open river of thought",
      },
      {
        title: "World model drift",
        detail: "Latent manifold divergence at 0.42 rad/s. Verify TTC paths.",
        action: "Inspect Rehoboam ring",
      },
    ],
    []
  );

  const specLines = useMemo(() => {
    const incident = issues[0]?.label ?? "reward_collapse";
    const severity = issues[0]?.severity ?? "medium";
    const coherence = Math.round((1 - divergenceLevel * 0.6) * 100);
    return [
      "{",
      "  \"protocol\": \"A2UI\",",
      `  \"incident\": \"${incident}\",`,
      `  \"severity\": \"${severity}\",`,
      "  \"component\": \"DiagnosticCard\",",
      "  \"panels\": [\"video\", \"reward_curve\", \"policy_diff\"],",
      "  \"actions\": [\"rollback\", \"boost_lr\", \"map_latent\"],",
      `  \"prediction_coherence\": ${coherence},`,
      "  \"stream\": \"realtime\"",
      "}",
    ];
  }, [issues, divergenceLevel]);

  return (
    <div className="syn-shell">
      <div className="syn-noise" />
      <NeuroNebula />
      <div className="syn-scroll">
        <div className="syn-grid">
        <header className="syn-head">
          <div className="syn-brand">
            <span className="syn-brand-title">MetaBonk</span>
            <span className="syn-brand-sub">Neuro-Synaptic Interface</span>
          </div>
          <nav className="syn-nav">
            {NAV.map((n) => (
              <Link key={n.to} to={n.to} className={n.to === "/" ? "active" : ""}>
                {n.label}
              </Link>
            ))}
          </nav>
          <div className="syn-status">
            <div className="syn-kpi">
              <span>System sync</span>
              <strong>{statusQ.isError ? "offline" : "live"}</strong>
            </div>
            <div className="syn-kpi">
              <span>Active agents</span>
              <strong>{activeCount}</strong>
            </div>
            <div className="syn-kpi">
              <span>Reflex cadence</span>
              <strong>{avgActHz ? `${avgActHz.toFixed(1)} hz` : "--"}</strong>
            </div>
            <div className="syn-kpi">
              <span>Last pulse</span>
              <strong>{lastSeen ? timeAgo(lastSeen) : "--"}</strong>
            </div>
          </div>
        </header>

        <section className="syn-left">
          <article className="syn-panel syn-panel-system">
            <div className="syn-panel-header">
              <div>
                <h3>System 1 - Liquid Instinct</h3>
                <p>Liquid time-constant mapped to viscosity and flow.</p>
              </div>
              <div className="syn-panel-metric">
                <span>Entropy</span>
                <strong>{Number.isFinite(avgEntropy) ? avgEntropy.toFixed(2) : "--"}</strong>
              </div>
            </div>
            <div className="syn-panel-body">
              <FluidField className="syn-canvas" hue={190} energy={Math.min(1, avgEntropy)} />
              <div className="syn-overlay">
                <div>
                  <span className="label">Flow state</span>
                  <strong>{turbulence}</strong>
                </div>
                <div>
                  <span className="label">Viscosity</span>
                  <strong>{(1 - Math.min(1, avgEntropy)).toFixed(2)}</strong>
                </div>
              </div>
            </div>
          </article>

          <article className="syn-panel syn-panel-flow">
            <div className="syn-panel-header">
              <div>
                <h3>Mamba Flow Field</h3>
                <p>Selective retention rendered as vector weather.</p>
              </div>
              <div className="syn-panel-metric">
                <span>Selection</span>
                <strong>{gatingWeight.toFixed(2)}</strong>
              </div>
            </div>
            <div className="syn-panel-body">
              <VectorField className="syn-canvas" intensity={avgEntropy} />
              <div className="syn-legend">
                <span className="dot bright" /> Retained
                <span className="dot dim" /> Discarded
              </div>
            </div>
          </article>

          <article className="syn-panel syn-panel-signal">
            <div className="syn-panel-header">
              <div>
                <h3>Reflex Telemetry</h3>
                <p>High-frequency proprioception feed.</p>
              </div>
            </div>
            <div className="syn-mini-grid">
              <div>
                <span>Obs fps</span>
                <strong>{avgObsFps ? avgObsFps.toFixed(1) : "--"}</strong>
              </div>
              <div>
                <span>Act hz</span>
                <strong>{avgActHz ? avgActHz.toFixed(1) : "--"}</strong>
              </div>
              <div>
                <span>Stream ok</span>
                <strong>{health ? `${streamOkPct}%` : "--"}</strong>
              </div>
              <div>
                <span>Noise floor</span>
                <strong>{fmtFixed(Math.max(0, 0.12 + avgEntropy * 0.2), 2)}</strong>
              </div>
            </div>
          </article>
        </section>

        <section className="syn-core">
          <div className="syn-world">
            <WorldProjection className="world-canvas" intensity={divergenceLevel} />
            <div className="world-overlay">
              <div className="world-meta">
                <p className="world-title">World Model Projection</p>
                <h2>Hive Mind Cartography</h2>
                <p className="world-lead">
                  Latent manifolds, test-time compute, and the golden path of the Omega protocol.
                </p>
                <div className="world-chips">
                  <span>coherence {Math.round((1 - divergenceLevel * 0.6) * 100)}%</span>
                  <span>dreams 512 codes</span>
                  <span>ttl {health?.heartbeat.ttl_s ?? 0}s</span>
                </div>
              </div>
              <div className="world-ring">
                <DivergenceRing level={divergenceLevel} />
                <div className="world-ring-meta">
                  <span>Rehoboam divergence</span>
                  <strong>{(divergenceLevel * 100).toFixed(0)}%</strong>
                  <small>prediction delta</small>
                </div>
              </div>
            </div>
          </div>

          <div className="syn-core-row">
            <article className="syn-panel syn-panel-latent">
              <div className="syn-panel-header">
                <div>
                  <h3>Latent Manifold</h3>
                  <p>VQ-VAE codebook mapped into 3D space.</p>
                </div>
                <div className="syn-panel-metric">
                  <span>Dream drift</span>
                  <strong>{(divergenceLevel * 0.9 + 0.12).toFixed(2)}</strong>
                </div>
              </div>
              <div className="syn-panel-body">
                <LatentField className="syn-canvas" drift={divergenceLevel} />
                <div className="syn-legend">
                  <span className="dot loop" /> Loop
                  <span className="dot drift" /> Drift
                </div>
              </div>
            </article>

            <article className="syn-panel syn-panel-orchestrator">
              <div className="syn-panel-header">
                <div>
                  <h3>Generative UI Forge</h3>
                  <p>Context-aware cards streamed via A2UI.</p>
                </div>
                <div className="syn-panel-metric">
                  <span>Spec rate</span>
                  <strong>{fmtFixed(0.12 + avgEntropy * 0.3, 2)} hz</strong>
                </div>
              </div>
              <SpecStream lines={specLines} />
              <div className="syn-actions">
                <button className="syn-action">Rollback weights</button>
                <button className="syn-action ghost">Boost learning rate</button>
                <button className="syn-action">Visualize latent space</button>
              </div>
            </article>
          </div>
        </section>

        <section className="syn-right">
          <article className="syn-panel syn-panel-system2">
            <div className="syn-panel-header">
              <div>
                <h3>System 2 - Crystalline Reason</h3>
                <p>Reasoning trees, pruning, and TTC cycles.</p>
              </div>
              <div className="syn-panel-metric">
                <span>Pruning</span>
                <strong>{issues.length}</strong>
              </div>
            </div>
            <div className="syn-panel-body">
              <ReasoningTree intensity={Math.min(4, issues.length)} />
              <div className="syn-meta">
                <div>
                  <span>Beam width</span>
                  <strong>7</strong>
                </div>
                <div>
                  <span>Reflexion</span>
                  <strong>3 cycles</strong>
                </div>
                <div>
                  <span>Cadence</span>
                  <strong>0.5 hz</strong>
                </div>
              </div>
            </div>
          </article>

          <article className="syn-panel syn-panel-routing">
            <div className="syn-panel-header">
              <div>
                <h3>Mixture-of-Reasonings</h3>
                <p>Expert routing constellation.</p>
              </div>
              <div className="syn-panel-metric">
                <span>Active cores</span>
                <strong>{Math.min(6, Math.max(2, onlineCount))}</strong>
              </div>
            </div>
            <div className="syn-panel-body">
              <RoutingConstellation focusIndex={focusIndex} />
              <div className="syn-meta-line">
                <span>Router load</span>
                <strong>{fmtFixed(Math.min(0.98, 0.5 + avgEntropy * 0.4), 2)}</strong>
              </div>
            </div>
          </article>

          <article className="syn-panel syn-panel-nudge">
            <div className="syn-panel-header">
              <div>
                <h3>Agent-to-User Proposals</h3>
                <p>Proactive nudges from the Orchestrator.</p>
              </div>
            </div>
            <div className="syn-nudges">
              {nudges.map((n) => (
                <div key={n.title} className="syn-nudge">
                  <h4>{n.title}</h4>
                  <p>{n.detail}</p>
                  <button>{n.action}</button>
                </div>
              ))}
            </div>
          </article>
        </section>

        <section className="syn-lower">
          <article className="syn-panel syn-panel-swarm">
            <div className="syn-panel-header">
              <div>
                <h3>Swarm Tactical Overlay</h3>
                <p>Orthographic projection of federated agents.</p>
              </div>
              <div className="syn-panel-metric">
                <span>Agents online</span>
                <strong>{onlineCount}</strong>
              </div>
            </div>
            <div className="syn-swarm-grid">
              <div className="syn-swarm-map">
                <span className="ping" />
                <span className="ping ping-alt" />
                <span className="path" />
                <span className="path path-2" />
                {Array.from({ length: 34 }).map((_, i) => (
                  <span
                    key={i}
                    className="agent"
                    style={{
                      left: `${8 + (i * 12) % 84}%`,
                      top: `${6 + (i * 17) % 82}%`,
                    }}
                  />
                ))}
              </div>
              <div className="syn-swarm-metrics">
                <div>
                  <span>Recon</span>
                  <strong>{Math.max(0, 12 - issues.length)}</strong>
                </div>
                <div>
                  <span>Assault</span>
                  <strong>{Math.max(3, Math.round(activeCount / 2))}</strong>
                </div>
                <div>
                  <span>Support</span>
                  <strong>{Math.max(2, Math.round(activeCount / 3))}</strong>
                </div>
                <div>
                  <span>Shared grad</span>
                  <strong>{fmtFixed(0.4 + avgEntropy * 0.3, 2)}</strong>
                </div>
              </div>
            </div>
          </article>

          <article className="syn-panel syn-panel-merge">
            <div className="syn-panel-header">
              <div>
                <h3>TIES Merge Forge</h3>
                <p>Particle collision and sign election.</p>
              </div>
              <div className="syn-panel-metric">
                <span>Merge energy</span>
                <strong>{fmtFixed((avgReward || 0) * 0.12 + 0.88, 2)}</strong>
              </div>
            </div>
            <div className="syn-panel-body">
              <ForgeField className="syn-canvas" heat={Math.min(1, avgEntropy + 0.2)} />
              <div className="syn-meta-line">
                <span>Trim loss</span>
                <strong>{fmtFixed(Math.min(0.4, 0.18 + avgEntropy * 0.2), 2)}</strong>
              </div>
            </div>
          </article>

          <article className="syn-panel syn-panel-issues">
            <div className="syn-panel-header">
              <div>
                <h3>Active Divergences</h3>
                <p>Negative work: rejected paths and fractures.</p>
              </div>
            </div>
            <div className="syn-issues">
              {issues.length === 0 ? (
                <div className="syn-issue-empty">No violations in the last 10 minutes.</div>
              ) : (
                issues.slice(0, 6).map((issue) => (
                  <div key={issue.id} className="syn-issue">
                    <div>
                      <strong>{issue.label}</strong>
                      <span>{issue.hint ?? "Anomaly detected"}</span>
                    </div>
                    <em>{issue.last_seen ? timeAgo(issue.last_seen) : "recent"}</em>
                  </div>
                ))
              )}
            </div>
          </article>
        </section>

        <footer className="syn-footer">
          <div>
            <span className="label">Cluster status</span>
            <strong>{statusQ.isError ? "degraded" : "stable"}</strong>
          </div>
          <div>
            <span className="label">Health laggards</span>
            <strong>{health?.laggards ?? 0}</strong>
          </div>
          <div>
            <span className="label">Stream integrity</span>
            <strong>{health ? `${streamOkPct}%` : "--"}</strong>
          </div>
          <div>
            <span className="label">API error rate</span>
            <strong>{health ? `${(health.api.error_rate * 100).toFixed(1)}%` : "--"}</strong>
          </div>
        </footer>
        </div>
      </div>
    </div>
  );
}
