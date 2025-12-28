import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type TelemetryLite = {
  cpu?: { usage_pct?: number };
  gpu?: { util_pct?: number; temp_c?: number };
};

type HolographicRendererProps = {
  seed: number;
  telemetry?: TelemetryLite | null;
};

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function Field({ seed, telemetry }: HolographicRendererProps) {
  const group = useRef<THREE.Group | null>(null);
  const { geometry, lines } = useMemo(() => {
    const rand = mulberry32(seed >>> 0);
    const n = 420;
    const positions = new Float32Array(n * 3);
    const points: Array<[number, number, number]> = [];
    for (let i = 0; i < n; i++) {
      const r = Math.pow(rand(), 0.35) * 1.6;
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
      points.push([x, y, z]);
      positions[i * 3 + 0] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const edges: number[] = [];
    for (let i = 0; i < n; i++) {
      if (rand() > 0.08) continue;
      const a = i;
      const b = Math.floor(rand() * n);
      edges.push(a, b);
    }
    const linePos = new Float32Array(edges.length * 3);
    for (let i = 0; i < edges.length; i++) {
      const p = points[edges[i]];
      linePos[i * 3 + 0] = p[0];
      linePos[i * 3 + 1] = p[1];
      linePos[i * 3 + 2] = p[2];
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    return { geometry: g, lines: lg };
  }, [seed]);

  const cpu = clamp01(Number(telemetry?.cpu?.usage_pct ?? 0) / 100);
  const gpu = clamp01(Number(telemetry?.gpu?.util_pct ?? 0) / 100);
  const heat = clamp01((Number(telemetry?.gpu?.temp_c ?? 40) - 40) / 40);
  const spin = 0.18 + 0.65 * gpu + 0.25 * cpu;

  useFrame((state, delta) => {
    const g = group.current;
    if (!g) return;
    g.rotation.y += delta * spin * 0.35;
    g.rotation.x += delta * spin * 0.15;
    g.position.z = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
  });

  const pointColor = new THREE.Color().setHSL(0.82 - heat * 0.22, 0.95, 0.62);
  const lineColor = new THREE.Color().setHSL(0.55, 0.95, 0.55);

  return (
    <group ref={group}>
      <lineSegments geometry={lines}>
        <lineBasicMaterial color={lineColor} transparent opacity={0.25 + 0.35 * gpu} />
      </lineSegments>
      <points geometry={geometry}>
        <pointsMaterial
          size={0.02 + 0.02 * gpu}
          sizeAttenuation
          color={pointColor}
          transparent
          opacity={0.65 + 0.25 * cpu}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

export default function HolographicRenderer({ seed, telemetry }: HolographicRendererProps) {
  return (
    <div style={{ height: 360, borderRadius: 14, overflow: "hidden", border: "1px solid rgba(255,255,255,.08)" }}>
      <Canvas camera={{ position: [0, 0, 3.2], fov: 55 }}>
        <color attach="background" args={["#07030d"]} />
        <ambientLight intensity={0.65} />
        <pointLight position={[4, 3, 4]} intensity={1.2} color="#00e5ff" />
        <pointLight position={[-3, -2, -2]} intensity={0.7} color="#ff2d8d" />
        <Field seed={seed} telemetry={telemetry} />
      </Canvas>
    </div>
  );
}

