import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useMseVideoTexture } from "../hooks/useMseVideoTexture";

export type MonitorEffect = "static" | "bars" | "dvd" | null;

type CrtProps = {
  position: [number, number, number];
  rotation?: [number, number, number];
  scale?: number;
  streamUrl?: string | null;
  label: string;
  isHero?: boolean;
  color?: string;
  effect?: MonitorEffect;
};

function buildLabelTexture(text: string, color: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(0,0,0,0)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "20px 'Space Mono', ui-monospace, monospace";
    ctx.fillStyle = color;
    ctx.fillText(text, 6, 38);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function useInterstitialTexture(effect: MonitorEffect) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const textureRef = useRef<THREE.CanvasTexture | null>(null);
  const dvdState = useRef({ x: 40, y: 60, vx: 3.1, vy: 2.4, colorIdx: 0 });

  if (!canvasRef.current) {
    const cvs = document.createElement("canvas");
    cvs.width = 512;
    cvs.height = 320;
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
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#d8ffe9";
      ctx.font = "bold 42px 'Space Mono', ui-monospace, monospace";
      ctx.fillText("NO SIGNAL", w * 0.22, h * 0.55);
    } else if (effect === "bars") {
      const colors = ["#c0c0c0", "#c0c000", "#00c0c0", "#00c000", "#c000c0", "#c00000", "#0000c0"];
      const bw = w / colors.length;
      colors.forEach((c, i) => {
        ctx.fillStyle = c;
        ctx.fillRect(i * bw, 0, bw + 1, h);
      });
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

export function CrtModel({
  position,
  rotation = [0, 0, 0],
  scale = 1,
  streamUrl,
  label,
  isHero,
  color = "#7EE7FF",
  effect = null,
}: CrtProps) {
  const texture = useMseVideoTexture(streamUrl || undefined);
  const effectTex = useInterstitialTexture(effect);
  const screenGeo = useMemo(() => new THREE.SphereGeometry(1, 32, 32, 0, Math.PI, 0, Math.PI), []);
  const boxGeo = useMemo(() => new THREE.BoxGeometry(1.05, 0.85, 0.6), []);
  const labelTex = useMemo(() => buildLabelTexture(label, color), [label, color]);

  const map = effect ? effectTex : texture;

  return (
    <group position={position} rotation={rotation} scale={scale}>
      <mesh geometry={boxGeo} position={[0, 0, 0]}>
        <meshStandardMaterial color="#111" roughness={0.6} metalness={0.4} />
      </mesh>

      <mesh position={[0, 0, 0.28]} rotation={[0, 0, 0]} scale={[0.55, 0.45, 0.5]} geometry={screenGeo}>
        {map ? (
          <meshStandardMaterial
            map={map}
            emissive="white"
            emissiveMap={map}
            emissiveIntensity={isHero ? 2.5 : 1.2}
            toneMapped={false}
            roughness={0.2}
            metalness={0.8}
          />
        ) : (
          <meshBasicMaterial color="#020202" />
        )}
      </mesh>

      <mesh position={[0.42, -0.35, 0.31]}>
        <circleGeometry args={[0.012]} />
        <meshBasicMaterial color={streamUrl ? "#00ffaa" : "#ff3300"} toneMapped={false} />
      </mesh>

      <mesh position={[-0.46, -0.5, 0.4]}>
        <planeGeometry args={[0.6, 0.12]} />
        <meshBasicMaterial map={labelTex} transparent toneMapped={false} />
      </mesh>
    </group>
  );
}
