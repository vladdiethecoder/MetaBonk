import { useEffect, useRef } from "react";
import type { Heartbeat } from "../api";
import MseMp4Video from "./MseMp4Video";
import Go2rtcEmbed from "./Go2rtcEmbed";

export type MonitorEffect = "static" | "bars" | "dvd" | null;

type MonitorProps = {
  agent: Heartbeat | null;
  label: string;
  isHero?: boolean;
  effect: MonitorEffect;
};

export default function CctvMonitor({ agent, label, isHero, effect }: MonitorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dvdState = useRef({ x: 40, y: 60, vx: 2.4, vy: 2.1, colorIdx: 0 });
  const frameRef = useRef(0);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const rect = el.parentElement?.getBoundingClientRect();
      if (!rect) return;
      el.width = Math.max(1, Math.floor(rect.width));
      el.height = Math.max(1, Math.floor(rect.height));
    });
    if (el.parentElement) ro.observe(el.parentElement);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = canvasRef.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    const colors = ["#ff4040", "#38ff6f", "#52a9ff", "#ffd65a", "#f472d0"];
    let raf = 0;

    const draw = () => {
      const w = el.width || 1;
      const h = el.height || 1;
      frameRef.current += 1;
      ctx.clearRect(0, 0, w, h);

      // Scanlines (always).
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      for (let y = 0; y < h; y += 4) ctx.fillRect(0, y, w, 2);

      if (effect === "static") {
        const img = ctx.createImageData(w, h);
        for (let i = 0; i < img.data.length; i += 4) {
          const v = Math.random() * 255;
          img.data[i] = v;
          img.data[i + 1] = v;
          img.data[i + 2] = v;
          img.data[i + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = "#d7f5f0";
        ctx.font = "bold 36px 'Space Mono', ui-monospace, monospace";
        ctx.fillText("NO SIGNAL", Math.max(16, w * 0.32), h * 0.55);
      } else if (effect === "bars") {
        const barColors = ["#c0c0c0", "#c0c000", "#00c0c0", "#00c000", "#c000c0", "#c00000", "#0000c0"];
        const bw = w / barColors.length;
        barColors.forEach((c, i) => {
          ctx.fillStyle = c;
          ctx.fillRect(i * bw, 0, bw + 1, h);
        });
      } else if (effect === "dvd") {
        const st = dvdState.current;
        st.x += st.vx * 6;
        st.y += st.vy * 6;
        if (st.x <= 16 || st.x >= w - 90) {
          st.vx *= -1;
          st.colorIdx = (st.colorIdx + 1) % colors.length;
        }
        if (st.y <= 36 || st.y >= h - 36) {
          st.vy *= -1;
          st.colorIdx = (st.colorIdx + 1) % colors.length;
        }
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = colors[st.colorIdx];
        ctx.font = "bold 42px 'Space Mono', ui-monospace, monospace";
        ctx.fillText("DVD", st.x, st.y);
      } else {
        const now = new Date();
        const ms = Math.floor(now.getMilliseconds() / 10)
          .toString()
          .padStart(2, "0");
        const ts = `${now.toLocaleTimeString()}:${ms}`;
        ctx.font = "16px 'Space Mono', ui-monospace, monospace";
        ctx.fillStyle = "rgba(80,255,200,0.8)";
        ctx.fillText(label, 18, 26);
        ctx.fillText(ts, Math.max(18, w - 170), 26);
        if (frameRef.current % 60 < 30) {
          ctx.fillStyle = "#ff3b55";
          ctx.beginPath();
          ctx.arc(w - 26, h - 26, 7, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#fefefe";
          ctx.font = "12px 'Space Mono', ui-monospace, monospace";
          ctx.fillText("REC", w - 62, h - 22);
        }
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [effect, label]);

  const streamUrl = String(agent?.stream_url ?? "");
  const controlUrl = String((agent as any)?.control_url ?? "");
  const fallbackUrl = controlUrl ? `${controlUrl.replace(/\\/+$/, "")}/frame.jpg` : undefined;
  const instanceId = String((agent as any)?.instance_id ?? "");
  const go2rtcBase = String((agent as any)?.go2rtc_base_url ?? "").trim();
  const go2rtcName = String((agent as any)?.go2rtc_stream_name ?? "").trim();
  const go2rtcUrl =
    go2rtcBase && go2rtcName
      ? `${go2rtcBase.replace(/\\/+$/, "")}/stream.html?${new URLSearchParams({ src: go2rtcName, mode: "webrtc" }).toString()}`
      : "";
  const useGo2rtc = Boolean(go2rtcUrl);

  return (
    <div className={`cctv-monitor ${isHero ? "hero" : ""}`}>
      <div className="cctv-screen">
        {useGo2rtc && !effect ? (
          <Go2rtcEmbed url={go2rtcUrl} className="cctv-video" title={`go2rtc ${instanceId || label}`} />
        ) : streamUrl && !effect ? (
          <MseMp4Video url={streamUrl} className="cctv-video" fallbackUrl={fallbackUrl} exclusiveKey={instanceId || streamUrl} />
        ) : (
          <div className="cctv-noise-bg" />
        )}
        <canvas ref={canvasRef} className="cctv-overlay" />
      </div>
      <div className="cctv-bezel">
        <span className="cctv-brand">SONY</span>
      </div>
    </div>
  );
}
