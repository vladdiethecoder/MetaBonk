import { useEffect, useMemo, useRef, useState } from "react";
import useTauriEvent from "../hooks/useTauriEvent";
import { isTauri } from "../lib/tauri";

type TelemetryPacket = {
  ts?: number;
  cpu?: { usage_pct?: number; cores?: number };
  memory?: { used_mb?: number; total_mb?: number };
  gpu?: { util_pct?: number | null; mem_used_mb?: number | null; mem_total_mb?: number | null; temp_c?: number | null } | null;
};

type AnticipatoryProps = {
  windowSeconds?: number;
  onTelemetry?: (t: TelemetryPacket) => void;
};

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function sparkPath(values: number[], width: number, height: number) {
  const vs = values.filter((v) => Number.isFinite(v));
  if (!vs.length) return "";
  const lo = Math.min(...vs);
  const hi = Math.max(...vs);
  const span = Math.max(1e-9, hi - lo);
  const n = values.length;
  const scaleX = n <= 1 ? 0 : width / (n - 1);
  const pts = values.map((v, i) => {
    const x = i * scaleX;
    const t = clamp01((v - lo) / span);
    const y = (1 - t) * height;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return pts.join(" ");
}

function ewma(values: number[], alpha: number) {
  let s = 0;
  let init = false;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (!init) {
      s = v;
      init = true;
    } else {
      s = alpha * v + (1 - alpha) * s;
    }
  }
  return init ? s : 0;
}

export default function Anticipatory({ windowSeconds = 60, onTelemetry }: AnticipatoryProps) {
  const [samples, setSamples] = useState<TelemetryPacket[]>([]);
  const lastTsRef = useRef<number>(0);

  useTauriEvent<TelemetryPacket>("system-telemetry", (payload) => {
    const ts = Number(payload?.ts ?? 0);
    if (Number.isFinite(ts) && ts <= lastTsRef.current) return;
    lastTsRef.current = ts;
    setSamples((prev) => {
      const next = [...prev, payload].slice(-Math.max(10, Math.round(windowSeconds)));
      return next;
    });
    onTelemetry?.(payload);
  });

  useEffect(() => {
    if (isTauri()) return;
    // Browser-mode demo feed: emit synthetic telemetry so the UI still shows motion.
    let t = 0;
    const handle = window.setInterval(() => {
      t += 1;
      const cpu = 20 + 20 * Math.sin(t / 7) + 8 * Math.sin(t / 3.7);
      const gpu = 10 + 40 * Math.max(0, Math.sin(t / 9));
      const payload: TelemetryPacket = {
        ts: Date.now() / 1000,
        cpu: { usage_pct: cpu, cores: 12 },
        memory: { used_mb: 4096 + 256 * Math.sin(t / 11), total_mb: 32768 },
        gpu: { util_pct: gpu, mem_used_mb: 4200 + 300 * Math.sin(t / 13), mem_total_mb: 16384, temp_c: 42 + 18 * Math.sin(t / 10) },
      };
      setSamples((prev) => [...prev, payload].slice(-Math.max(10, Math.round(windowSeconds))));
      onTelemetry?.(payload);
    }, 1000);
    return () => window.clearInterval(handle);
  }, [onTelemetry, windowSeconds]);

  const cpuSeries = useMemo(() => samples.map((s) => Number(s.cpu?.usage_pct ?? 0)), [samples]);
  const gpuSeries = useMemo(() => samples.map((s) => Number(s.gpu?.util_pct ?? 0)), [samples]);
  const cpuNow = cpuSeries.length ? cpuSeries[cpuSeries.length - 1] : 0;
  const gpuNow = gpuSeries.length ? gpuSeries[gpuSeries.length - 1] : 0;
  const cpuForecast = useMemo(() => ewma(cpuSeries.slice(-12), 0.35), [cpuSeries]);
  const gpuForecast = useMemo(() => ewma(gpuSeries.slice(-12), 0.35), [gpuSeries]);

  const cpuAvg = useMemo(() => ewma(cpuSeries, 0.15), [cpuSeries]);
  const gpuAvg = useMemo(() => ewma(gpuSeries, 0.15), [gpuSeries]);
  const anomaly = useMemo(() => {
    const dc = Math.abs(cpuNow - cpuAvg) / 100;
    const dg = Math.abs(gpuNow - gpuAvg) / 100;
    return clamp01(0.15 + 0.85 * (0.55 * dc + 0.45 * dg));
  }, [cpuAvg, cpuNow, gpuAvg, gpuNow]);

  const last = samples.length ? samples[samples.length - 1] : null;
  const memUsed = Number(last?.memory?.used_mb ?? 0);
  const memTotal = Number(last?.memory?.total_mb ?? 0);
  const memPct = memTotal > 0 ? (memUsed / memTotal) * 100 : 0;

  return (
    <div>
      <div className="card-header">
        <div>
          <h3>Anticipatory Layer</h3>
          <p className="muted">Lightweight forecasting from live system telemetry (“predict the substrate”).</p>
        </div>
        <div className={`badge ${anomaly > 0.7 ? "warn" : ""}`}>anomaly {Math.round(anomaly * 100)}%</div>
      </div>
      <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="lab-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
          <div className="card" style={{ padding: 10 }}>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
              cpu
            </div>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontWeight: 900, fontSize: 18 }}>{Math.round(cpuNow)}%</div>
              <div className="muted" style={{ fontSize: 12 }}>
                next ≈ {Math.round(cpuForecast)}%
              </div>
            </div>
            <svg width="100%" height="48" viewBox="0 0 160 48" style={{ marginTop: 6 }}>
              <path d={sparkPath(cpuSeries.slice(-32), 160, 48)} fill="none" stroke="rgba(0,229,255,.9)" strokeWidth="2" />
            </svg>
          </div>
          <div className="card" style={{ padding: 10 }}>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
              gpu
            </div>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontWeight: 900, fontSize: 18 }}>{Math.round(gpuNow)}%</div>
              <div className="muted" style={{ fontSize: 12 }}>
                next ≈ {Math.round(gpuForecast)}%
              </div>
            </div>
            <svg width="100%" height="48" viewBox="0 0 160 48" style={{ marginTop: 6 }}>
              <path d={sparkPath(gpuSeries.slice(-32), 160, 48)} fill="none" stroke="rgba(255,45,141,.9)" strokeWidth="2" />
            </svg>
          </div>
          <div className="card" style={{ padding: 10 }}>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
              memory
            </div>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontWeight: 900, fontSize: 18 }}>{Math.round(memPct)}%</div>
              <div className="muted" style={{ fontSize: 12 }}>
                {Math.round(memUsed)} / {Math.round(memTotal)} MB
              </div>
            </div>
            <div style={{ height: 10, borderRadius: 999, background: "rgba(255,255,255,.08)", overflow: "hidden", marginTop: 10 }}>
              <div style={{ width: `${clamp01(memPct / 100) * 100}%`, height: "100%", background: "rgba(255,230,0,.9)" }} />
            </div>
          </div>
        </div>

        <div className="muted" style={{ fontSize: 12 }}>
          source: <span className="mono">{isTauri() ? "tauri system-telemetry" : "browser simulation"}</span>
        </div>
      </div>
    </div>
  );
}

