import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { useWebglCounter } from "../hooks/useWebglCounter";

export default function FrontendHealthOverlay() {
  const loc = useLocation();
  const enabled = useMemo(() => {
    if (!import.meta.env.DEV) return false;
    const qs = new URLSearchParams(loc.search);
    return qs.get("debug") === "1";
  }, [loc.search]);
  const webglCount = useWebglCounter();
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, [enabled]);

  if (!enabled) return null;
  const queryStates = (window as any).__mbQueryStates ?? {};
  const pageSize = (window as any).__mbPageSize ?? null;

  return (
    <div className="health-overlay">
      <div className="health-title">Frontend Health</div>
      <div className="health-row">
        <span className="health-label">route</span>
        <span>{loc.pathname}</span>
      </div>
      <div className="health-row">
        <span className="health-label">webgl</span>
        <span>{webglCount}</span>
      </div>
      <div className="health-row">
        <span className="health-label">page</span>
        <span>{pageSize ? `${Math.round(pageSize.width)}×${Math.round(pageSize.height)}` : "—"}</span>
      </div>
      <div className="health-section">queries</div>
      <div className="health-list">
        {Object.keys(queryStates).length === 0 ? (
          <div className="muted">no query state</div>
        ) : (
          Object.entries(queryStates).map(([k, v]: any) => (
            <div key={k} className="health-row">
              <span className="health-label">{k}</span>
              <span className={v.error ? "health-bad" : v.loading ? "health-warn" : "health-ok"}>
                {v.error ? "error" : v.loading ? "loading" : "ok"}
              </span>
            </div>
          ))
        )}
      </div>
      <div className="health-foot">tick {tick}</div>
    </div>
  );
}
