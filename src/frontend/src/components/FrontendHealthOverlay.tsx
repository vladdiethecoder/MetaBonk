import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { bumpWebglReset } from "../hooks/useWebglReset";
import { getWebglLostStats, useWebglCounter } from "../hooks/useWebglCounter";

export default function FrontendHealthOverlay() {
  const loc = useLocation();
  const enabled = useMemo(() => {
    if (!import.meta.env.DEV) return false;
    const qs = new URLSearchParams(loc.search);
    return qs.get("debug") === "1";
  }, [loc.search]);
  const webglCount = useWebglCounter();
  const qc = useQueryClient();
  const [showAll, setShowAll] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const id = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(id);
  }, [enabled]);

  if (!enabled) return null;
  const all = qc.getQueryCache().getAll();
  const failed = all.filter((q) => q.state.status === "error");
  const fetching = all.filter((q) => q.state.fetchStatus === "fetching");
  const visibleQueries = showAll ? all : failed;
  const { count: webglLost, lastLostAt } = getWebglLostStats();
  const pageSize = (window as any).__mbPageSize ?? null;
  const debug = {
    route: loc.pathname,
    size: pageSize ? `${Math.round(pageSize.width)}x${Math.round(pageSize.height)}` : "unknown",
    webgl: webglCount,
    webglLost,
    failedQueries: failed.map((q) => String(q.queryHash)),
  };

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
        <span className="health-label">webgl lost</span>
        <span>{webglLost ? `${webglLost} (${lastLostAt ? new Date(lastLostAt).toLocaleTimeString() : "—"})` : "0"}</span>
      </div>
      <div className="health-row">
        <span className="health-label">page</span>
        <span>{pageSize ? `${Math.round(pageSize.width)}×${Math.round(pageSize.height)}` : "—"}</span>
      </div>
      <div className="health-row">
        <span className="health-label">fetching</span>
        <span>{fetching.length}</span>
      </div>
      <div className="health-section">queries</div>
      <div className="health-list">
        {visibleQueries.length === 0 ? (
          <div className="muted">no query state</div>
        ) : (
          visibleQueries.map((q) => {
            const status = q.state.status === "error" ? "error" : q.state.fetchStatus === "fetching" ? "loading" : "ok";
            return (
              <div key={q.queryHash} className="health-row">
                <span className="health-label">{String(q.queryKey[0] ?? q.queryHash)}</span>
                <span className={status === "error" ? "health-bad" : status === "loading" ? "health-warn" : "health-ok"}>{status}</span>
              </div>
            );
          })
        )}
      </div>
      <div className="health-actions">
        <button className="btn btn-ghost" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "show errors" : "show all"}
        </button>
        <button className="btn btn-ghost" onClick={() => bumpWebglReset()}>
          reset webgl
        </button>
        <button
          className="btn btn-ghost"
          onClick={() => {
            try {
              navigator.clipboard.writeText(JSON.stringify(debug, null, 2));
            } catch {}
          }}
        >
          copy debug
        </button>
      </div>
      <div className="health-foot">tick {tick}</div>
    </div>
  );
}
