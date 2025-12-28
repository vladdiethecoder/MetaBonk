import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import PageShell from "../components/PageShell";
import QueryStateGate from "../components/QueryStateGate";
import { fetchClips, type ClipRecord } from "../api";
import { fmtFixed, timeAgo } from "../lib/format";

const DEFAULT_ORCH = "/api";

const resolveOrchBase = () => {
  if (typeof window === "undefined") return DEFAULT_ORCH;
  const w = window as any;
  const explicit = typeof w.__MB_ORCH_URL__ === "string" ? String(w.__MB_ORCH_URL__) : "";
  if (explicit) return explicit.replace(/\/+$/, "");
  const envUrl = (import.meta as any)?.env?.VITE_ORCH_URL;
  if (envUrl) return String(envUrl).replace(/\/+$/, "");
  if ("__TAURI__" in w) return "http://127.0.0.1:8040";
  return DEFAULT_ORCH;
};

const clipHref = (orchBase: string, clipUrl: string) => {
  const base = String(orchBase || DEFAULT_ORCH).replace(/\/+$/, "");
  const raw = String(clipUrl || "").trim();
  if (!raw) return "";
  if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
  if (raw.startsWith("/api")) return raw;
  if (raw.startsWith("/")) return `${base}${raw}`;
  return `${base}/${raw}`;
};

export default function Clips() {
  const orchBase = useMemo(() => resolveOrchBase(), []);
  const [tag, setTag] = useState<string>("");
  const [q, setQ] = useState<string>("");
  const [before, setBefore] = useState<number | null>(null);
  const [items, setItems] = useState<ClipRecord[]>([]);

  const clipsQ = useQuery({
    queryKey: ["clips", tag, before],
    queryFn: () =>
      fetchClips({
        limit: 40,
        before,
        tag: tag || undefined,
      }),
    refetchInterval: 8000,
  });

  useEffect(() => {
    const next = clipsQ.data?.items ?? [];
    setItems((prev) => (before ? [...prev, ...next] : next));
  }, [clipsQ.data?.items, before]);

  useEffect(() => {
    setBefore(null);
  }, [tag]);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    if (!qq) return items;
    return items.filter((it) => {
      const hay = `${it.tag ?? ""} ${it.run_id ?? ""} ${it.worker_id ?? ""} ${it.agent_name ?? ""} ${it.seed ?? ""}`.toLowerCase();
      return hay.includes(qq);
    });
  }, [items, q]);

  const canLoadMore = Boolean(clipsQ.data?.next_before);

  return (
    <PageShell className="clips-page">
      <div className="card">
        <div className="row-between">
          <h1>Clips</h1>
          <div className="muted">retention via `scripts/prune_artifacts.py`</div>
        </div>
        <div className="row" style={{ gap: 10, flexWrap: "wrap", marginTop: 10 }}>
          <label className="row" style={{ gap: 8 }}>
            <span className="muted">tag</span>
            <select value={tag} onChange={(e) => setTag(e.target.value)}>
              <option value="">all</option>
              <option value="pb">pb</option>
              <option value="clutch">clutch</option>
              <option value="attract_fame">attract_fame</option>
              <option value="attract_shame">attract_shame</option>
            </select>
          </label>
          <label className="row" style={{ gap: 8, flex: 1, minWidth: 240 }}>
            <span className="muted">search</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="run_id / worker / agent / seed…" style={{ flex: 1 }} />
          </label>
          <button
            className="btn btn-ghost"
            onClick={() => {
              setBefore(null);
              setItems([]);
            }}
          >
            refresh
          </button>
        </div>
      </div>

      <QueryStateGate label="Clips" queries={[clipsQ]}>
        <div className="card" style={{ marginTop: 12 }}>
          <div className="card-header">
            <div>
              <h3>Archive</h3>
              <p className="muted">{filtered.length} items</p>
            </div>
          </div>
          <div className="card-body">
            {filtered.length ? (
              <div className="table" style={{ width: "100%" }}>
                <div className="tr th">
                  <div className="td">time</div>
                  <div className="td">tag</div>
                  <div className="td">agent</div>
                  <div className="td numeric">score</div>
                  <div className="td numeric">dur</div>
                  <div className="td">clip</div>
                </div>
                {filtered.slice(0, 200).map((it) => {
                  const href = clipHref(orchBase, String(it.clip_url ?? ""));
                  return (
                    <div key={`${it.clip_url}|${it.timestamp}`} className="tr">
                      <div className="td muted">{timeAgo(it.timestamp)}</div>
                      <div className="td mono">{it.tag ?? "—"}</div>
                      <div className="td">{it.agent_name ?? it.worker_id ?? "—"}</div>
                      <div className="td numeric">{it.final_score != null ? fmtFixed(it.final_score, 2) : "—"}</div>
                      <div className="td numeric">{it.match_duration_sec != null ? `${Math.round(it.match_duration_sec)}s` : "—"}</div>
                      <div className="td">
                        {href ? (
                          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                            <a className="btn btn-ghost" href={href} target="_blank" rel="noreferrer">
                              open
                            </a>
                            <a className="btn btn-ghost" href={href} download>
                              download
                            </a>
                          </div>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="muted">no clips yet</div>
            )}

            <div className="row" style={{ gap: 10, marginTop: 12 }}>
              <button
                className="btn btn-ghost"
                disabled={!canLoadMore}
                onClick={() => {
                  const next = clipsQ.data?.next_before ?? null;
                  if (next) setBefore(next);
                }}
              >
                load more
              </button>
              {clipsQ.data?.next_before ? <span className="muted mono">cursor: {Math.round(Number(clipsQ.data.next_before))}</span> : null}
            </div>
          </div>
        </div>
      </QueryStateGate>
    </PageShell>
  );
}

