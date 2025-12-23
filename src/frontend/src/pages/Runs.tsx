import { useQuery } from "@tanstack/react-query";
import { useMemo, useState, useRef, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import VirtualList, { type ListChildComponentProps } from "../components/VirtualList";
import AutoSizer from "react-virtualized-auto-sizer";
import { fetchRuns, fetchRunsCompare, Run, RunCompareResponse, RunMetricSeries } from "../api";
import useContextFilters from "../hooks/useContextFilters";
import { copyToClipboard, fmtFixed, fmtNum, timeAgo } from "../lib/format";
import { useLocation, useNavigate } from "react-router-dom";

function RunsCoreViz() {
  const CoreMesh = () => {
    const ref = useRef<THREE.Mesh | null>(null);
    useFrame((state) => {
      if (!ref.current) return;
      const t = state.clock.getElapsedTime();
      ref.current.rotation.x = t * 0.25;
      ref.current.rotation.y = t * 0.35;
    });
    return (
      <mesh ref={ref}>
        <dodecahedronGeometry args={[1, 0]} />
        <meshBasicMaterial wireframe color="#7bffe6" transparent opacity={0.45} />
      </mesh>
    );
  };
  return (
    <Canvas className="runs-core-canvas" dpr={[1, 2]} camera={{ position: [0, 0, 3], fov: 50 }}>
      <color attach="background" args={["#020405"]} />
      <ambientLight intensity={0.8} />
      <CoreMesh />
    </Canvas>
  );
}

export default function Runs() {
  const runsQ = useQuery({ queryKey: ["runs"], queryFn: fetchRuns, refetchInterval: 5000 });
  const runs = (runsQ.data ?? []) as Run[];
  const { ctx, windowSeconds } = useContextFilters();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<string>("all");
  const [sort, setSort] = useState<"updated" | "best" | "step">("updated");
  const [selected, setSelected] = useState<string | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState<string[]>([]);
  const loc = useLocation();
  const nav = useNavigate();

  const compareQ = useQuery<RunCompareResponse>({
    queryKey: ["runsCompare", compareSelection.join(",")],
    queryFn: () => fetchRunsCompare(compareSelection, ["reward", "score"], { window_s: 6 * 3600, stride: 2 }),
    enabled: compareSelection.length >= 2,
  });

  const statuses = useMemo(() => {
    const s = new Set<string>();
    for (const r of runs) s.add(String(r.status ?? ""));
    return ["all", ...Array.from(s).filter(Boolean).sort()];
  }, [runs]);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    const now = Date.now() / 1000;
    const cutoff = windowSeconds ? now - windowSeconds : null;
    const arr = runs.filter((r) => {
      if (ctx.run !== "all" && String(r.run_id ?? "") !== ctx.run) return false;
      if (ctx.policy !== "all") {
        const pol = String((r.config as any)?.policy_name ?? "");
        if (pol && pol !== ctx.policy) return false;
      }
      if (cutoff != null && Number(r.updated_ts ?? 0) < cutoff) return false;
      if (status !== "all" && String(r.status) !== status) return false;
      if (!qq) return true;
      const hay = `${r.run_id} ${r.experiment_id} ${r.status}`.toLowerCase();
      return hay.includes(qq);
    });
    arr.sort((a, b) => {
      if (sort === "best") return (b.best_reward ?? 0) - (a.best_reward ?? 0);
      if (sort === "step") return (b.last_step ?? 0) - (a.last_step ?? 0);
      return (b.updated_ts ?? 0) - (a.updated_ts ?? 0);
    });
    return arr;
  }, [runs, q, status, sort, ctx.run, ctx.policy, windowSeconds]);

  const selectedRun = useMemo(() => filtered.find((r) => String(r.run_id) === String(selected)) ?? null, [filtered, selected]);
  const activeCount = useMemo(() => runs.filter((r) => String(r.status ?? "").toLowerCase() !== "completed").length, [runs]);
  const bestOverall = useMemo(() => runs.reduce((m, r) => Math.max(m, Number(r.best_reward ?? 0)), 0), [runs]);
  const newestTs = useMemo(() => (runs.length ? runs.reduce((m, r) => Math.max(m, Number(r.updated_ts ?? 0)), 0) : null), [runs]);
  const dataAge = newestTs != null ? Date.now() / 1000 - newestTs : null;

  const compareRuns = useMemo(() => (compareQ.data?.runs ?? []).map((r) => r.run_id), [compareQ.data]);

  useEffect(() => {
    const qs = new URLSearchParams(loc.search);
    const id = qs.get("run") ?? qs.get("run_id");
    if (id) setSelected(String(id));
  }, [loc.search]);
  const metricsByRun = useMemo(() => {
    const map: Record<string, RunMetricSeries[]> = {};
    for (const series of compareQ.data?.metrics ?? []) {
      (map[series.run_id] ??= []).push(series);
    }
    return map;
  }, [compareQ.data]);

  const diffKeys = useMemo(() => {
    const selectedRuns = runs.filter((r) => compareSelection.includes(r.run_id));
    if (selectedRuns.length < 2) return [];
    const keys = new Set<string>();
    selectedRuns.forEach((r) => Object.keys(r.config ?? {}).forEach((k) => keys.add(k)));
    const out: string[] = [];
    keys.forEach((k) => {
      const vals = new Set(selectedRuns.map((r) => JSON.stringify((r.config ?? {})[k])));
      if (vals.size > 1) out.push(k);
    });
    return out.sort();
  }, [compareSelection, runs]);

  const toggleCompare = (runId: string) => {
    setCompareSelection((prev) => {
      if (prev.includes(runId)) return prev.filter((id) => id !== runId);
      return [...prev, runId];
    });
  };

  const RunRow = ({ index, style }: ListChildComponentProps) => {
    const r = filtered[index];
    const isSelected = String(selectedRun?.run_id) === String(r.run_id);
    return (
      <div
        style={style}
        className={`v-row ${isSelected ? "active" : ""}`}
        onClick={() => {
          setSelected(r.run_id);
          const next = new URLSearchParams(loc.search);
          next.set("run", r.run_id);
          next.set("run_id", r.run_id);
          next.set("step", String(r.last_step ?? ""));
          next.set("ts", String(r.updated_ts ?? ""));
          nav({ pathname: loc.pathname, search: next.toString() }, { replace: true });
        }}
      >
        {compareMode && (
          <div className="v-cell" style={{ width: 40, justifyContent: "center" }}>
            <input
              type="checkbox"
              checked={compareSelection.includes(r.run_id)}
              onChange={() => toggleCompare(r.run_id)}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        )}
        <div className="v-cell mono" style={{ flex: 1.2 }} title={r.run_id}>
          {r.run_id}
        </div>
        <div className="v-cell mono" style={{ flex: 1.5 }} title={r.experiment_id}>
          {r.experiment_id}
        </div>
        <div className="v-cell" style={{ width: 100 }}>
          <span className="pill">{r.status}</span>
        </div>
        <div className="v-cell numeric" style={{ width: 80 }}>
          {fmtFixed(r.best_reward, 2)}
        </div>
        <div className="v-cell numeric" style={{ width: 80 }}>
          {fmtFixed(r.last_reward, 2)}
        </div>
        <div className="v-cell numeric" style={{ width: 80 }}>
          {fmtNum(r.last_step)}
        </div>
        <div className="v-cell muted numeric" style={{ width: 100 }}>
          {timeAgo(r.updated_ts)}
        </div>
      </div>
    );
  };

  const miniSeries = (series: RunMetricSeries | undefined) => {
    const pts = series?.points ?? [];
    if (pts.length < 2) return null;
    const w = 160;
    const h = 48;
    const pad = 6;
    const minV = Math.min(...pts.map((p) => p.value));
    const maxV = Math.max(...pts.map((p) => p.value));
    const span = Math.max(1e-6, maxV - minV);
    const coords = pts.map((p, i) => {
      const x = pad + (i / (pts.length - 1)) * (w - pad * 2);
      const y = pad + (1 - (p.value - minV) / span) * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return (
      <svg viewBox={`0 0 ${w} ${h}`} className="mini-chart">
        <polyline fill="none" stroke="rgba(0,229,255,.9)" strokeWidth="2" points={coords.join(" ")} />
      </svg>
    );
  };

  return (
    <div className="grid runs-grid" style={{ gridTemplateColumns: "1.2fr 0.8fr" }}>
      <section className="card runs-core-card" style={{ gridColumn: "1 / -1" }}>
        <div className="row-between">
          <h2>Run Constellation</h2>
          <span className="badge">{fmtNum(runs.length)} total</span>
        </div>
        <div className="runs-core-body">
          <div>
            <div className="muted">Temporal alignment • rollouts • policy drift</div>
            <div className="runs-core-kpis">
              <div>
                <span className="label">Active runs</span>
                <strong>{fmtNum(activeCount)}</strong>
              </div>
              <div>
                <span className="label">Best reward</span>
                <strong>{fmtFixed(bestOverall, 2)}</strong>
              </div>
              <div>
                <span className="label">Compare set</span>
                <strong>{compareSelection.length}</strong>
              </div>
            </div>
          </div>
          <RunsCoreViz />
        </div>
      </section>
      <section className="card flex-card">
        <div className="row-between">
          <h2>Runs</h2>
          <span className="badge">{fmtNum(filtered.length)} shown</span>
        </div>
        <div className="kpis kpis-wrap" style={{ marginTop: 10 }}>
          <div className="kpi">
            <div className="label">Total</div>
            <div className="value">{fmtNum(runs.length)}</div>
          </div>
          <div className="kpi">
            <div className="label">Active</div>
            <div className="value">{fmtNum(activeCount)}</div>
          </div>
          <div className="kpi">
            <div className="label">Best Overall</div>
            <div className="value">{fmtFixed(bestOverall, 2)}</div>
          </div>
        </div>

        <div className="toolbar">
          <div className="toolbar-left">
            <input className="input" placeholder="search run / experiment…" value={q} onChange={(e) => setQ(e.target.value)} />
            <select className="select" value={status} onChange={(e) => setStatus(e.target.value)}>
              {statuses.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <select className="select" value={sort} onChange={(e) => setSort(e.target.value as any)}>
              <option value="updated">sort: updated</option>
              <option value="best">sort: best</option>
              <option value="step">sort: step</option>
            </select>
            <button className={`btn btn-ghost ${compareMode ? "active" : ""}`} onClick={() => setCompareMode((v) => !v)}>
              compare
            </button>
          </div>
          <div className="toolbar-right">
            <span className={`pill ${dataAge != null && dataAge > 30 ? "pill-missing" : "pill-ok"}`}>
              data age {dataAge == null || newestTs == null ? "—" : timeAgo(newestTs)}
            </span>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                if (!selectedRun) return;
                await copyToClipboard(selectedRun.run_id);
              }}
              disabled={!selectedRun}
            >
              copy run id
            </button>
          </div>
        </div>

        <div className="v-table-header" style={{ marginTop: 10 }}>
          {compareMode && <div style={{ width: 40 }} />}
          <div style={{ flex: 1.2 }}>Run</div>
          <div style={{ flex: 1.5 }}>Experiment</div>
          <div style={{ width: 100 }}>Status</div>
          <div style={{ width: 80, textAlign: "right" }}>Best</div>
          <div style={{ width: 80, textAlign: "right" }}>Last</div>
          <div style={{ width: 80, textAlign: "right" }}>Step</div>
          <div style={{ width: 100, textAlign: "right" }}>Updated</div>
        </div>

        <div className="v-table-body" style={{ flex: 1, minHeight: 0 }}>
          {filtered.length ? (
            <AutoSizer>
              {({ height, width }) => (
                <VirtualList height={height} width={width} itemCount={filtered.length} itemSize={42}>
                  {RunRow}
                </VirtualList>
              )}
            </AutoSizer>
          ) : (
            <div className="muted" style={{ marginTop: 8 }}>
              no runs yet
            </div>
          )}
        </div>
      </section>

      <section className="card">
        <div className="row-between">
          <h2>{compareMode ? "Run Compare" : "Run Detail"}</h2>
          <span className="badge">{compareMode ? `${fmtNum(compareSelection.length)} selected` : selectedRun ? "selected" : "pick one"}</span>
        </div>
        {compareMode ? (
          <>
            {compareSelection.length < 2 ? (
              <div className="muted" style={{ marginTop: 10 }}>
                Select at least 2 runs to compare.
              </div>
            ) : (
              <>
                <div className="panel" style={{ marginTop: 10 }}>
                  <div className="muted">Compared runs</div>
                  <div className="statline">
                    {compareRuns.map((rid) => (
                      <span key={rid} className="chip">
                        {rid}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="panel" style={{ marginTop: 10 }}>
                  <div className="muted">Metric preview</div>
                  <div className="compare-grid">
                    {compareRuns.map((rid) => (
                      <div key={rid} className="compare-card">
                        <div className="mono">{rid}</div>
                        {miniSeries(metricsByRun[rid]?.find((m) => m.metric === "reward")) ?? <div className="muted">no data</div>}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel" style={{ marginTop: 10 }}>
                  <div className="muted">Config diff</div>
                  {!diffKeys.length ? (
                    <div className="muted">no config differences detected</div>
                  ) : (
                    <div className="kv" style={{ marginTop: 6 }}>
                      {diffKeys.slice(0, 16).map((k) => (
                        <div key={`${k}-row`} style={{ display: "contents" }}>
                          <div className="k">{k}</div>
                          <div className="v mono">differs</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </>
        ) : !selectedRun ? (
          <div className="muted">Click a run row to inspect config + metadata.</div>
        ) : (
          <>
            <div className="kv" style={{ marginTop: 10 }}>
              <div className="k">run</div>
              <div className="v mono">{selectedRun.run_id}</div>
              <div className="k">experiment</div>
              <div className="v mono">{selectedRun.experiment_id}</div>
              <div className="k">status</div>
              <div className="v">{selectedRun.status}</div>
              <div className="k">best</div>
              <div className="v">{fmtFixed(selectedRun.best_reward, 4)}</div>
              <div className="k">last</div>
              <div className="v">{fmtFixed(selectedRun.last_reward, 4)}</div>
              <div className="k">step</div>
              <div className="v">{fmtNum(selectedRun.last_step)}</div>
              <div className="k">updated</div>
              <div className="v">{timeAgo(selectedRun.updated_ts)}</div>
              <div className="k">created</div>
              <div className="v">{timeAgo(selectedRun.created_ts)}</div>
            </div>
            <div style={{ marginTop: 12 }}>
              <div className="muted">Config</div>
              <pre className="code code-tall">{JSON.stringify(selectedRun.config ?? {}, null, 2)}</pre>
            </div>
            <div className="panel" style={{ marginTop: 12 }}>
              <div className="muted">Artifacts</div>
              <div className="statline">
                <span className="pill">config</span>
                <span className="pill">checkpoint</span>
                <span className="pill">rollouts</span>
                <span className="pill">logs</span>
              </div>
              <div className="muted" style={{ marginTop: 6 }}>
                Attachments populate when artifact lineage is enabled.
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
