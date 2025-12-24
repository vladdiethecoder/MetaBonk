import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import VirtualList, { type ListChildComponentProps } from "../components/VirtualList";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { fetchHistoricLeaderboard, fetchInstanceTimeline, fetchInstances, HistoricLeaderboardEntry, InstanceTimelineResponse, InstanceView } from "../api";
import { useEventStream } from "../hooks/useEventStream";
import useContextFilters from "../hooks/useContextFilters";
import { useContextDrawer } from "../hooks/useContextDrawer";
import { deriveReasonCode } from "../hooks/useIssues";
import { clamp01, copyToClipboard, fmtFixed, fmtNum, fmtPct01, timeAgo } from "../lib/format";
import { HEARTBEAT_SCHEMA_VERSION, schemaMismatchLabel } from "../lib/schema";
import PageShell from "../components/PageShell";
import ElementSizer from "../components/ElementSizer";
import QueryStateGate from "../components/QueryStateGate";
import useActivationResizeKick from "../hooks/useActivationResizeKick";
import { bumpWebglCount, reportWebglLost } from "../hooks/useWebglCounter";
import { useWebglResetNonce } from "../hooks/useWebglReset";

function InstanceLatticeViz() {
  const [lost, setLost] = useState(false);
  const resetNonce = useWebglResetNonce();
  useEffect(() => {
    bumpWebglCount(1);
    return () => bumpWebglCount(-1);
  }, []);
  const LatticeMesh = () => {
    const ref = useRef<THREE.Mesh | null>(null);
    useFrame((state) => {
      if (!ref.current) return;
      const t = state.clock.getElapsedTime();
      ref.current.rotation.y = t * 0.3;
      ref.current.rotation.z = t * 0.2;
    });
    return (
      <mesh ref={ref}>
        <icosahedronGeometry args={[1, 1]} />
        <meshBasicMaterial wireframe color="#7bffe6" transparent opacity={0.45} />
      </mesh>
    );
  };
  if (lost) return <div className="canvas-placeholder">WebGL context lost</div>;
  return (
    <Canvas
      className="instances-core-canvas"
      key={`instances-core-${resetNonce}`}
      dpr={[1, 2]}
      camera={{ position: [0, 0, 3.2], fov: 50 }}
      onCreated={({ gl }) => {
        const onLost = (evt: Event) => {
          evt.preventDefault();
          setLost(true);
          reportWebglLost();
        };
        const onRestore = () => setLost(false);
        gl.domElement.addEventListener("webglcontextlost", onLost, { passive: false });
        gl.domElement.addEventListener("webglcontextrestored", onRestore);
      }}
    >
      <color attach="background" args={["#020405"]} />
      <ambientLight intensity={0.8} />
      <LatticeMesh />
    </Canvas>
  );
}

function Sparkline({ values, color }: { values: number[] | null | undefined; color: string }) {
  const vals = (values ?? []).filter((v) => Number.isFinite(v));
  if (vals.length < 2) return <div className="sparkline muted">—</div>;
  const w = 90;
  const h = 24;
  const pad = 3;
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const span = Math.max(1e-6, maxV - minV);
  const pts = vals.map((v, i) => {
    const x = pad + (i / (vals.length - 1)) * (w - pad * 2);
    const y = pad + (1 - (v - minV) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${h}`}>
      <polyline fill="none" stroke={color} strokeWidth="2" points={pts.join(" ")} />
    </svg>
  );
}

function TelemetryChart({ points }: { points: Array<{ score?: number; reward?: number }> }) {
  if (!points.length) return <div className="muted">no telemetry yet</div>;
  const scores = points.map((p) => Number(p.score ?? 0));
  const rewards = points.map((p) => Number(p.reward ?? 0));
  const all = [...scores, ...rewards];
  const minV = Math.min(...all);
  const maxV = Math.max(...all);
  const span = Math.max(1e-6, maxV - minV);
  const w = 320;
  const h = 120;
  const pad = 10;
  const toPts = (vals: number[]) =>
    vals.map((v, i) => {
      const x = pad + (i / Math.max(1, vals.length - 1)) * (w - pad * 2);
      const y = pad + (1 - (v - minV) / span) * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
  return (
    <svg className="telemetry-chart" viewBox={`0 0 ${w} ${h}`}>
      <rect x="0" y="0" width={w} height={h} rx="10" fill="rgba(6,8,12,.6)" stroke="rgba(43,21,60,.8)" />
      <polyline fill="none" stroke="var(--blue)" strokeWidth="2" points={toPts(scores).join(" ")} />
      <polyline fill="none" stroke="var(--yellow)" strokeWidth="2" points={toPts(rewards).join(" ")} />
    </svg>
  );
}

export default function Instances() {
  const loc = useLocation();
  const isActive = loc.pathname === "/instances";
  useActivationResizeKick(isActive);
  const [kickKey, setKickKey] = useState(0);
  useEffect(() => {
    if (isActive) setKickKey((k) => k + 1);
  }, [isActive]);
  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 2000 });
  const histQ = useQuery({ queryKey: ["historicLeaderboard"], queryFn: () => fetchHistoricLeaderboard(200, "best_score"), refetchInterval: 5000 });
  const instances = (instQ.data ?? {}) as Record<string, InstanceView>;
  const nav = useNavigate();
  const { ctx, windowSeconds } = useContextFilters();
  const events = useEventStream(220);
  const [q, setQ] = useState("");
  const [policy, setPolicy] = useState<string>("all");
  const [sort, setSort] = useState<"score" | "step" | "seen">("score");
  const [selected, setSelected] = useState<string | null>(null);
  const openContext = useContextDrawer();
  const timelineQ = useQuery<InstanceTimelineResponse>({
    queryKey: ["instanceTimeline", selected, windowSeconds],
    queryFn: () => fetchInstanceTimeline(selected as string, windowSeconds ?? 600, 120),
    enabled: Boolean(selected),
    refetchInterval: 3000,
  });

  useEffect(() => {
    const qs = new URLSearchParams(loc.search);
    const id = qs.get("id") ?? qs.get("instance") ?? qs.get("iid") ?? qs.get("instance_id");
    if (id) setSelected(String(id));
  }, [loc.search]);

  const rows = useMemo(() => {
    const histById: Record<string, HistoricLeaderboardEntry> = {};
    for (const r of (histQ.data ?? []) as HistoricLeaderboardEntry[]) {
      if (!r?.instance_id) continue;
      histById[String(r.instance_id)] = r;
    }
    const out = Object.entries(instances).map(([id, v]) => {
      const hb = v.heartbeat;
      const hist = histById[String(id)];
      const score = Number(hb?.steam_score ?? hb?.reward ?? 0);
      const step = Number(hb?.step ?? 0);
      const ts = Number(hb?.ts ?? 0);
      const streamUrl = String(hb?.stream_url ?? "");
      const streamOk = Boolean(hb?.stream_ok);
      const schemaMismatch = hb?.schema_version != null && Number(hb.schema_version) !== HEARTBEAT_SCHEMA_VERSION;
      const streamState = schemaMismatch ? "schema" : !streamUrl ? "missing" : streamOk ? "ok" : "stale";
      const streamError = hb?.stream_error ?? hb?.streamer_last_error ?? null;
      const reason = schemaMismatch ? schemaMismatchLabel(hb?.schema_version) : deriveReasonCode(hb);
      const survival = hb?.survival_prob != null ? clamp01(Number(hb.survival_prob)) : null;
      const danger = hb?.danger_level != null ? clamp01(Number(hb.danger_level)) : survival != null ? clamp01(1 - survival) : null;
      const name = (hb?.display_name ?? id) as string;
      const pol = (hb?.policy_name ?? "—") as string;
      const device = (hb?.worker_device ?? "—") as string;
      const runId = hb?.run_id ?? null;
      return {
        id,
        name,
        policy: pol,
        score,
        step,
        ts,
        device,
        streamState,
        streamError,
        streamUrl,
        hb,
        reason,
        schemaMismatch,
        survival,
        danger,
        cfg: v.config ?? null,
        best_score: hist?.best_score ?? null,
        best_step: hist?.best_step ?? null,
        runId,
        sparks: v.telemetry?.sparks ?? null,
        history: v.telemetry?.history ?? [],
      };
    });

    const qq = q.trim().toLowerCase();
    const filtered = out.filter((r) => {
      if (ctx.policy !== "all" && r.policy !== ctx.policy) return false;
      if (ctx.run !== "all" && String(r.runId ?? "") !== ctx.run) return false;
      if (windowSeconds) {
        const now = Date.now() / 1000;
        if (Number(r.ts ?? 0) < now - windowSeconds) return false;
      }
      if (policy !== "all" && r.policy !== policy) return false;
      if (!qq) return true;
      const hay = `${r.id} ${r.name} ${r.policy}`.toLowerCase();
      return hay.includes(qq);
    });

    filtered.sort((a, b) => {
      if (sort === "step") return b.step - a.step;
      if (sort === "seen") return b.ts - a.ts;
      return b.score - a.score;
    });
    return filtered;
  }, [instances, histQ.data, q, policy, sort, ctx.policy, ctx.run, windowSeconds]);

  const policies = useMemo(() => {
    const s = new Set<string>();
    for (const r of rows) s.add(String(r.policy ?? ""));
    return ["all", ...Array.from(s).filter(Boolean).sort()];
  }, [rows]);

  const selectedRow = useMemo(() => rows.find((r) => String(r.id) === String(selected)) ?? null, [rows, selected]);
  const newestTs = useMemo(() => (rows.length ? rows.reduce((m, r) => Math.max(m, Number(r.ts ?? 0)), 0) : null), [rows]);
  const dataAge = newestTs != null ? Date.now() / 1000 - newestTs : null;
  const selectedEvents = useMemo(() => {
    if (!selectedRow) return [];
    return events.filter((e) => String(e.instance_id ?? "") === String(selectedRow.id)).slice(-12).reverse();
  }, [events, selectedRow]);
  const frameUrl = useMemo(() => {
    const base = selectedRow?.hb?.control_url ?? "";
    if (!base) return "";
    return `${String(base).replace(/\/+$/, "")}/frame.jpg`;
  }, [selectedRow]);
  const timelinePoints = useMemo(() => {
    if (timelineQ.data?.points?.length) return timelineQ.data.points;
    return selectedRow?.history ?? [];
  }, [timelineQ.data, selectedRow]);

  const InstanceRow = ({ index, style }: ListChildComponentProps) => {
    const r = rows[index];
    const isSelected = String(selectedRow?.id) === String(r.id);
    return (
      <div
        style={style}
        className={`v-row ${isSelected ? "active" : ""}`}
        onClick={() => {
          setSelected(r.id);
          const next = new URLSearchParams(loc.search);
          next.set("instance", r.id);
          next.set("instance_id", r.id);
          if (r.runId) next.set("run_id", String(r.runId));
          next.set("step", String(r.step ?? ""));
          next.set("ts", String(r.ts ?? ""));
          nav({ pathname: loc.pathname, search: next.toString() }, { replace: true });
        }}
      >
        <div className="v-cell" style={{ flex: 1.5 }}>
          <div style={{ fontWeight: 800, overflow: "hidden", textOverflow: "ellipsis" }}>{r.name}</div>
          <div className="muted mono" style={{ fontSize: "0.85em" }}>
            {r.id}
          </div>
        </div>
        <div className="v-cell" style={{ flex: 1 }} title={r.policy}>
          {r.policy}
        </div>
        <div className="v-cell muted" style={{ width: 100 }} title={r.device}>
          {r.device}
        </div>
        <div className="v-cell" style={{ width: 80 }}>
          <span className={`pill ${r.streamState === "ok" ? "pill-ok" : "pill-missing"}`}>{r.streamState}</span>
        </div>
        <div className="v-cell muted mono" style={{ width: 80 }} title={r.reason ?? ""}>
          {r.reason ?? "—"}
        </div>
        <div className="v-cell" style={{ width: 80 }}>
          <div>{fmtFixed(r.score, 2)}</div>
          <div className="muted" style={{ fontSize: "0.8em" }}>
            best {r.best_score == null ? "—" : fmtFixed(r.best_score, 2)}
          </div>
        </div>
        <div className="v-cell" style={{ width: 100 }}>
          <div className="sparks">
            <Sparkline values={r.sparks?.score} color="var(--blue)" />
            <Sparkline values={r.sparks?.reward} color="var(--yellow)" />
            <Sparkline values={r.sparks?.entropy} color="var(--pink)" />
            <Sparkline values={r.sparks?.stream_fps} color="rgba(0,255,136,.9)" />
          </div>
        </div>
        <div className="v-cell" style={{ width: 80 }}>
          <div>{fmtNum(r.step)}</div>
          <div className="muted" style={{ fontSize: "0.8em" }}>
            peak {r.best_step == null ? "—" : fmtNum(r.best_step)}
          </div>
        </div>
        <div className="v-cell" style={{ width: 60 }}>
          {r.survival == null ? "—" : fmtPct01(r.survival, 0)}
        </div>
        <div className="v-cell" style={{ width: 60 }}>
          {r.danger == null ? "—" : fmtPct01(r.danger, 0)}
        </div>
        <div className="v-cell muted" style={{ width: 80, fontSize: "0.9em" }}>
          {timeAgo(r.ts)}
        </div>
      </div>
    );
  };

  return (
    <QueryStateGate label="Instances" queries={[instQ, histQ, timelineQ]}>
      <PageShell className="grid page-grid instances-grid" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 0.7fr)" }}>
      <section className="card instances-core-card" style={{ gridColumn: "1 / -1" }}>
        <div className="row-between">
          <h2>Instance Lattice</h2>
          <span className="badge">{fmtNum(Object.keys(instances).length)} live</span>
        </div>
        <div className="instances-core-body">
          <div>
            <div className="muted">Telemetry coherence • hazard fields • stream integrity</div>
            <div className="instances-core-kpis">
              <div>
                <span className="label">Selected</span>
                <strong>{selected ?? "—"}</strong>
              </div>
              <div>
                <span className="label">Events</span>
                <strong>{events.length}</strong>
              </div>
              <div>
                <span className="label">Policy</span>
                <strong>{policy === "all" ? "all" : policy}</strong>
              </div>
            </div>
          </div>
          {isActive ? <InstanceLatticeViz /> : <div className="canvas-placeholder" />}
        </div>
      </section>
      <section className="card flex-card">
        <div className="row-between">
          <h2>Instances</h2>
          <span className="badge">{fmtNum(rows.length)} online</span>
        </div>
        <div className="toolbar">
          <div className="toolbar-left">
            <input className="input" placeholder="search instance / name / policy…" value={q} onChange={(e) => setQ(e.target.value)} />
            <select className="select" value={policy} onChange={(e) => setPolicy(e.target.value)}>
              {policies.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <select className="select" value={sort} onChange={(e) => setSort(e.target.value as any)}>
              <option value="score">sort: score</option>
              <option value="step">sort: step</option>
              <option value="seen">sort: last seen</option>
            </select>
          </div>
          <div className="toolbar-right">
            <span className={`pill ${dataAge != null && dataAge > 20 ? "pill-missing" : "pill-ok"}`}>
              data age {dataAge == null || newestTs == null ? "—" : timeAgo(newestTs)}
            </span>
            <Link className="btn btn-ghost" to="/stream">
              open stream HUD
            </Link>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                if (!selectedRow) return;
                await copyToClipboard(selectedRow.id);
              }}
              disabled={!selectedRow}
            >
              copy id
            </button>
          </div>
        </div>

        <div className="v-table-header" style={{ marginTop: 10 }}>
          <div style={{ flex: 1.5 }}>Agent</div>
          <div style={{ flex: 1 }}>Policy</div>
          <div style={{ width: 100 }}>Device</div>
          <div style={{ width: 80 }}>Stream</div>
          <div style={{ width: 80 }}>Reason</div>
          <div style={{ width: 80 }}>Score</div>
          <div style={{ width: 100 }}>Trends</div>
          <div style={{ width: 80 }}>Step</div>
          <div style={{ width: 60 }}>Surv</div>
          <div style={{ width: 60 }}>Dngr</div>
          <div style={{ width: 80 }}>Seen</div>
        </div>

        <div className="v-table-body" style={{ flex: 1, minHeight: 0 }}>
          {rows.length ? (
            <ElementSizer key={kickKey}>
              {({ height, width }) => (
                <VirtualList height={height} width={width} itemCount={rows.length} itemSize={72}>
                  {InstanceRow}
                </VirtualList>
              )}
            </ElementSizer>
          ) : (
            <div className="muted" style={{ marginTop: 8 }}>
              no instances yet
            </div>
          )}
        </div>
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Instance Detail</h2>
          <span className="badge">{selectedRow ? "selected" : "pick one"}</span>
        </div>
        {!selectedRow ? (
          <div className="muted">Click a row to inspect stream + telemetry fields.</div>
        ) : (
          <>
            <div className="kv" style={{ marginTop: 10 }}>
              <div className="k">name</div>
              <div className="v">{selectedRow.name}</div>
              <div className="k">instance</div>
              <div className="v mono">{selectedRow.id}</div>
              <div className="k">policy</div>
              <div className="v">{selectedRow.policy}</div>
              <div className="k">status</div>
              <div className="v">{selectedRow.hb?.status ?? "—"}</div>
              <div className="k">reason</div>
              <div className="v mono">{selectedRow.reason ?? "—"}</div>
              <div className="k">device</div>
              <div className="v">{selectedRow.device}</div>
              <div className="k">stream</div>
              <div className="v">
                <span className={`pill ${selectedRow.streamState === "ok" ? "pill-ok" : "pill-missing"}`}>{selectedRow.streamState}</span>
                {selectedRow.streamError ? <span className="muted"> • {selectedRow.streamError}</span> : null}
              </div>
              <div className="k">clients</div>
              <div className="v">
                {selectedRow.hb?.stream_active_clients ?? 0}/{selectedRow.hb?.stream_max_clients ?? "—"}
              </div>
              <div className="k">score</div>
              <div className="v">
                {fmtFixed(selectedRow.score, 3)} <span className="muted">best {selectedRow.best_score == null ? "—" : fmtFixed(selectedRow.best_score, 3)}</span>
              </div>
              <div className="k">step</div>
              <div className="v">
                {fmtNum(selectedRow.step)} <span className="muted">peak {selectedRow.best_step == null ? "—" : fmtNum(selectedRow.best_step)}</span>
              </div>
              <div className="k">seen</div>
              <div className="v">{timeAgo(selectedRow.ts)}</div>
              <div className="k">luck</div>
              <div className="v">
                {selectedRow.hb?.luck_mult != null ? `${fmtFixed(Number(selectedRow.hb.luck_mult), 2)}x` : "—"}{" "}
                <span className="muted">{selectedRow.hb?.luck_label ?? ""}</span>
              </div>
              <div className="k">borgars</div>
              <div className="v">{selectedRow.hb?.borgar_count ?? "—"}</div>
              <div className="k">enemies</div>
              <div className="v">{selectedRow.hb?.enemy_count ?? "—"}</div>
            </div>

            <div style={{ marginTop: 10 }}>
              {selectedRow.streamUrl ? (
                <div className="stream-placeholder">
                  <div className="muted">stream</div>
                  <div className="row-between" style={{ marginTop: 6 }}>
                    <a className="btn btn-ghost" href={selectedRow.streamUrl} target="_blank" rel="noreferrer">
                      open stream
                    </a>
                    <div className="row" style={{ gap: 8 }}>
                      {frameUrl ? (
                        <a className="btn btn-ghost" href={frameUrl} target="_blank" rel="noreferrer">
                          frame.jpg
                        </a>
                      ) : null}
                      <span className="muted">{selectedRow.hb?.stream_type ?? "—"}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="muted stream-placeholder">stream unavailable</div>
              )}
            </div>

            <div className="panel" style={{ marginTop: 10 }}>
              <div className="row-between">
                <div className="muted">Telemetry timeline</div>
                <span className="badge">{fmtNum(timelinePoints.length)} pts</span>
              </div>
              <div style={{ marginTop: 8 }}>
                <TelemetryChart points={timelinePoints} />
                <div className="muted" style={{ marginTop: 6 }}>score (blue) • reward (yellow)</div>
              </div>
            </div>

            <details className="details">
              <summary className="muted">Raw heartbeat</summary>
              <pre className="code code-tall">{JSON.stringify(selectedRow.hb ?? {}, null, 2)}</pre>
            </details>
            <div className="panel" style={{ marginTop: 10 }}>
              <div className="row-between">
                <div className="muted">Flight recorder</div>
                <span className="badge">{fmtNum(selectedEvents.length)} events</span>
              </div>
              <div className="events" style={{ marginTop: 8 }}>
                {selectedEvents.map((e) => (
                  <div
                    key={e.event_id}
                    className="event"
                    style={{ cursor: "pointer" }}
                    onClick={() =>
                      openContext({
                        title: e.message,
                        kind: "event",
                        instanceId: e.instance_id ?? null,
                        runId: e.run_id ?? null,
                        ts: e.ts,
                        details: e.payload ?? {},
                      })
                    }
                  >
                    <span className="badge">{e.event_type}</span>
                    <span>{e.message}</span>
                    <span className="muted">{timeAgo(e.ts)}</span>
                  </div>
                ))}
                {!selectedEvents.length && <div className="muted">no recent events</div>}
              </div>
            </div>
            {selectedRow.cfg ? (
              <details className="details">
                <summary className="muted">Assigned config</summary>
                <pre className="code code-tall">{JSON.stringify(selectedRow.cfg ?? {}, null, 2)}</pre>
              </details>
            ) : null}
          </>
        )}
      </section>
      </PageShell>
    </QueryStateGate>
  );
}
