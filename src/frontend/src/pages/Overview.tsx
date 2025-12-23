import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useRef, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { EffectComposer, Bloom, Scanline } from "@react-three/postprocessing";
import * as THREE from "three";
import { FixedSizeList as List, ListChildComponentProps } from "react-window";
import AutoSizer from "react-virtualized-auto-sizer";
import { fetchOverviewHealth, fetchOverviewIssues, fetchPbtMute, fetchPolicies, fetchStatus, fetchWorkers, setPbtMute } from "../api";
import { useEventStream } from "../hooks";
import useIssues from "../hooks/useIssues";
import { useContextDrawer } from "../hooks/useContextDrawer";
import useContextFilters from "../hooks/useContextFilters";
import { fmtFixed, fmtNum, timeAgo } from "../lib/format";

const hashString = (input: string) => {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

function HoloGlobe({
  nodes,
  errorRate = 0,
}: {
  nodes: Array<{ id: string; health: number; active: boolean }>;
  errorRate?: number;
}) {
  const points = useMemo(() => {
    return nodes.map((n) => {
      const seed = hashString(n.id);
      const u = (seed % 1000) / 1000;
      const v = ((seed >> 10) % 1000) / 1000;
      const theta = u * Math.PI * 2;
      const phi = Math.acos(2 * v - 1);
      const r = 1;
      return new THREE.Vector3(r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
    });
  }, [nodes]);

  const ringSpike = errorRate > 0.04;

  const NodePoints = () => {
    const geo = useMemo(() => {
      const arr = new Float32Array(points.length * 3);
      points.forEach((p, i) => {
        arr[i * 3 + 0] = p.x;
        arr[i * 3 + 1] = p.y;
        arr[i * 3 + 2] = p.z;
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
      return g;
    }, [points]);
    return (
      <points geometry={geo}>
        <pointsMaterial color="#7bffe6" size={0.04} sizeAttenuation />
      </points>
    );
  };

  const GlobeGroup = () => {
    const ref = useRef<THREE.Group | null>(null);
    useFrame((state) => {
      if (!ref.current) return;
      ref.current.rotation.y = state.clock.getElapsedTime() * 0.25;
    });
    return (
      <group ref={ref}>
        <mesh>
          <icosahedronGeometry args={[1, 2]} />
          <meshBasicMaterial wireframe color="#6fffe6" transparent opacity={0.35} />
        </mesh>
        <NodePoints />
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[1.25, 0.01, 12, 120]} />
          <meshBasicMaterial color={ringSpike ? "#ff8b5a" : "#5df1da"} transparent opacity={0.5} />
        </mesh>
        <mesh rotation={[Math.PI / 3, 0.6, 0]}>
          <torusGeometry args={[1.38, 0.008, 12, 120]} />
          <meshBasicMaterial color="#7df0ff" transparent opacity={0.35} />
        </mesh>
      </group>
    );
  };

  return (
    <Canvas className="holo-globe-canvas" dpr={[1, 2]} camera={{ position: [0, 0, 3.2], fov: 50 }}>
      <color attach="background" args={["#020405"]} />
      <ambientLight intensity={0.8} />
      <GlobeGroup />
      <EffectComposer>
        <Bloom intensity={0.9} luminanceThreshold={0.2} />
        <Scanline density={1.2} opacity={0.2} />
      </EffectComposer>
    </Canvas>
  );
}

export default function Overview() {
  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 2000 });
  const status = statusQ.data;
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 2000 });
  const workers = workersQ.data ?? {};
  const { ctx, windowSeconds } = useContextFilters();
  const healthWindow = windowSeconds ?? 300;
  const healthQ = useQuery({
    queryKey: ["overviewHealth", healthWindow],
    queryFn: () => fetchOverviewHealth(healthWindow),
    refetchInterval: 4000,
  });
  const issuesQ = useQuery({
    queryKey: ["overviewIssues", healthWindow],
    queryFn: () => fetchOverviewIssues(Math.max(600, healthWindow)),
    refetchInterval: 5000,
  });
  const policiesQ = useQuery({ queryKey: ["policies"], queryFn: fetchPolicies, refetchInterval: 4000 });
  const pbtMuteQ = useQuery({ queryKey: ["pbtMute"], queryFn: fetchPbtMute, refetchInterval: 4000 });
  const qc = useQueryClient();
  const pbtMuteMut = useMutation({
    mutationFn: setPbtMute,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["pbtMute"] });
      qc.invalidateQueries({ queryKey: ["policies"] });
    },
  });
  const events = useEventStream(100);
  const issues = useIssues(200);
  const openContext = useContextDrawer();
  const [eventType, setEventType] = useState<string>("all");
  const [q, setQ] = useState("");
  const fmtPct = (v?: number | null, digits = 1) => {
    if (v == null || !Number.isFinite(v)) return "—";
    return `${(v * 100).toFixed(digits)}%`;
  };

  const eventTypes = useMemo(() => {
    const s = new Set<string>();
    for (const e of events) s.add(String(e.event_type ?? ""));
    return ["all", ...Array.from(s).filter(Boolean).sort()];
  }, [events]);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    const now = Date.now() / 1000;
    const cutoff = windowSeconds ? now - windowSeconds : null;
    return events.filter((e) => {
      if (eventType !== "all" && String(e.event_type) !== eventType) return false;
      if (ctx.run !== "all" && String(e.run_id ?? "") !== ctx.run) return false;
      if (ctx.policy !== "all") {
        const pol = String((e as any).policy_name ?? (e.payload as any)?.policy_name ?? (e.payload as any)?.policy ?? "");
        if (pol !== ctx.policy) return false;
      }
      if (cutoff != null && Number(e.ts ?? 0) < cutoff) return false;
      if (!qq) return true;
      const hay = `${e.event_type} ${e.instance_id ?? ""} ${e.run_id ?? ""} ${e.message ?? ""}`.toLowerCase();
      return hay.includes(qq);
    });
  }, [events, eventType, q, ctx.run, ctx.policy, windowSeconds]);

  const EventRow = ({ index, style }: ListChildComponentProps) => {
    if (!filtered.length) {
      return (
        <div style={style} className="event muted">
          no events yet
        </div>
      );
    }
    const e = filtered[index];
    if (!e) return null;
    return (
      <div
        style={{ ...style, cursor: "pointer" }}
        className="event"
        onClick={() =>
          openContext({
            title: e.message,
            kind: "event",
            instanceId: e.instance_id ?? null,
            runId: e.run_id ?? null,
            ts: e.ts,
            details: { ...(e.payload ?? {}), step: e.step ?? (e.payload as any)?.step ?? null },
          })
        }
      >
        <span className="badge">{e.event_type}</span>
        <span>{e.message}</span>
        {e.step != null ? <span className="muted">step {fmtNum(e.step)}</span> : null}
        <span className="muted">{timeAgo(e.ts)}</span>
      </div>
    );
  };

  const countsByType = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of events) {
      const k = String(e.event_type ?? "");
      if (!k) continue;
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
  }, [events]);

  const workerRows = useMemo(() => {
    return Object.values(workers)
      .map((hb) => {
        const id = String(hb.instance_id ?? "—");
        const name = String(hb.display_name ?? id);
        const policy = String(hb.policy_name ?? "—");
        const step = Number(hb.step ?? 0);
        const score = Number(hb.steam_score ?? hb.reward ?? 0);
        const seen = Number(hb.ts ?? 0);
        const streamUrl = hb.stream_url ?? "";
        const streamOk = Boolean(hb.stream_ok);
        const streamState = !streamUrl ? "missing" : streamOk ? "ok" : "stale";
        const streamAge = hb.stream_last_frame_ts ? timeAgo(hb.stream_last_frame_ts) : "—";
        const device = String(hb.worker_device ?? "—");
        return {
          id,
          name,
          policy,
          step,
          score,
          seen,
          streamState,
          streamAge,
          streamUrl,
          streamOk,
          device,
          status: hb.status ?? "—",
          streamError: hb.stream_error ?? hb.streamer_last_error ?? "",
          streamBackend: hb.stream_backend ?? "",
          pipewireOk: hb.pipewire_node_ok,
          runId: hb.run_id ?? null,
        };
      })
      .filter((r) => {
        if (ctx.policy !== "all" && String(r.policy ?? "") !== ctx.policy) return false;
        if (ctx.run !== "all" && String(r.runId ?? "") !== ctx.run) return false;
        if (windowSeconds) {
          const now = Date.now() / 1000;
          if (Number(r.seen ?? 0) < now - windowSeconds) return false;
        }
        return true;
      })
      .sort((a, b) => b.score - a.score);
  }, [workers, ctx.policy, ctx.run, windowSeconds]);

  const streamStats = useMemo(() => {
    let ok = 0;
    let missing = 0;
    let stale = 0;
    let noPipewire = 0;
    for (const r of workerRows) {
      if (r.status === "no_pipewire" || r.pipewireOk === false) noPipewire += 1;
      if (r.streamState === "ok") ok += 1;
      else if (r.streamState === "stale") stale += 1;
      else missing += 1;
    }
    return { ok, missing, stale, noPipewire };
  }, [workerRows]);

  const streamAlerts = useMemo(() => {
    return workerRows
      .filter((r) => r.status !== "running" || r.streamState !== "ok")
      .slice(0, 6)
      .map((r) => {
        let reason = r.status !== "running" ? r.status : "";
        if (!reason && r.streamState === "missing") reason = "no_stream";
        if (!reason && r.streamState === "stale") reason = "stale_stream";
        if (r.streamError) reason = r.streamError;
        return { id: r.id, name: r.name, reason };
      });
  }, [workerRows]);

  const policyRows = useMemo(() => {
    const data = policiesQ.data ?? {};
    return Object.entries(data)
      .map(([name, rec]) => {
        const policy = String(rec?.policy_name ?? name);
        const evalScore = rec?.eval?.mean_return ?? rec?.eval_score ?? null;
        const liveScore = rec?.steam_score ?? null;
        const version = rec?.policy_version ?? null;
        const lastMutation = rec?.last_mutation_ts ?? null;
        const lastEval = rec?.eval?.last_eval_ts ?? rec?.last_eval_ts ?? null;
        const muted = rec?.pbt_muted ?? pbtMuteQ.data?.policies?.[policy] ?? false;
        const active = rec?.active_instances?.length ?? 0;
        const assigned = rec?.assigned_instances?.length ?? 0;
        return {
          policy,
          evalScore,
          liveScore,
          version,
          lastMutation,
          lastEval,
          muted,
          active,
          assigned,
          scoreSource: rec?.score_source ?? "—",
        };
      })
      .sort((a, b) => {
        const av = Number(a.evalScore ?? a.liveScore ?? 0);
        const bv = Number(b.evalScore ?? b.liveScore ?? 0);
        return bv - av;
      });
  }, [policiesQ.data, pbtMuteQ.data]);

  const health = healthQ.data;
  const overviewIssues = issuesQ.data ?? [];
  const issuesList = overviewIssues.length ? overviewIssues : issues;
  const globeNodes = useMemo(
    () =>
      Object.values(workers).map((w) => ({
        id: String(w.instance_id ?? ""),
        health: Number(w.stream_ok ? 1 : 0.4),
        active: Boolean(w.stream_ok),
      })),
    [workers]
  );

  return (
    <div className="grid overview-grid" style={{ gridTemplateColumns: "1fr 1.35fr 0.85fr" }}>
      <section className="card warroom-card" style={{ gridColumn: "1 / -1" }}>
        <div className="row-between">
          <h2>Global Hologram</h2>
          <span className="badge">{healthQ.isError ? "offline" : "live"}</span>
        </div>
        <div className="warroom-body">
          <div>
            <div className="muted">Cluster topology • orbital health rings • traffic arcs</div>
            <div className="warroom-kpis">
              <div>
                <span className="label">Active nodes</span>
                <strong>{globeNodes.length}</strong>
              </div>
              <div>
                <span className="label">Error rate</span>
                <strong>{health ? fmtPct(health.api.error_rate, 2) : "—"}</strong>
              </div>
              <div>
                <span className="label">Latency p95</span>
                <strong>{health ? `${fmtFixed(health.api.p95_ms, 1)}ms` : "—"}</strong>
              </div>
            </div>
          </div>
          <HoloGlobe nodes={globeNodes} errorRate={health?.api.error_rate ?? 0} />
        </div>
      </section>
      <section className="card">
        <div className="row-between">
          <h2>Cluster Health</h2>
          <span className="badge">{statusQ.isError ? "offline" : "live"}</span>
        </div>
        {!status ? <div className="muted">loading…</div> : (
          <div className="kpis">
            <div className="kpi">
              <div className="label">Workers</div>
              <div className="value">{fmtNum(status.workers)}</div>
            </div>
            <div className="kpi">
              <div className="label">Policies</div>
              <div className="value">{fmtNum(status.policies?.length ?? 0)}</div>
            </div>
            <div className="kpi">
              <div className="label">Now</div>
              <div className="value">{new Date(status.timestamp * 1000).toLocaleTimeString()}</div>
            </div>
          </div>
        )}
        <div className="statline">
          <span className={`dot ${statusQ.isError ? "dot-bad" : "dot-ok"}`} />
          <span className="muted">
            API {statusQ.isError ? "unreachable" : "reachable"} • last update {timeAgo((status as any)?.timestamp)}
          </span>
        </div>
        {!!status?.policies?.length && (
          <div className="panel" style={{ marginTop: 10 }}>
            <div className="muted">Policies</div>
            <div className="statline">
              {status.policies.slice(0, 14).map((p) => (
                <span key={p} className="chip">
                  {p}
                </span>
              ))}
              {status.policies.length > 14 ? <span className="muted">+{status.policies.length - 14} more</span> : null}
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Golden Signals</h2>
          <span className="badge">{healthQ.isError ? "offline" : "live"}</span>
        </div>
        {!health ? (
          <div className="muted">loading…</div>
        ) : (
          <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, marginTop: 10 }}>
            <div className="panel">
              <div className="muted">API (RED)</div>
              <div className="kpis" style={{ marginTop: 6 }}>
                <div className="kpi">
                  <div className="label">Traffic</div>
                  <div className="value">{fmtFixed(health.api.req_rate, 2)}/s</div>
                </div>
                <div className="kpi">
                  <div className="label">Latency p95</div>
                  <div className="value">{fmtFixed(health.api.p95_ms, 1)}ms</div>
                </div>
                <div className="kpi">
                  <div className="label">Errors</div>
                  <div className="value">{fmtPct(health.api.error_rate, 1)}</div>
                </div>
              </div>
            </div>
            <div className="panel">
              <div className="muted">Heartbeats</div>
              <div className="kpis" style={{ marginTop: 6 }}>
                <div className="kpi">
                  <div className="label">Rate</div>
                  <div className="value">{fmtFixed(health.heartbeat.rate, 2)}/s</div>
                </div>
                <div className="kpi">
                  <div className="label">Late</div>
                  <div className="value">{fmtNum(health.heartbeat.late)}</div>
                </div>
                <div className="kpi">
                  <div className="label">Workers</div>
                  <div className="value">{fmtNum(health.heartbeat.workers)}</div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 6 }}>TTL {fmtFixed(health.heartbeat.ttl_s, 1)}s</div>
            </div>
            <div className="panel">
              <div className="muted">Streaming</div>
              <div className="kpis" style={{ marginTop: 6 }}>
                <div className="kpi">
                  <div className="label">OK</div>
                  <div className="value">{fmtNum(health.stream.ok)}</div>
                </div>
                <div className="kpi">
                  <div className="label">Stale</div>
                  <div className="value">{fmtNum(health.stream.stale)}</div>
                </div>
                <div className="kpi">
                  <div className="label">Missing</div>
                  <div className="value">{fmtNum(health.stream.missing)}</div>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 6 }}>
                p95 frame age {health.stream.p95_frame_age_s == null ? "—" : `${fmtFixed(health.stream.p95_frame_age_s, 1)}s`}
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Live Workers</h2>
          <span className="badge">{fmtNum(workerRows.length)} online</span>
        </div>
        {!workerRows.length ? (
          <div className="muted">no workers connected</div>
        ) : (
          <table className="table table-compact" style={{ marginTop: 10 }}>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Policy</th>
                <th>Device</th>
                <th>Stream</th>
                <th>Step</th>
                <th>Seen</th>
              </tr>
            </thead>
            <tbody>
              {workerRows.slice(0, 12).map((r) => (
                <tr key={r.id}>
                  <td>
                    <div style={{ fontWeight: 800 }}>{r.name}</div>
                    <div className="muted mono">{r.id}</div>
                  </td>
                  <td>{r.policy}</td>
                  <td className="muted">{r.device}</td>
                  <td>
                    <span className={`pill ${r.streamState === "ok" ? "pill-ok" : "pill-missing"}`}>
                      {r.streamState}
                    </span>
                    <div className="muted" style={{ marginTop: 4 }}>
                      {r.streamState === "ok" ? r.streamAge : r.streamBackend || "—"}
                    </div>
                  </td>
                  <td>{fmtNum(r.step)}</td>
                  <td className="muted">{timeAgo(r.seen)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Stream Health</h2>
          <span className="badge">{workersQ.isError ? "offline" : "live"}</span>
        </div>
        <div className="kpis kpis-wrap" style={{ marginTop: 10 }}>
          <div className="kpi">
            <div className="label">Stream OK</div>
            <div className="value">{fmtNum(streamStats.ok)}</div>
          </div>
          <div className="kpi">
            <div className="label">Stale</div>
            <div className="value">{fmtNum(streamStats.stale)}</div>
          </div>
          <div className="kpi">
            <div className="label">Missing</div>
            <div className="value">{fmtNum(streamStats.missing)}</div>
          </div>
          <div className="kpi">
            <div className="label">No PipeWire</div>
            <div className="value">{fmtNum(streamStats.noPipewire)}</div>
          </div>
        </div>
        <div className="events" style={{ marginTop: 10 }}>
          {streamAlerts.map((a) => (
            <div
              key={a.id}
              className="event"
              style={{ cursor: "pointer" }}
              onClick={() =>
                openContext({
                  title: `Stream alert: ${a.name}`,
                  kind: "stream",
                  instanceId: a.id,
                  details: { reason: a.reason },
                })
              }
            >
              <span className="badge">alert</span>
              <span>{a.name}</span>
              <span className="muted">{a.reason || "issue"}</span>
            </div>
          ))}
          {!streamAlerts.length && <div className="muted">no stream alerts</div>}
        </div>
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Top Issues</h2>
          <span className="badge">{fmtNum(issuesList.length)} active</span>
        </div>
        <div className="events" style={{ marginTop: 10 }}>
          {issuesList.map((issue) => (
            <div
              key={issue.id}
              className="event"
              style={{ cursor: "pointer" }}
              onClick={() =>
                openContext({
                  title: issue.label,
                  kind: "event",
                  details: {
                    count: issue.count,
                    severity: issue.severity,
                    hint: (issue as any).hint,
                    instances: (issue as any).instances?.slice?.(0, 6) ?? [],
                  },
                })
              }
            >
              <span className="badge">{issue.label}</span>
              <span className="muted">{fmtNum(issue.count)} hits</span>
              {(issue as any).last_seen || (issue as any).lastSeen ? (
                <span className="muted">last {timeAgo((issue as any).last_seen ?? (issue as any).lastSeen)}</span>
              ) : null}
            </div>
          ))}
          {!issuesList.length && <div className="muted">no active issues</div>}
        </div>
      </section>

      <section className="card" style={{ gridColumn: "1 / -1" }}>
        <div className="row-between">
          <h2>Policy Ops</h2>
          <div className="statline">
            <span className="muted">PBT</span>
            <button
              className="btn btn-ghost"
              onClick={() => pbtMuteMut.mutate({ muted: !pbtMuteQ.data?.muted })}
              disabled={pbtMuteMut.isPending || !pbtMuteQ.data}
            >
              {pbtMuteQ.data?.muted ? "muted" : "active"}
            </button>
          </div>
        </div>
        {!policyRows.length ? (
          <div className="muted">no policies yet</div>
        ) : (
          <table className="table table-compact" style={{ marginTop: 10 }}>
            <thead>
              <tr>
                <th>Policy</th>
                <th>Version</th>
                <th>Eval</th>
                <th>Live</th>
                <th>Mutation</th>
                <th>Active</th>
                <th>PBT</th>
              </tr>
            </thead>
            <tbody>
              {policyRows.map((p) => (
                <tr key={p.policy}>
                  <td>{p.policy}</td>
                  <td className="muted">{p.version == null ? "—" : fmtNum(p.version)}</td>
                  <td>
                    <div>{p.evalScore == null ? "—" : fmtFixed(Number(p.evalScore), 3)}</div>
                    <div className="muted">{p.lastEval ? `eval ${timeAgo(p.lastEval)}` : "no eval"}</div>
                  </td>
                  <td>
                    <div>{p.liveScore == null ? "—" : fmtFixed(Number(p.liveScore), 3)}</div>
                    <div className="muted">{p.scoreSource}</div>
                  </td>
                  <td className="muted">{p.lastMutation ? timeAgo(p.lastMutation) : "—"}</td>
                  <td className="muted">{fmtNum(p.active)} / {fmtNum(p.assigned)}</td>
                  <td>
                    <button
                      className="btn btn-ghost"
                      onClick={() => pbtMuteMut.mutate({ policy_name: p.policy, muted: !p.muted })}
                      disabled={pbtMuteMut.isPending}
                    >
                      {p.muted ? "unmute" : "mute"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card" style={{ gridColumn: "1 / -1" }}>
        <div className="row-between">
          <h2>Alerts & Events</h2>
          <span className="badge">{fmtNum(events.length)} buffered</span>
        </div>
        <div className="toolbar">
          <div className="toolbar-left">
            <input className="input" placeholder="search events (type, instance, message…)" value={q} onChange={(e) => setQ(e.target.value)} />
            <select className="select" value={eventType} onChange={(e) => setEventType(e.target.value)}>
              {eventTypes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <div className="toolbar-right">
            {countsByType.slice(0, 3).map(([t, n]) => (
              <span key={t} className="pill">
                <span className="badge">{t}</span>
                <span className="muted">{fmtNum(n)}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="events" style={{ marginTop: 10, height: 360 }}>
          <AutoSizer>
            {({ height, width }) => (
              <List height={height} width={width} itemCount={filtered.length || 1} itemSize={32}>
                {EventRow}
              </List>
            )}
          </AutoSizer>
        </div>
      </section>
    </div>
  );
}
