import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { fetchPbtMute, fetchPolicies, fetchStatus, fetchWorkers, setPbtMute } from "../api";
import { useEventStream } from "../hooks";
import useContextFilters from "../hooks/useContextFilters";
import { fmtFixed, fmtNum, timeAgo } from "../lib/format";

export default function Overview() {
  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 2000 });
  const status = statusQ.data;
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 2000 });
  const workers = workersQ.data ?? {};
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
  const { ctx, windowSeconds } = useContextFilters();
  const [eventType, setEventType] = useState<string>("all");
  const [q, setQ] = useState("");

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

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1.35fr 0.85fr" }}>
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
            <div key={a.id} className="event">
              <span className="badge">alert</span>
              <span>{a.name}</span>
              <span className="muted">{a.reason || "issue"}</span>
            </div>
          ))}
          {!streamAlerts.length && <div className="muted">no stream alerts</div>}
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
        <div className="events" style={{ marginTop: 10 }}>
          {filtered.map((e) => (
            <div key={e.event_id} className="event">
              <span className="badge">{e.event_type}</span>
              <span>{e.message}</span>
              <span className="muted">{timeAgo(e.ts)}</span>
            </div>
          ))}
          {!filtered.length && <div className="muted">no events yet</div>}
        </div>
      </section>
    </div>
  );
}
