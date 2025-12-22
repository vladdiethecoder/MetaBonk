import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchInstances } from "../api";
import type { InstanceView } from "../api";
import { contextDrawerEventName, ContextDrawerPayload } from "../hooks/useContextDrawer";
import { useEventStream } from "../hooks/useEventStream";
import { fmtNum, timeAgo } from "../lib/format";

export default function ContextDrawer() {
  const [open, setOpen] = useState(false);
  const [payload, setPayload] = useState<ContextDrawerPayload | null>(null);
  const events = useEventStream(240);
  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 2500 });

  useEffect(() => {
    const handler = (evt: Event) => {
      const detail = (evt as unknown as CustomEvent<ContextDrawerPayload>).detail;
      if (!detail) return;
      setPayload(detail);
      setOpen(true);
    };
    window.addEventListener(contextDrawerEventName, handler as any);
    return () => window.removeEventListener(contextDrawerEventName, handler as any);
  }, []);

  const instance = useMemo(() => {
    const iid = payload?.instanceId;
    if (!iid) return null;
    const map = (instQ.data ?? {}) as Record<string, InstanceView>;
    return map[String(iid)] ?? null;
  }, [payload?.instanceId, instQ.data]);

  const latestTelemetry = useMemo(() => {
    const hist = instance?.telemetry?.history ?? [];
    if (!hist.length) return null;
    return hist[hist.length - 1];
  }, [instance]);

  const frameUrl = useMemo(() => {
    const base = instance?.heartbeat?.control_url ?? "";
    if (!base) return "";
    return `${String(base).replace(/\\/+$/, "")}/frame.jpg`;
  }, [instance]);

  const instanceEvents = useMemo(() => {
    if (!payload?.instanceId) return [];
    return events
      .filter((e) => String(e.instance_id ?? "") === String(payload.instanceId))
      .slice(-8)
      .reverse();
  }, [events, payload?.instanceId]);

  const detailEntries = useMemo(() => Object.entries(payload?.details ?? {}).slice(0, 12), [payload?.details]);

  return (
    <>
      <div className={`drawer-backdrop ${open ? "open" : ""}`} onClick={() => setOpen(false)} />
      <aside className={`drawer ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-header">
          <div>
            <div className="drawer-title">Context</div>
            <div className="muted">{payload?.kind ?? "event"}</div>
          </div>
          <button className="btn btn-ghost" onClick={() => setOpen(false)}>
            close
          </button>
        </div>
        <div className="drawer-body">
          <div className="panel">
            <div style={{ fontWeight: 700 }}>{payload?.title ?? "detail"}</div>
            {payload?.ts ? <div className="muted">{timeAgo(payload.ts)}</div> : null}
            {payload?.instanceId ? <div className="muted mono">instance {payload.instanceId}</div> : null}
            {payload?.runId ? <div className="muted mono">run {payload.runId}</div> : null}
          </div>

          {detailEntries.length ? (
            <div className="panel">
              <div className="muted">Details</div>
              <div className="kv">
                {detailEntries.map(([k, v]) => (
                  <div key={`${k}-row`} style={{ display: "contents" }}>
                    <div className="k">{k}</div>
                    <div className="v mono">{String(v)}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          {instance ? (
            <div className="panel">
              <div className="row-between">
                <div className="muted">Instance snapshot</div>
                <span className="badge">{instance.heartbeat?.status ?? "—"}</span>
              </div>
              <div className="kv" style={{ marginTop: 6 }}>
                <div className="k">policy</div>
                <div className="v">{instance.heartbeat?.policy_name ?? "—"}</div>
                <div className="k">step</div>
                <div className="v">{fmtNum(instance.heartbeat?.step ?? 0)}</div>
                <div className="k">score</div>
                <div className="v">{instance.heartbeat?.steam_score ?? instance.heartbeat?.reward ?? "—"}</div>
                <div className="k">stream</div>
                <div className="v">{instance.heartbeat?.stream_ok ? "ok" : "stale"}</div>
              </div>
            </div>
          ) : null}

          {latestTelemetry ? (
            <div className="panel">
              <div className="muted">Telemetry (latest)</div>
              <div className="kv" style={{ marginTop: 6 }}>
                <div className="k">score</div>
                <div className="v">{fmtNum(latestTelemetry.score ?? 0)}</div>
                <div className="k">reward</div>
                <div className="v">{fmtNum(latestTelemetry.reward ?? 0)}</div>
                <div className="k">obs fps</div>
                <div className="v">{latestTelemetry.obs_fps ?? "—"}</div>
                <div className="k">act hz</div>
                <div className="v">{latestTelemetry.act_hz ?? "—"}</div>
                <div className="k">entropy</div>
                <div className="v">{latestTelemetry.action_entropy ?? "—"}</div>
                <div className="k">stream age</div>
                <div className="v">{latestTelemetry.stream_age_s == null ? "—" : `${fmtNum(latestTelemetry.stream_age_s)}s`}</div>
              </div>
            </div>
          ) : null}

          {(instance?.heartbeat?.stream_url || frameUrl) ? (
            <div className="panel">
              <div className="muted">Quick links</div>
              <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 6 }}>
                {instance?.heartbeat?.stream_url ? (
                  <a className="btn btn-ghost" href={instance.heartbeat.stream_url} target="_blank" rel="noreferrer">
                    stream
                  </a>
                ) : null}
                {frameUrl ? (
                  <a className="btn btn-ghost" href={frameUrl} target="_blank" rel="noreferrer">
                    frame.jpg
                  </a>
                ) : null}
              </div>
            </div>
          ) : null}

          <div className="panel">
            <div className="row-between">
              <div className="muted">Recent events</div>
              <span className="badge">{fmtNum(instanceEvents.length)}</span>
            </div>
            <div className="events" style={{ marginTop: 6 }}>
              {instanceEvents.map((e) => (
                <div key={e.event_id} className="event">
                  <span className="badge">{e.event_type}</span>
                  <span>{e.message}</span>
                  <span className="muted">{timeAgo(e.ts)}</span>
                </div>
              ))}
              {!instanceEvents.length && <div className="muted">no recent events</div>}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
