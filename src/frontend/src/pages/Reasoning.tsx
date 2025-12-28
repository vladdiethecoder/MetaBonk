import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import PageShell from "../components/PageShell";
import QueryStateGate from "../components/QueryStateGate";
import useActivationResizeKick from "../hooks/useActivationResizeKick";
import { fetchInstances, fetchWorkerStatus, type InstanceView } from "../api";
import { fmtPct01 } from "../lib/format";

type ThoughtPacket = {
  __meta_event?: string;
  step?: number | null;
  strategy?: string;
  confidence?: number;
  content?: string;
  instance_id?: string;
  worker_id?: number;
  ts?: number;
  ts_unix?: number;
  payload?: any;
};

type ReasoningProps = {
  embedded?: boolean;
  defaultInstanceId?: string | null;
};

export default function Reasoning({ embedded = false, defaultInstanceId = null }: ReasoningProps) {
  const loc = useLocation();
  const embeddedMode = Boolean(embedded);
  const isActive = !embeddedMode && loc.pathname === "/reasoning";
  useActivationResizeKick(isActive);

  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 2000 });
  const instances = (instQ.data ?? {}) as Record<string, InstanceView>;

  const [selected, setSelected] = useState<string>("");
  const lastDefaultRef = useRef<string>("");
  const defaultId = defaultInstanceId ? String(defaultInstanceId) : "";
  useEffect(() => {
    if (embeddedMode) return;
    const qs = new URLSearchParams(loc.search);
    const id = qs.get("id") ?? qs.get("instance") ?? qs.get("iid") ?? qs.get("instance_id");
    if (id && String(id) !== selected) setSelected(String(id));
  }, [embeddedMode, loc.search, selected]);

  useEffect(() => {
    if (!embeddedMode) return;
    if (!defaultId) return;
    const last = lastDefaultRef.current;
    if (!selected || selected === last) {
      if (selected !== defaultId) setSelected(defaultId);
    }
    lastDefaultRef.current = defaultId;
  }, [embeddedMode, defaultId, selected]);

  useEffect(() => {
    if (selected) return;
    if (defaultId && instances[defaultId]) {
      setSelected(defaultId);
      return;
    }
    const ids = Object.keys(instances);
    if (ids.length) setSelected(ids[0]);
  }, [instances, selected, defaultId]);

  const selectedHb = selected ? instances[selected]?.heartbeat : null;
  const controlUrl = selectedHb?.control_url ? String(selectedHb.control_url) : "";

  const statusQ = useQuery({
    queryKey: ["workerStatus", controlUrl],
    queryFn: () => fetchWorkerStatus(controlUrl),
    enabled: Boolean(controlUrl),
    refetchInterval: 500,
  });

  const mb2 = (statusQ.data?.metabonk2 ?? null) as any;
  const plan = (mb2?.debug?.plan ?? []) as any[];

  const [thoughts, setThoughts] = useState<ThoughtPacket[]>([]);
  const tauriAvailable = typeof window !== "undefined" && Boolean((window as any).__TAURI__);

  useEffect(() => {
    if (!tauriAvailable) return;
    let unlisten: null | (() => void) = null;
    (async () => {
      try {
        const mod = await import("@tauri-apps/api/event");
        unlisten = await mod.listen<ThoughtPacket>("agent-thought", (event) => {
          const pkt = (event.payload ?? {}) as ThoughtPacket;
          setThoughts((prev) => [pkt, ...prev].slice(0, 200));
        });
      } catch {
        // Ignore: not running under Tauri, or plugin unavailable.
      }
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, [tauriAvailable]);

  const thoughtRows = useMemo(() => {
    if (!selected) return thoughts;
    const sel = String(selected);
    return thoughts.filter((t) => String(t?.instance_id ?? "") === sel);
  }, [selected, thoughts]);

  const rows = useMemo(() => {
    const ids = Object.keys(instances);
    ids.sort();
    return ids.map((id) => {
      const hb = instances[id]?.heartbeat;
      const name = String(hb?.display_name ?? id);
      const policy = String(hb?.policy_name ?? "—");
      const url = String(hb?.control_url ?? "");
      return { id, name, policy, url };
    });
  }, [instances]);

  const content = (
    <>
      <div className="card">
        <div className="row-between">
          <div style={{ flex: 1 }}>
            <div className="title">Reasoning Monitor</div>
            <div className="muted">Live System 1/System 2 + intent/skill trace (from worker /status)</div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label className="muted" style={{ fontSize: 12 }}>
              instance
            </label>
            <select className="select" value={selected} onChange={(e) => setSelected(e.target.value)}>
              {rows.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · {r.policy}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <QueryStateGate query={instQ} label="instances" />

      <div className="grid" style={{ gridTemplateColumns: embeddedMode ? "1fr" : "1fr 1fr", gap: 12 }}>
        <div className="card" style={{ gridColumn: "1 / -1" }}>
          <div className="row-between">
            <div className="title">Thought Stream</div>
            <div className="muted" style={{ marginLeft: "auto" }}>
              {tauriAvailable ? "live (Tauri)" : "run under Tauri to enable live thought packets"}
            </div>
          </div>
          {tauriAvailable && thoughtRows.length === 0 ? (
            <div className="muted" style={{ marginTop: 8 }}>
              waiting for agent-thought events…
            </div>
          ) : tauriAvailable ? (
            <div style={{ marginTop: 8, maxHeight: embeddedMode ? 220 : 260, overflow: "auto" }}>
              {thoughtRows.slice(0, 50).map((t, i) => (
                <div key={i} className="card" style={{ marginBottom: 8 }}>
                  <div className="row-between">
                    <div className="muted" style={{ fontSize: 12 }}>
                      step {t.step ?? "—"} · {String(t.instance_id ?? "—")}
                    </div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      {fmtPct01(Number(t.confidence ?? 0))}
                    </div>
                  </div>
                  <div style={{ fontWeight: 600, marginTop: 4 }}>{String(t.strategy ?? "—")}</div>
                  <div className="muted" style={{ marginTop: 4, whiteSpace: "pre-wrap" }}>
                    {String(t.content ?? "")}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="muted" style={{ marginTop: 8 }}>
              (non-Tauri mode) Thought packets are disabled.
            </div>
          )}
        </div>

        <div className="card">
          <div className="title">MetaBonk2</div>
          {!controlUrl ? (
            <div className="muted">selected instance has no control_url</div>
          ) : statusQ.isLoading ? (
            <div className="muted">loading worker status…</div>
          ) : statusQ.isError ? (
            <div className="muted">failed to fetch worker status</div>
          ) : !mb2 ? (
            <div className="muted">metabonk2 not enabled on worker</div>
          ) : (
            <div className="kv">
              <div className="k">mode</div>
              <div className="v">{String(mb2.mode ?? "—")}</div>
              <div className="k">intent</div>
              <div className="v">{String(mb2.intent ?? "—")}</div>
              <div className="k">skill</div>
              <div className="v">
                {String(mb2.skill ?? "—")}{" "}
                <span className="muted">({Number(mb2.skill_remaining ?? 0)} steps)</span>
              </div>
              <div className="k">confidence</div>
              <div className="v">{fmtPct01(Number(mb2.confidence ?? 0))}</div>
              <div className="k">novelty</div>
              <div className="v">{fmtPct01(Number(mb2.novelty ?? 0))}</div>
              <div className="k">uncertainty</div>
              <div className="v">{fmtPct01(Number(mb2.uncertainty ?? 0))}</div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="title">Plan</div>
          {!mb2 ? (
            <div className="muted">—</div>
          ) : plan.length === 0 ? (
            <div className="muted">no plan (System 1 fast path)</div>
          ) : (
            <ol>
              {plan.slice(0, 12).map((p, i) => (
                <li key={i}>
                  <span>{String(p?.name ?? "intent")}</span>{" "}
                  <span className="muted">({Number(p?.estimated_steps ?? 0)} steps)</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div className="card">
        <div className="row-between">
          <div className="title">Raw Debug</div>
          <div className="muted" style={{ marginLeft: "auto" }}>
            {controlUrl ? `${controlUrl.replace(/\/+$/, "")}/status` : "—"}
          </div>
        </div>
        <pre className="code" style={{ maxHeight: embeddedMode ? 240 : 360, overflow: "auto" }}>
          {JSON.stringify(statusQ.data ?? null, null, 2)}
        </pre>
      </div>
    </>
  );

  return embeddedMode ? <div className="reasoning-embed">{content}</div> : <PageShell>{content}</PageShell>;
}
