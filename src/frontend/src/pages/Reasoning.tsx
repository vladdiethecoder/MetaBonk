import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import PageShell from "../components/PageShell";
import QueryStateGate from "../components/QueryStateGate";
import useActivationResizeKick from "../hooks/useActivationResizeKick";
import { fetchInstances, fetchWorkerStatus, type InstanceView } from "../api";
import { fmtPct01 } from "../lib/format";

export default function Reasoning() {
  const loc = useLocation();
  const isActive = loc.pathname === "/reasoning";
  useActivationResizeKick(isActive);

  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 2000 });
  const instances = (instQ.data ?? {}) as Record<string, InstanceView>;

  const [selected, setSelected] = useState<string>("");
  useEffect(() => {
    const qs = new URLSearchParams(loc.search);
    const id = qs.get("id") ?? qs.get("instance") ?? qs.get("iid") ?? qs.get("instance_id");
    if (id && String(id) !== selected) setSelected(String(id));
  }, [loc.search, selected]);

  useEffect(() => {
    if (selected) return;
    const ids = Object.keys(instances);
    if (ids.length) setSelected(ids[0]);
  }, [instances, selected]);

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

  return (
    <PageShell>
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

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
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
        <pre className="code" style={{ maxHeight: 360, overflow: "auto" }}>
          {JSON.stringify(statusQ.data ?? null, null, 2)}
        </pre>
      </div>
    </PageShell>
  );
}
