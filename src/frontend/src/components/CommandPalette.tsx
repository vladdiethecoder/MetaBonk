import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchInstances, fetchRuns } from "../api";
import type { InstanceView, Run } from "../api";

export default function CommandPalette() {
  const nav = useNavigate();
  const loc = useLocation();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 4000 });
  const runsQ = useQuery({ queryKey: ["runs"], queryFn: fetchRuns, refetchInterval: 8000 });
  const qs = loc.search || "";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!open) setQ("");
  }, [open]);

  const commands = useMemo(() => {
    const list: Array<{ id: string; label: string; action: () => void; meta?: string }> = [];
    list.push({ id: "open-lobby", label: "Open Lobby", action: () => nav(`/${qs}`) });
    list.push({ id: "open-neural", label: "Open Neural Interface", action: () => nav(`/neural${qs}`) });
    list.push({ id: "open-lab", label: "Open Laboratory", action: () => nav(`/lab${qs}`) });
    list.push({ id: "open-codex", label: "Open Codex", action: () => nav(`/codex${qs}`) });

    list.push({ id: "open-lab-runs", label: "Open Runs", action: () => nav(`/lab/runs${qs}`) });
    list.push({ id: "open-lab-instances", label: "Open Instances", action: () => nav(`/lab/instances${qs}`) });
    list.push({ id: "open-lab-build", label: "Open Build Lab", action: () => nav(`/lab/build${qs}`) });
    list.push({ id: "open-lab-discovery", label: "Open Discovery Inspector", action: () => nav(`/lab/discovery${qs}`) });
    list.push({ id: "open-codex-skills", label: "Open Skills", action: () => nav(`/codex/skills${qs}`) });
    list.push({ id: "open-codex-brain", label: "Open Neural Atlas", action: () => nav(`/codex/brain${qs}`) });
    list.push({ id: "open-broadcast", label: "Open Broadcast Overlay", action: () => nav(`/neural/broadcast${qs}`) });

    const instances = (instQ.data ?? {}) as Record<string, InstanceView>;
    Object.values(instances).slice(0, 40).forEach((v) => {
      const id = String(v.heartbeat?.instance_id ?? "");
      const name = String(v.heartbeat?.display_name ?? id);
      if (!id) return;
      list.push({
        id: `instance-${id}`,
        label: `Focus instance ${name}`,
        meta: id,
        action: () => nav(`/lab/instances?instance=${id}${qs ? `&${qs.slice(1)}` : ""}`),
      });
    });

    const runs = (runsQ.data ?? []) as Run[];
    runs.slice(0, 30).forEach((r) => {
      list.push({
        id: `run-${r.run_id}`,
        label: `Open run ${r.run_id}`,
        meta: r.experiment_id,
        action: () => nav(`/lab/runs?run=${r.run_id}${qs ? `&${qs.slice(1)}` : ""}`),
      });
    });
    return list;
  }, [instQ.data, runsQ.data, nav, qs]);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    if (!qq) return commands.slice(0, 16);
    return commands.filter((c) => `${c.label} ${c.meta ?? ""}`.toLowerCase().includes(qq)).slice(0, 18);
  }, [commands, q]);

  if (!open) return null;

  return (
    <div className="palette-backdrop" onClick={() => setOpen(false)}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <div className="palette-input-row">
          <input
            className="input"
            autoFocus
            placeholder="Type a command..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <span className="badge">⌘K</span>
        </div>
        <div className="palette-list">
          {filtered.map((cmd) => (
            <button
              key={cmd.id}
              className="palette-item"
              onClick={() => {
                cmd.action();
                setOpen(false);
              }}
            >
              <span>{cmd.label}</span>
              {cmd.meta ? <span className="muted mono">{cmd.meta}</span> : null}
            </button>
          ))}
          {!filtered.length && <div className="muted">no matches</div>}
        </div>
      </div>
    </div>
  );
}
