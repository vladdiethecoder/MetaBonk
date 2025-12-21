import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { fetchRuns, Run } from "../api";
import useContextFilters from "../hooks/useContextFilters";
import { copyToClipboard, fmtFixed, fmtNum, timeAgo } from "../lib/format";

export default function Runs() {
  const runsQ = useQuery({ queryKey: ["runs"], queryFn: fetchRuns, refetchInterval: 5000 });
  const runs = (runsQ.data ?? []) as Run[];
  const { ctx, windowSeconds } = useContextFilters();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<string>("all");
  const [sort, setSort] = useState<"updated" | "best" | "step">("updated");
  const [selected, setSelected] = useState<string | null>(null);

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

  return (
    <div className="grid" style={{ gridTemplateColumns: "1.2fr 0.8fr" }}>
      <section className="card">
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
          </div>
          <div className="toolbar-right">
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

        <table className="table table-hover" style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Run</th>
              <th>Experiment</th>
              <th>Status</th>
              <th>Best</th>
              <th>Last</th>
              <th>Step</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr
                key={r.run_id}
                className={String(selectedRun?.run_id) === String(r.run_id) ? "active" : ""}
                onClick={() => setSelected(r.run_id)}
                style={{ cursor: "pointer" }}
              >
                <td className="mono">{r.run_id}</td>
                <td className="mono">{r.experiment_id}</td>
                <td>
                  <span className="pill">{r.status}</span>
                </td>
                <td>{fmtFixed(r.best_reward, 2)}</td>
                <td>{fmtFixed(r.last_reward, 2)}</td>
                <td>{fmtNum(r.last_step)}</td>
                <td className="muted">{timeAgo(r.updated_ts)}</td>
              </tr>
            ))}
            {!filtered.length && (
              <tr>
                <td colSpan={7} className="muted">
                  no runs yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Run Detail</h2>
          <span className="badge">{selectedRun ? "selected" : "pick one"}</span>
        </div>
        {!selectedRun ? (
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
          </>
        )}
      </section>
    </div>
  );
}
