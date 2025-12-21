import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { fetchHistoricLeaderboard, fetchInstances, HistoricLeaderboardEntry, InstanceView } from "../api";
import useContextFilters from "../hooks/useContextFilters";
import { clamp01, copyToClipboard, fmtFixed, fmtNum, fmtPct01, timeAgo } from "../lib/format";

export default function Instances() {
  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 2000 });
  const histQ = useQuery({ queryKey: ["historicLeaderboard"], queryFn: () => fetchHistoricLeaderboard(200, "best_score"), refetchInterval: 5000 });
  const instances = (instQ.data ?? {}) as Record<string, InstanceView>;
  const loc = useLocation();
  const { ctx, windowSeconds } = useContextFilters();
  const [q, setQ] = useState("");
  const [policy, setPolicy] = useState<string>("all");
  const [sort, setSort] = useState<"score" | "step" | "seen">("score");
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const qs = new URLSearchParams(loc.search);
    const id = qs.get("id") ?? qs.get("instance") ?? qs.get("iid");
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
      const streamState = !streamUrl ? "missing" : streamOk ? "ok" : "stale";
      const streamError = hb?.stream_error ?? hb?.streamer_last_error ?? null;
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
        survival,
        danger,
        cfg: v.config ?? null,
        best_score: hist?.best_score ?? null,
        best_step: hist?.best_step ?? null,
        runId,
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

  return (
    <div className="grid" style={{ gridTemplateColumns: "1.3fr 0.7fr" }}>
      <section className="card">
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

        <table className="table table-hover" style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Policy</th>
              <th>Device</th>
              <th>Stream</th>
              <th>Score</th>
              <th>Step</th>
              <th>Survival</th>
              <th>Danger</th>
              <th>Seen</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.id}
                className={String(selectedRow?.id) === String(r.id) ? "active" : ""}
                onClick={() => setSelected(r.id)}
                style={{ cursor: "pointer" }}
              >
                <td>
                  <div style={{ fontWeight: 800 }}>{r.name}</div>
                  <div className="muted mono">{r.id}</div>
                </td>
                <td>{r.policy}</td>
                <td className="muted">{r.device}</td>
                <td>
                  <span className={`pill ${r.streamState === "ok" ? "pill-ok" : "pill-missing"}`}>{r.streamState}</span>
                </td>
                <td>
                  <div>{fmtFixed(r.score, 2)}</div>
                  <div className="muted">best {r.best_score == null ? "—" : fmtFixed(r.best_score, 2)}</div>
                </td>
                <td>
                  <div>{fmtNum(r.step)}</div>
                  <div className="muted">peak {r.best_step == null ? "—" : fmtNum(r.best_step)}</div>
                </td>
                <td>{r.survival == null ? "—" : fmtPct01(r.survival, 0)}</td>
                <td>{r.danger == null ? "—" : fmtPct01(r.danger, 0)}</td>
                <td className="muted">{timeAgo(r.ts)}</td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={9} className="muted">
                  no instances yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
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
              <div className="k">device</div>
              <div className="v">{selectedRow.device}</div>
              <div className="k">stream</div>
              <div className="v">
                <span className={`pill ${selectedRow.streamState === "ok" ? "pill-ok" : "pill-missing"}`}>{selectedRow.streamState}</span>
                {selectedRow.streamError ? <span className="muted"> • {selectedRow.streamError}</span> : null}
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
                    <span className="muted">{selectedRow.hb?.stream_type ?? "—"}</span>
                  </div>
                </div>
              ) : (
                <div className="muted stream-placeholder">stream unavailable</div>
              )}
            </div>

            <details className="details">
              <summary className="muted">Raw heartbeat</summary>
              <pre className="code code-tall">{JSON.stringify(selectedRow.hb ?? {}, null, 2)}</pre>
            </details>
            {selectedRow.cfg ? (
              <details className="details">
                <summary className="muted">Assigned config</summary>
                <pre className="code code-tall">{JSON.stringify(selectedRow.cfg ?? {}, null, 2)}</pre>
              </details>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}
