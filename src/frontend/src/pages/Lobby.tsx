import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchOverviewHealth, fetchOverviewIssues, fetchStatus, fetchWorkers } from "../api";
import { fmtFixed, fmtNum, fmtPct01, timeAgo } from "../lib/format";

export default function Lobby() {
  const statusQ = useQuery({ queryKey: ["status"], queryFn: fetchStatus, refetchInterval: 3000 });
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 3000 });
  const healthQ = useQuery({
    queryKey: ["overviewHealth"],
    queryFn: () => fetchOverviewHealth(240),
    refetchInterval: 5000,
  });
  const issuesQ = useQuery({
    queryKey: ["overviewIssues"],
    queryFn: () => fetchOverviewIssues(600),
    refetchInterval: 6000,
  });

  const workers = Object.values(workersQ.data ?? {});
  const running = workers.filter((w) => w.status === "running");
  const lastSeen = workers
    .map((w) => w.ts)
    .filter(Boolean)
    .sort((a, b) => b - a)[0];
  const health = healthQ.data;
  const issues = issuesQ.data ?? [];
  const streamTotal = health ? health.stream.ok + health.stream.stale + health.stream.missing : 0;
  const streamOkPct = health && streamTotal ? Math.round((health.stream.ok / streamTotal) * 100) : null;

  return (
    <div className="page lobby-page">
      <section className="card lobby-hero">
        <div>
          <h1>Lobby</h1>
          <p className="muted">
            High-level health, worker status, and launch readiness. Use the Omega control in the top bar to start training or gameplay.
          </p>
          <div className="lobby-links">
            <Link to="/agents" className="btn btn-ghost">
              Agents
            </Link>
            <Link to="/discovery" className="btn btn-ghost">
              Discovery
            </Link>
            <Link to="/analytics" className="btn btn-ghost">
              Analytics
            </Link>
            <Link to="/stream" className="btn btn-ghost">
              Stream
            </Link>
            <Link to="/supervisor" className="btn btn-ghost">
              Supervisor
            </Link>
          </div>
        </div>
        <div className="kpis kpis-wrap">
          <div className="kpi">
            <div className="label">System</div>
            <div className="value">{statusQ.isError ? "offline" : "live"}</div>
          </div>
          <div className="kpi">
            <div className="label">Workers</div>
            <div className="value">{fmtNum(workers.length)}</div>
          </div>
          <div className="kpi">
            <div className="label">Running</div>
            <div className="value">{fmtNum(running.length)}</div>
          </div>
          <div className="kpi">
            <div className="label">Last pulse</div>
            <div className="value">{lastSeen ? timeAgo(lastSeen) : "—"}</div>
          </div>
        </div>
      </section>

      <div className="lobby-grid">
        <section className="card flex-card">
          <h3>Worker Health</h3>
          <div className="muted">{workers.length ? "Live worker status" : "No workers detected yet"}</div>
          <div className="lobby-worker-list">
            {workers.length === 0 ? (
              <div className="muted">Start Omega to populate worker metrics.</div>
            ) : (
              workers.slice(0, 8).map((w) => (
                <div key={w.instance_id ?? w.display_name ?? w.policy_name} className="lobby-worker">
                  <div>
                    <strong>{w.display_name ?? w.instance_id ?? "Worker"}</strong>
                    <span className="muted">{w.policy_name ?? "policy"}</span>
                  </div>
                  <div className={`badge ${w.status === "running" ? "" : "warn"}`}>{w.status ?? "idle"}</div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="card flex-card">
          <h3>System Telemetry</h3>
          <div className="kpis kpis-wrap">
            <div className="kpi">
              <div className="label">API p95</div>
              <div className="value">{health ? `${fmtFixed(health.api.p95_ms, 1)}ms` : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Error rate</div>
              <div className="value">{health ? fmtPct01(health.api.error_rate, 1) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Heartbeat</div>
              <div className="value">{health ? fmtFixed(health.heartbeat.rate, 2) + "/s" : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Late nodes</div>
              <div className="value">{health ? fmtNum(health.heartbeat.late) : "—"}</div>
            </div>
          </div>
          <div className="muted" style={{ marginTop: 8 }}>
            TTL {health ? fmtFixed(health.heartbeat.ttl_s, 1) : "—"}s
          </div>
        </section>

        <section className="card flex-card">
          <h3>Stream Integrity</h3>
          <div className="kpis kpis-wrap">
            <div className="kpi">
              <div className="label">OK</div>
              <div className="value">{health ? fmtNum(health.stream.ok) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Stale</div>
              <div className="value">{health ? fmtNum(health.stream.stale) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Missing</div>
              <div className="value">{health ? fmtNum(health.stream.missing) : "—"}</div>
            </div>
            <div className="kpi">
              <div className="label">Quality</div>
              <div className="value">{streamOkPct == null ? "—" : `${streamOkPct}%`}</div>
            </div>
          </div>
          <div className="muted" style={{ marginTop: 8 }}>
            p95 frame age {health?.stream.p95_frame_age_s == null ? "—" : `${fmtFixed(health.stream.p95_frame_age_s, 1)}s`}
          </div>
        </section>

        <section className="card flex-card">
          <h3>Active Alerts</h3>
          {issues.length === 0 ? (
            <div className="muted">No incidents in the last 10 minutes.</div>
          ) : (
            <div className="events">
              {issues.slice(0, 6).map((issue) => (
                <div key={issue.id} className="event">
                  <span className="badge">{issue.severity}</span>
                  <div>
                    <strong>{issue.label}</strong>
                    <div className="muted">{issue.hint ?? "Anomaly detected"}</div>
                  </div>
                  <span className="muted" style={{ marginLeft: "auto" }}>
                    {issue.last_seen ? timeAgo(issue.last_seen) : "recent"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

