import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  buildDreamPolicy,
  buildWorldModel,
  fetchHistoricLeaderboard,
  fetchPretrainStatus,
  fetchPretrainJobs,
  fetchSkillToken,
  fetchSkillsSummary,
  fetchWorkers,
  generateSkillNames,
  HistoricLeaderboardEntry,
  PretrainJob,
  PretrainStatus,
  SkillsSummary,
  SkillTokenDetail,
} from "../api";
import { useEventStream } from "../hooks";
import { Link } from "react-router-dom";
import { fmtFixed, timeAgo } from "../lib/format";

function fmtNum(n: number | null | undefined) {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

function seqFeatures(seq: number[][]) {
  const T = seq.length;
  const D = T ? seq[0].length : 0;
  if (!T || !D) return { T, D, mean: [] as number[], meanAbs: [] as number[], maxAbs: [] as number[] };
  const mean = Array(D).fill(0) as number[];
  const meanAbs = Array(D).fill(0) as number[];
  const maxAbs = Array(D).fill(0) as number[];
  for (let t = 0; t < T; t++) {
    const row = seq[t] ?? [];
    for (let d = 0; d < D; d++) {
      const v = Number(row[d] ?? 0);
      mean[d] += v;
      const av = Math.abs(v);
      meanAbs[d] += av;
      if (av > maxAbs[d]) maxAbs[d] = av;
    }
  }
  for (let d = 0; d < D; d++) {
    mean[d] /= T;
    meanAbs[d] /= T;
  }
  return { T, D, mean, meanAbs, maxAbs };
}

function ArtifactRow({ label, a }: { label: string; a: PretrainStatus["artifacts"][keyof PretrainStatus["artifacts"]] }) {
  const exists = Boolean((a as any)?.exists);
  const path = String((a as any)?.path ?? "");
  const meta = (a as any)?.meta as Record<string, unknown> | undefined;
  const metaTxt =
    meta && Object.keys(meta).length
      ? Object.entries(meta)
          .map(([k, v]) => `${k}=${String(v)}`)
          .join(" · ")
      : "—";
  return (
    <tr>
      <td>{label}</td>
      <td className="muted">{path || "—"}</td>
      <td>
        <span className={exists ? "pill pill-ok" : "pill pill-missing"}>{exists ? "present" : "missing"}</span>
      </td>
      <td className="muted">{metaTxt}</td>
    </tr>
  );
}

function JobRow({ j, onSelect }: { j: PretrainJob; onSelect: (id: string) => void }) {
  const cls = j.status === "succeeded" ? "pill pill-ok" : j.status === "failed" ? "pill pill-missing" : "pill";
  return (
    <tr onClick={() => onSelect(j.job_id)} style={{ cursor: "pointer" }}>
      <td className="mono">{j.job_id}</td>
      <td>{j.kind}</td>
      <td>
        <span className={cls}>{j.status}</span>
      </td>
      <td className="muted">{timeAgo(j.started_ts)}</td>
      <td className="muted">{j.ended_ts ? timeAgo(j.ended_ts) : "—"}</td>
      <td className="muted">{j.returncode ?? "—"}</td>
    </tr>
  );
}

export default function Spy() {
  const qc = useQueryClient();
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [selectionLocked, setSelectionLocked] = useState(false);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);

  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 1000 });
  const histQ = useQuery({ queryKey: ["historicLeaderboard"], queryFn: () => fetchHistoricLeaderboard(200, "best_score"), refetchInterval: 5000 });
  const pretrainQ = useQuery({ queryKey: ["pretrainStatus"], queryFn: fetchPretrainStatus, refetchInterval: 5000 });
  const pretrainJobsQ = useQuery({ queryKey: ["pretrainJobs"], queryFn: fetchPretrainJobs, refetchInterval: 2000 });
  const skillsQ = useQuery({ queryKey: ["skillsSummary"], queryFn: fetchSkillsSummary, refetchInterval: 5000 });
  const tokenQ = useQuery({
    queryKey: ["skillToken", selectedToken],
    queryFn: () => fetchSkillToken(selectedToken as number),
    enabled: selectedToken !== null,
    refetchInterval: 5000,
  });
  const events = useEventStream(80);

  const workers = Object.values(workersQ.data ?? {});
  workers.sort((a, b) => (b.steam_score ?? b.reward ?? 0) - (a.steam_score ?? a.reward ?? 0));
  const histById = useMemo(() => {
    const m: Record<string, HistoricLeaderboardEntry> = {};
    for (const r of (histQ.data ?? []) as HistoricLeaderboardEntry[]) {
      if (!r?.instance_id) continue;
      m[String(r.instance_id)] = r;
    }
    return m;
  }, [histQ.data]);

  const pretrain = pretrainQ.data as PretrainStatus | undefined;
  const skills = skillsQ.data as SkillsSummary | undefined;
  const pretrainJobs = (pretrainJobsQ.data ?? []) as PretrainJob[];
  const dreamJob = pretrainJobs.find((j) => j.kind === "dream_policy" && j.status === "running") ?? null;
  const wmJob = pretrainJobs.find((j) => j.kind === "world_model" && j.status === "running") ?? null;
  const selectedJobObj = useMemo(() => pretrainJobs.find((j) => j.job_id === selectedJob) ?? null, [pretrainJobs, selectedJob]);

  const wmFormat = String((pretrain?.artifacts?.world_model_ckpt as any)?.meta?.format ?? "");
  const worldModelOk = Boolean(pretrain?.artifacts?.world_model_ckpt?.exists) && wmFormat !== "legacy";
  const ptRolloutsOk = Number(pretrain?.datasets?.video_rollouts_pt ?? 0) > 0;

  const topTokens = skills?.dataset?.token_top ?? [];

  useEffect(() => {
    if (selectionLocked) return;
    const qs = new URLSearchParams(window.location.search);
    const raw = qs.get("token") ?? qs.get("t");
    if (raw) {
      const tok = Number(raw);
      if (Number.isFinite(tok)) {
        setSelectedToken(tok);
        setSelectionLocked(true);
      }
    }
  }, [selectionLocked]);

  useEffect(() => {
    if (selectionLocked) return;
    if (selectedToken !== null) return;
    const first = topTokens[0]?.token;
    if (typeof first === "number") setSelectedToken(first);
  }, [selectionLocked, selectedToken, topTokens]);

  const genNames = useMutation({
    mutationFn: () => generateSkillNames(32, false),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["skillsSummary"] });
    },
  });

  const buildDream = useMutation({
    mutationFn: () => buildDreamPolicy(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["pretrainJobs"] });
      qc.invalidateQueries({ queryKey: ["pretrainStatus"] });
    },
  });

  const buildWm = useMutation({
    mutationFn: () => buildWorldModel(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["pretrainJobs"] });
      qc.invalidateQueries({ queryKey: ["pretrainStatus"] });
    },
  });

  const tokenDetail = tokenQ.data as SkillTokenDetail | undefined;
  const tokenFeat = useMemo(() => (tokenDetail ? seqFeatures(tokenDetail.decoded_action_seq ?? []) : null), [tokenDetail]);

  return (
    <div className="grid">
      <section className="card">
        <div className="row-between">
          <h2>Pretrain Status</h2>
          <span className="badge">disk-backed</span>
        </div>
        {!pretrain ? (
          <div className="muted">loading…</div>
        ) : (
          <>
            {(wmFormat === "legacy" || !pretrain.artifacts.world_model_ckpt.exists) && (
              <div className="row-between" style={{ marginTop: 8 }}>
                <div className="muted">
                  World Model is {pretrain.artifacts.world_model_ckpt.exists ? "present but incompatible" : "missing"}.
                  {wmFormat === "legacy" ? " (format=legacy)" : ""}
                  {!ptRolloutsOk ? " Add `.pt` rollouts first." : ""}
                  {wmJob ? ` (job ${wmJob.job_id} running)` : ""}
                </div>
                <button className="btn" onClick={() => buildWm.mutate()} disabled={buildWm.isPending || Boolean(wmJob) || !ptRolloutsOk}>
                  {wmJob ? "building…" : buildWm.isPending ? "starting…" : "build world model"}
                </button>
              </div>
            )}

            {!pretrain.artifacts.dream_policy_ckpt.exists && (
              <div className="row-between" style={{ marginTop: 8 }}>
                <div className="muted">
                  Dream Policy is missing. If you already have `.pt` rollouts and a World Model checkpoint, you can build it here.
                  {dreamJob ? ` (job ${dreamJob.job_id} running)` : ""}
                </div>
                <button className="btn" onClick={() => buildDream.mutate()} disabled={buildDream.isPending || Boolean(dreamJob) || !ptRolloutsOk || !worldModelOk}>
                  {dreamJob ? "building…" : buildDream.isPending ? "starting…" : "build dream policy"}
                </button>
              </div>
            )}

            <div className="kpis kpis-wrap">
              <div className="kpi">
                <div className="label">Video Demos</div>
                <div className="value">{fmtNum(pretrain.datasets.video_demos_npz)}</div>
              </div>
              <div className="kpi">
                <div className="label">Labeled Demos</div>
                <div className="value">{fmtNum(pretrain.datasets.video_labeled_npz)}</div>
              </div>
              <div className="kpi">
                <div className="label">PT Rollouts</div>
                <div className="value">{fmtNum(pretrain.datasets.video_rollouts_pt)}</div>
              </div>
            </div>

            <div className="split">
              <div>
                <div className="muted">Datasets</div>
                <div className="kv">
                  <div className="k">video demos</div>
                  <div className="v">{pretrain.datasets.video_demos_dir}</div>
                  <div className="k">labeled demos</div>
                  <div className="v">{pretrain.datasets.video_labeled_dir}</div>
                  <div className="k">pt rollouts</div>
                  <div className="v">{pretrain.datasets.video_rollouts_pt_dir}</div>
                </div>
              </div>
              <div>
                <div className="muted">Artifacts</div>
                <div className="kv">
                  <div className="k">skill names</div>
                  <div className="v">
                    {pretrain.artifacts.skill_names.path}{" "}
                    <span className={pretrain.artifacts.skill_names.exists ? "pill pill-ok" : "pill pill-missing"}>
                      {pretrain.artifacts.skill_names.exists ? "present" : "missing"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <table className="table table-compact" style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>Artifact</th>
                  <th>Path</th>
                  <th>Status</th>
                  <th>Meta</th>
                </tr>
              </thead>
              <tbody>
                <ArtifactRow label="IDM" a={pretrain.artifacts.idm_ckpt} />
                <ArtifactRow label="Reward RM" a={pretrain.artifacts.reward_ckpt} />
                <ArtifactRow label="Skill VQ-VAE" a={pretrain.artifacts.skill_ckpt} />
                <ArtifactRow label="World Model" a={pretrain.artifacts.world_model_ckpt} />
                <ArtifactRow label="Dream Policy" a={pretrain.artifacts.dream_policy_ckpt} />
              </tbody>
            </table>

            <div className="split">
              <div>
                <div className="muted">Pretrain Jobs</div>
                <div className="muted">Click a row to view logs.</div>
                <table className="table table-compact table-hover" style={{ marginTop: 10 }}>
                  <thead>
                    <tr>
                      <th>Job</th>
                      <th>Kind</th>
                      <th>Status</th>
                      <th>Started</th>
                      <th>Ended</th>
                      <th>RC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pretrainJobs.slice(0, 8).map((j) => (
                      <JobRow key={j.job_id} j={j} onSelect={(id) => setSelectedJob(id)} />
                    ))}
                    {!pretrainJobs.length ? (
                      <tr>
                        <td colSpan={6} className="muted">
                          no jobs yet
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
              <div>
                <div className="row-between">
                  <div className="muted">Job Log</div>
                  <button className="btn btn-ghost" onClick={() => qc.invalidateQueries({ queryKey: ["pretrainJobs"] })}>
                    refresh
                  </button>
                </div>
                {!selectedJobObj ? (
                  <div className="muted">Select a job to view its output.</div>
                ) : (
                  <>
                    <div className="kv">
                      <div className="k">job</div>
                      <div className="v mono">{selectedJobObj.job_id}</div>
                      <div className="k">kind</div>
                      <div className="v">{selectedJobObj.kind}</div>
                      <div className="k">status</div>
                      <div className="v">{selectedJobObj.status}</div>
                      <div className="k">started</div>
                      <div className="v">{timeAgo(selectedJobObj.started_ts)}</div>
                      <div className="k">ended</div>
                      <div className="v">{selectedJobObj.ended_ts ? timeAgo(selectedJobObj.ended_ts) : "—"}</div>
                    </div>
                    <pre className="code code-tall" style={{ marginTop: 10 }}>
                      {(selectedJobObj.log ?? []).slice(-200).join("\n") || "—"}
                    </pre>
                  </>
                )}
              </div>
            </div>
          </>
        )}
      </section>

      <section className="card">
        <h2>Live Workers</h2>
        {workersQ.isError ? (
          <div className="muted">Failed to load `/api/workers`.</div>
        ) : (
          <>
            <div className="muted">This is what the cluster is reporting right now.</div>
            <table className="table table-hover" style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>Instance</th>
                  <th>Policy</th>
                  <th>Step</th>
                  <th>Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {workers.slice(0, 20).map((w) => {
                  const h = histById[String(w.instance_id)];
                  return (
                    <tr key={w.instance_id}>
                      <td>
                        <Link className="mono" to={`/instances?id=${encodeURIComponent(String(w.instance_id))}`}>
                          {w.display_name ?? w.instance_id}
                        </Link>
                        <div className="muted mono">{w.instance_id}</div>
                      </td>
                      <td>{w.policy_name ?? "—"}</td>
                      <td>
                        <div>{fmtNum(w.step)}</div>
                        <div className="muted">peak {h?.best_step ?? "—"}</div>
                      </td>
                      <td>
                        <div>{fmtFixed(w.steam_score ?? w.reward ?? 0, 2)}</div>
                        <div className="muted">best {h?.best_score == null ? "—" : fmtFixed(h.best_score, 2)}</div>
                      </td>
                      <td className="muted">{w.status}</td>
                    </tr>
                  );
                })}
                {!workers.length && (
                  <tr>
                    <td colSpan={5} className="muted">
                      no workers connected
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section className="card">
        <h2>Skill Tokens (Top)</h2>
        {!skills ? (
          <div className="muted">loading…</div>
        ) : skills.dataset?.error ? (
          <div className="muted">dataset error: {skills.dataset.error}</div>
        ) : (
          <>
            <div className="row-between">
              <div className="muted">Derived from labeled video demos (`/api/skills/summary`).</div>
              <button className="btn" onClick={() => genNames.mutate()} disabled={genNames.isPending}>
                {genNames.isPending ? "naming…" : "generate names"}
              </button>
            </div>
            <table className="table table-compact" style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>Token</th>
                  <th>Name</th>
                  <th>Count</th>
                  <th>Avg Reward</th>
                </tr>
              </thead>
              <tbody>
                {topTokens.slice(0, 12).map((t) => (
                  <tr
                    key={t.token}
                    className={selectedToken === t.token ? "active" : ""}
                    onClick={() => setSelectedToken(t.token)}
                    style={{ cursor: "pointer" }}
                  >
                    <td>{t.token}</td>
                    <td>{t.name ?? "—"}</td>
                    <td>{fmtNum(t.count)}</td>
                    <td>{t.avg_reward.toFixed(3)}</td>
                  </tr>
                ))}
                {!topTokens.length && (
                  <tr>
                    <td colSpan={4} className="muted">
                      no labeled skill tokens found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </>
        )}
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Token Detail</h2>
          <div className="row-inline">
            <span className="badge">{selectedToken === null ? "select one" : `token ${selectedToken}`}</span>
            {selectedToken !== null ? (
              <Link className="btn btn-ghost" to={`/skills?t=${encodeURIComponent(String(selectedToken))}`}>
                open in Skills
              </Link>
            ) : null}
          </div>
        </div>
        {selectedToken === null ? (
          <div className="muted">Click a token row to inspect its decoded action sequence and stats.</div>
        ) : tokenQ.isError ? (
          <div className="muted">Failed to load `/api/skills/token/{selectedToken}`.</div>
        ) : !tokenDetail ? (
          <div className="muted">loading…</div>
        ) : (
          <>
            <div className="kv">
              <div className="k">name</div>
              <div className="v">{tokenDetail.name ?? "—"}</div>
              <div className="k">subtitle</div>
              <div className="v">{tokenDetail.subtitle ?? "—"}</div>
              <div className="k">tags</div>
              <div className="v">{tokenDetail.tags?.length ? tokenDetail.tags.join(", ") : "—"}</div>
              <div className="k">count</div>
              <div className="v">{tokenDetail.stat?.count ?? "—"}</div>
              <div className="k">avg reward</div>
              <div className="v">{tokenDetail.stat ? tokenDetail.stat.avg_reward.toFixed(4) : "—"}</div>
              <div className="k">seq</div>
              <div className="v">
                {tokenFeat ? `${tokenFeat.T} steps × ${tokenFeat.D} dims` : "—"}
              </div>
            </div>

            {tokenFeat && tokenFeat.D > 0 && (
              <table className="table table-compact" style={{ marginTop: 10 }}>
                <thead>
                  <tr>
                    <th>Dim</th>
                    <th>Mean</th>
                    <th>Mean |x|</th>
                    <th>Max |x|</th>
                  </tr>
                </thead>
                <tbody>
                  {tokenFeat.mean.map((m, i) => (
                    <tr key={i}>
                      <td className="muted">a{i}</td>
                      <td>{m.toFixed(4)}</td>
                      <td>{tokenFeat.meanAbs[i].toFixed(4)}</td>
                      <td>{tokenFeat.maxAbs[i].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <div className="details">
              <div className="muted">Decoded action sequence (first 12 steps)</div>
              <pre className="code">{JSON.stringify(tokenDetail.decoded_action_seq.slice(0, 12), null, 2)}</pre>
            </div>
          </>
        )}
      </section>

      <section className="card">
        <h2>Events</h2>
        <div className="events">
          {events.map((e) => (
            <div key={e.event_id} className="event">
              <span className="badge">{e.event_type}</span>
              <span>{e.message}</span>
              <span className="muted">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
            </div>
          ))}
          {!events.length && <div className="muted">no events yet</div>}
        </div>
      </section>
    </div>
  );
}
