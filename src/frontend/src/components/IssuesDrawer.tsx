import { Link, useLocation } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ackIssue, muteIssue } from "../api";
import { fmtNum, timeAgo } from "../lib/format";
import useIssues from "../hooks/useIssues";

type Props = {
  open: boolean;
  onClose: () => void;
};

export default function IssuesDrawer({ open, onClose }: Props) {
  const loc = useLocation();
  const issues = useIssues(240);
  const qc = useQueryClient();
  const ackMut = useMutation({
    mutationFn: (id: string) => ackIssue(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["overviewIssues"] }),
  });
  const muteMut = useMutation({
    mutationFn: ({ id, muted }: { id: string; muted: boolean }) => muteIssue(id, muted),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["overviewIssues"] }),
  });
  const formatEvidenceUrl = (url: string) => {
    if (!url) return url;
    if (url.startsWith("/api")) return url;
    if (url.startsWith("/")) return `/api${url}`;
    return url;
  };

  return (
    <>
      <div className={`drawer-backdrop ${open ? "open" : ""}`} onClick={onClose} />
      <aside className={`drawer ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-header">
          <div>
            <div className="drawer-title">Issues</div>
            <div className="muted">Grouped by impact</div>
          </div>
          <button className="btn btn-ghost" onClick={onClose}>
            close
          </button>
        </div>
        <div className="drawer-body">
          {!issues.length ? (
            <div className="muted">no active issues</div>
          ) : (
            issues.map((issue) => (
              <div key={issue.id} className={`issue-card issue-${issue.severity}`}>
                <div className="issue-header">
                  <span className="badge">{issue.label}</span>
                  <span className="muted">{fmtNum(issue.count)} hits</span>
                </div>
                <div className="issue-meta">
                  {issue.firstSeen ? <span>first {timeAgo(issue.firstSeen)}</span> : null}
                  {issue.lastSeen ? <span>last {timeAgo(issue.lastSeen)}</span> : null}
                </div>
                {issue.hint ? <div className="muted">{issue.hint}</div> : null}
                {issue.evidence?.length ? (
                  <div className="issue-evidence">
                    {issue.evidence.slice(0, 3).map((ev) => (
                      <a key={ev.url} className="chip chip-compact" href={formatEvidenceUrl(ev.url)} target="_blank" rel="noreferrer">
                        {ev.label ?? ev.kind}
                      </a>
                    ))}
                  </div>
                ) : null}
                {issue.instances.length ? (
                  <div className="issue-instances">
                    {issue.instances.slice(0, 6).map((iid) => (
                      <Link
                        key={iid}
                        className="chip chip-compact"
                        to={`/instances?instance=${iid}${loc.search ? `&${loc.search.slice(1)}` : ""}`}
                        onClick={onClose}
                      >
                        {iid}
                      </Link>
                    ))}
                    {issue.instances.length > 6 ? <span className="muted">+{issue.instances.length - 6} more</span> : null}
                  </div>
                ) : null}
                <div className="issue-actions">
                  <button
                    className="btn btn-ghost btn-compact"
                    onClick={() => ackMut.mutate(issue.id)}
                    disabled={ackMut.isPending}
                  >
                    {issue.acknowledged ? "acknowledged" : "ack"}
                  </button>
                  <button
                    className="btn btn-ghost btn-compact"
                    onClick={() => muteMut.mutate({ id: issue.id, muted: !issue.muted })}
                    disabled={muteMut.isPending}
                  >
                    {issue.muted ? "unmute" : "mute"}
                  </button>
                  <Link className="btn btn-ghost btn-compact" to={`/instances${loc.search || ""}`} onClick={onClose}>
                    open instances
                  </Link>
                </div>
              </div>
            ))
          )}
        </div>
      </aside>
    </>
  );
}
