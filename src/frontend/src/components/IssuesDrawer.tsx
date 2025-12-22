import { Link } from "react-router-dom";
import { fmtNum, timeAgo } from "../lib/format";
import useIssues from "../hooks/useIssues";

type Props = {
  open: boolean;
  onClose: () => void;
};

export default function IssuesDrawer({ open, onClose }: Props) {
  const issues = useIssues(240);

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
                {issue.instances.length ? (
                  <div className="issue-instances">
                    {issue.instances.slice(0, 6).map((iid) => (
                      <Link key={iid} className="chip chip-compact" to={`/instances?instance=${iid}`} onClick={onClose}>
                        {iid}
                      </Link>
                    ))}
                    {issue.instances.length > 6 ? <span className="muted">+{issue.instances.length - 6} more</span> : null}
                  </div>
                ) : null}
                <div className="issue-actions">
                  <Link className="btn btn-ghost btn-compact" to="/instances" onClick={onClose}>
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
