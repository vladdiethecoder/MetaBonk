import { Link } from "react-router-dom";
import { isTauri } from "../lib/tauri";

export default function Settings() {
  return (
    <div className="page settings-page">
      <section className="card">
        <h1>Settings</h1>
        <p className="muted">Synthetic Eye defaults, launch configuration, and power-user toggles.</p>
        <div className="row" style={{ gap: 12, flexWrap: "wrap", marginTop: 12 }}>
          <Link className="btn btn-ghost" to="/discovery">
            Discovery
          </Link>
          <Link className="btn btn-ghost" to="/analytics">
            Analytics
          </Link>
          {isTauri() ? (
            <Link className="btn btn-ghost" to="/supervisor">
              Advanced: Supervisor
            </Link>
          ) : null}
        </div>
      </section>
    </div>
  );
}

