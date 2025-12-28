import { Link } from "react-router-dom";
import { isTauri } from "../lib/tauri";

export default function Discovery() {
  return (
    <div className="page discovery-page">
      <section className="card">
        <h1>Discovery</h1>
        <p className="muted">
          Autonomous discovery pipeline (phases 0–3) for building a learned action space and training-ready configs.
        </p>
        {isTauri() ? (
          <div className="row" style={{ gap: 12, flexWrap: "wrap", marginTop: 12 }}>
            <Link className="btn btn-ghost" to="/settings">
              Settings
            </Link>
            <Link className="btn btn-ghost" to="/supervisor">
              Advanced: Supervisor
            </Link>
          </div>
        ) : (
          <div className="muted" style={{ marginTop: 12 }}>
            Discovery controls require the Tauri app (`npx tauri dev`).
          </div>
        )}
      </section>
    </div>
  );
}

