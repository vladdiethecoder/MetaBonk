import { Link } from "react-router-dom";

export default function Analytics() {
  return (
    <div className="page analytics-page">
      <section className="card">
        <h1>Analytics</h1>
        <p className="muted">Runs, instances, learned skills, and deep diagnostics.</p>
      </section>
      <div className="lab-grid" style={{ marginTop: 12 }}>
        <section className="card">
          <h3>Runs</h3>
          <p className="muted">Training curves, artifacts, and comparisons.</p>
          <Link to="/runs" className="btn btn-ghost">
            Open Runs
          </Link>
        </section>
        <section className="card">
          <h3>Instances</h3>
          <p className="muted">Live workers, telemetry, and stream integrity.</p>
          <Link to="/instances" className="btn btn-ghost">
            Open Instances
          </Link>
        </section>
        <section className="card">
          <h3>Skills</h3>
          <p className="muted">Skill token atlas, prototypes, and effect stats.</p>
          <Link to="/skills" className="btn btn-ghost">
            Open Skills
          </Link>
        </section>
        <section className="card">
          <h3>NeuroSynaptic</h3>
          <p className="muted">Deep neural diagnostics and interactive views.</p>
          <Link to="/brain" className="btn btn-ghost">
            Open Neural Atlas
          </Link>
        </section>
        <section className="card">
          <h3>Build Lab</h3>
          <p className="muted">Architecture experiments and hyperparameter probes.</p>
          <Link to="/build" className="btn btn-ghost">
            Open Build Lab
          </Link>
        </section>
        <section className="card">
          <h3>Supervisor</h3>
          <p className="muted">Advanced process control and Tauri logs.</p>
          <Link to="/supervisor" className="btn btn-ghost">
            Open Supervisor
          </Link>
        </section>
      </div>
    </div>
  );
}

