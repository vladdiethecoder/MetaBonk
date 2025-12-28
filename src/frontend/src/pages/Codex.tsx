import { Link } from "react-router-dom";

export default function Codex() {
  return (
    <div className="page codex-page">
      <section className="card">
        <h1>Codex</h1>
        <p className="muted">Interpretability surfaces: skills, semantics, and neural atlas views.</p>
      </section>

      <div className="lab-grid">
        <section className="card">
          <h3>Skill Library</h3>
          <p className="muted">Browse learned skill tokens and prototypes.</p>
          <Link to="/codex/skills" className="btn btn-ghost">
            Open Skills
          </Link>
        </section>
        <section className="card">
          <h3>NeuroSynaptic</h3>
          <p className="muted">Deep neural diagnostics and interactive atlas.</p>
          <Link to="/codex/brain" className="btn btn-ghost">
            Open Neural Atlas
          </Link>
        </section>
      </div>
    </div>
  );
}

