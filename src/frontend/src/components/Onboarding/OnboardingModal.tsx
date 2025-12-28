import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import useLaunchConfig from "../../hooks/useLaunchConfig";
import { isTauri } from "../../lib/tauri";

type StepId = "welcome" | "synthetic-eye" | "env" | "discovery" | "train";

type Step = {
  id: StepId;
  title: string;
  subtitle: string;
};

const STORAGE_ONBOARDED = "mb:onboarded";
const steps: Step[] = [
  {
    id: "welcome",
    title: "Welcome",
    subtitle: "A guided setup for first-time MetaBonk runs.",
  },
  {
    id: "synthetic-eye",
    title: "Synthetic Eye",
    subtitle: "Confirm the zero-copy vision path is healthy.",
  },
  {
    id: "env",
    title: "Environment",
    subtitle: "Choose the environment ID for discovery + training artifacts.",
  },
  {
    id: "discovery",
    title: "Autonomous Discovery",
    subtitle: "Build a learned action space (phases 0–3).",
  },
  {
    id: "train",
    title: "Ready to Train",
    subtitle: "Start Omega with sane defaults, then iterate in Settings.",
  },
];

export default function OnboardingModal() {
  const [open, setOpen] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [launchCfg, setLaunchCfg] = useLaunchConfig();

  const step = steps[Math.min(steps.length - 1, Math.max(0, stepIdx))]!;
  const stepLabel = useMemo(() => `Step ${stepIdx + 1} / ${steps.length}`, [stepIdx]);

  useEffect(() => {
    if (!isTauri()) return;
    try {
      const v = window.localStorage.getItem(STORAGE_ONBOARDED);
      if (!v) setOpen(true);
    } catch {
      setOpen(true);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (evt: KeyboardEvent) => {
      if (evt.key === "Escape") {
        evt.preventDefault();
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const complete = (value: string) => {
    try {
      window.localStorage.setItem(STORAGE_ONBOARDED, value);
    } catch {
      // ignore
    }
    setOpen(false);
  };

  if (!open) return null;

  return (
    <div className="palette-backdrop" role="presentation">
      <div className="palette" role="dialog" aria-modal="true" aria-label="MetaBonk onboarding wizard">
        <div className="row-between" style={{ alignItems: "baseline", gap: 12 }}>
          <div>
            <div className="label">{stepLabel}</div>
            <div style={{ fontSize: 22, fontWeight: 900, letterSpacing: 0.3 }}>{step.title}</div>
            <div className="muted">{step.subtitle}</div>
          </div>
          <div className="row" style={{ gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
            <button className="btn btn-ghost btn-compact" onClick={() => complete("skipped")}>
              skip
            </button>
          </div>
        </div>

        <div className="card" style={{ marginTop: 10 }}>
          {step.id === "welcome" ? (
            <div className="card-body">
              <div className="muted">
                This wizard sets up the recommended path: Synthetic Eye → discovery artifacts → training.
              </div>
              <div className="muted" style={{ marginTop: 10 }}>
                You can re-run everything later from <Link to="/settings">Settings</Link> and{" "}
                <Link to="/discovery">Discovery</Link>.
              </div>
            </div>
          ) : null}

          {step.id === "synthetic-eye" ? (
            <div className="card-body">
              <div className="muted">
                Synthetic Eye is the GPU-only, zero-copy observation path (DMA-BUF + fences → CUDA import).
              </div>
              <div className="muted" style={{ marginTop: 10 }}>
                Run the bench in <Link to="/settings">Settings</Link> to validate ingest FPS.
              </div>
            </div>
          ) : null}

          {step.id === "env" ? (
            <div className="card-body">
              <label className="field" style={{ margin: 0 }}>
                <div className="label">env id</div>
                <input
                  value={launchCfg.envId}
                  onChange={(e) => setLaunchCfg((prev) => ({ ...prev, envId: e.target.value }))}
                  placeholder="MegaBonk"
                  autoFocus
                />
              </label>
              <div className="muted" style={{ marginTop: 10 }}>
                This key is used for discovery cache paths and is forwarded to Omega as `METABONK_ENV_ID`.
              </div>
            </div>
          ) : null}

          {step.id === "discovery" ? (
            <div className="card-body">
              <div className="muted">
                Discovery builds `cache/discovery/&lt;env_id&gt;/*` artifacts used to compress the action space.
              </div>
              <div className="row" style={{ gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                <Link className="btn btn-ghost" to={`/discovery?env=${encodeURIComponent(launchCfg.envId)}`}>
                  open discovery
                </Link>
              </div>
            </div>
          ) : null}

          {step.id === "train" ? (
            <div className="card-body">
              <div className="muted">Start Omega from the top bar (or from Settings for advanced flags).</div>
              <div className="row" style={{ gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                <Link className="btn btn-ghost" to="/settings">
                  open settings
                </Link>
                <Link className="btn btn-ghost" to="/agents">
                  open agents
                </Link>
              </div>
            </div>
          ) : null}
        </div>

        <div className="row-between" style={{ marginTop: 10, alignItems: "center" }}>
          <button
            className="btn btn-ghost"
            onClick={() => setStepIdx((v) => Math.max(0, v - 1))}
            disabled={stepIdx === 0}
          >
            back
          </button>
          <div className="row" style={{ gap: 10 }}>
            {stepIdx < steps.length - 1 ? (
              <button className="btn" onClick={() => setStepIdx((v) => Math.min(steps.length - 1, v + 1))}>
                next
              </button>
            ) : (
              <button className="btn" onClick={() => complete("1")}>
                finish
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
