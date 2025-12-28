import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import useLaunchConfig from "../../hooks/useLaunchConfig";
import { isTauri } from "../../lib/tauri";
import { fmtFixed } from "../../lib/format";
import { tauriInvoke, type BenchResult, type DiscoveryStatus } from "../../lib/tauri_api";
import { useTauriRuntime } from "../../tauri/RuntimeProvider";

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
    subtitle: "Start Omega with sane defaults, then iterate in the Laboratory.",
  },
];

export default function OnboardingModal() {
  const { tauriReady, omegaRunning, discoveryRunning } = useTauriRuntime();
  const [open, setOpen] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [launchCfg, setLaunchCfg] = useLaunchConfig();
  const [benchBusy, setBenchBusy] = useState(false);
  const [benchResult, setBenchResult] = useState<BenchResult | null>(null);
  const [discoveryStatus, setDiscoveryStatus] = useState<DiscoveryStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    if (!tauriReady || !open) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await tauriInvoke<DiscoveryStatus>("discovery_status", { env_id: launchCfg.envId });
        if (!cancelled) setDiscoveryStatus(status);
      } catch {
        if (!cancelled) setDiscoveryStatus(null);
      }
    };
    poll();
    const t = window.setInterval(poll, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [tauriReady, open, launchCfg.envId]);

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

  const runBench = async () => {
    if (!tauriReady || benchBusy) return;
    setError(null);
    setBenchBusy(true);
    setBenchResult(null);
    try {
      const res = await tauriInvoke<BenchResult>("run_synthetic_eye_bench", {
        width: 1280,
        height: 720,
        lockstep: launchCfg.syntheticEyeLockstep,
        seconds: 3.0,
        full_bridge: true,
      });
      setBenchResult(res);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBenchBusy(false);
    }
  };

  const runDiscovery = async () => {
    if (!tauriReady || discoveryRunning) return;
    setError(null);
    try {
      await tauriInvoke("start_discovery", {
        env_id: launchCfg.envId,
        env_adapter: launchCfg.syntheticEye ? "synthetic-eye" : "mock",
        exploration_budget: 2000,
        hold_frames: 30,
      });
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const stopDiscovery = async () => {
    if (!tauriReady || !discoveryRunning) return;
    setError(null);
    try {
      await tauriInvoke("stop_discovery");
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  const startOmega = async () => {
    if (!tauriReady) return;
    setError(null);
    try {
      await tauriInvoke("start_omega", {
        mode: launchCfg.mode,
        workers: launchCfg.workers,
        env_id: launchCfg.envId,
        synthetic_eye: launchCfg.syntheticEye,
        synthetic_eye_lockstep: launchCfg.syntheticEyeLockstep,
        obs_backend: launchCfg.obsBackend || null,
        use_discovered_actions: launchCfg.useDiscoveredActions,
        silicon_cortex: launchCfg.siliconCortex,
      });
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
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
                You can re-run everything later from{" "}
                <Link to="/" onClick={() => setOpen(false)}>
                  the Lobby
                </Link>{" "}
                and{" "}
                <Link to="/lab" onClick={() => setOpen(false)}>
                  the Laboratory
                </Link>
                .
              </div>
            </div>
          ) : null}

          {step.id === "synthetic-eye" ? (
            <div className="card-body">
              <div className="muted">
                Synthetic Eye is the GPU-only, zero-copy observation path (DMA-BUF + fences → CUDA import).
              </div>
              <div className="row" style={{ gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                <button className="btn btn-ghost" disabled={!tauriReady || benchBusy} onClick={runBench}>
                  {benchBusy ? "running…" : "run 3s bench"}
                </button>
                <Link className="btn btn-ghost" to="/" onClick={() => setOpen(false)}>
                  open lobby
                </Link>
              </div>
              {benchResult ? (
                <div className="kv" style={{ marginTop: 12 }}>
                  <div className="k">fps</div>
                  <div className="v">{benchResult.fps ? benchResult.fps.toFixed(1) : "—"}</div>
                  <div className="k">p50</div>
                  <div className="v">
                    {benchResult.per_frame_ms?.p50 != null ? `${fmtFixed(benchResult.per_frame_ms.p50, 2)}ms` : "—"}
                  </div>
                  <div className="k">p95</div>
                  <div className="v">
                    {benchResult.per_frame_ms?.p95 != null ? `${fmtFixed(benchResult.per_frame_ms.p95, 2)}ms` : "—"}
                  </div>
                </div>
              ) : null}
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
              <div className="pipeline" style={{ marginTop: 12 }}>
                <div className="pipeline-node">
                  <div className="pipeline-title">Phase 0</div>
                  <div className="muted">Input enumeration</div>
                  <div className="pill">{discoveryStatus?.phase_0_complete ? "✓" : "—"}</div>
                </div>
                <div className="pipeline-node">
                  <div className="pipeline-title">Phase 1</div>
                  <div className="muted">Effect mapping</div>
                  <div className="pill">{discoveryStatus?.phase_1_complete ? "✓" : "—"}</div>
                </div>
                <div className="pipeline-node">
                  <div className="pipeline-title">Phase 2</div>
                  <div className="muted">Clustering</div>
                  <div className="pill">{discoveryStatus?.phase_2_complete ? "✓" : "—"}</div>
                </div>
                <div className="pipeline-node">
                  <div className="pipeline-title">Phase 3</div>
                  <div className="muted">Action space</div>
                  <div className="pill">{discoveryStatus?.phase_3_complete ? "✓" : "—"}</div>
                </div>
              </div>
              <div className="row" style={{ gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                <button className="btn btn-ghost" disabled={!tauriReady || discoveryRunning} onClick={runDiscovery}>
                  {discoveryRunning ? "running…" : "run phases 0–3"}
                </button>
                <button className="btn btn-ghost" disabled={!tauriReady || !discoveryRunning} onClick={stopDiscovery}>
                  stop
                </button>
                <Link className="btn btn-ghost" to="/lab/discovery" onClick={() => setOpen(false)}>
                  open artifacts
                </Link>
              </div>
            </div>
          ) : null}

          {step.id === "train" ? (
            <div className="card-body">
              <div className="muted">
                Start Omega now, then inspect live streams and the mind panel in Neural Interface.
              </div>
              <div className="row" style={{ gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                <button className="btn btn-ghost" disabled={!tauriReady || omegaRunning} onClick={startOmega}>
                  {omegaRunning ? "omega running" : "start omega"}
                </button>
                <Link className="btn btn-ghost" to="/lab" onClick={() => setOpen(false)}>
                  open lab
                </Link>
                <Link className="btn btn-ghost" to="/neural" onClick={() => setOpen(false)}>
                  open neural
                </Link>
              </div>
            </div>
          ) : null}
        </div>

        {error ? (
          <div className="card" style={{ marginTop: 10, borderColor: "#ff8b5a" }}>
            <div className="label">error</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{error}</div>
          </div>
        ) : null}

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
