import { useMemo, useState } from "react";
import PageShell from "../components/PageShell";
import useLaunchConfig from "../hooks/useLaunchConfig";
import Evolution, { type EvolutionGenome } from "../adaptive_interface/evolution";
import SemanticCore, { type SemanticIntent } from "../semantic_interaction/core";
import HolographicRenderer from "../holographic_viz/renderer";
import Anticipatory from "../predictive/anticipatory";

type TelemetryLite = {
  cpu?: { usage_pct?: number };
  gpu?: { util_pct?: number; temp_c?: number };
};

export default function Singularity() {
  const [launchCfg] = useLaunchConfig();
  const [genome, setGenome] = useState<EvolutionGenome | null>(null);
  const [lastIntent, setLastIntent] = useState<SemanticIntent | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryLite | null>(null);

  const seed = genome?.seed ?? 1337;
  const layout = genome?.layout ?? "grid";

  const headline = useMemo(() => {
    if (!genome) return "Prototype surface: semantic control + telemetry + holographic viz.";
    const parts = [
      `layout=${genome.layout}`,
      `complexity=${Math.round(genome.complexity * 100)}%`,
      `novelty=${Math.round(genome.novelty * 100)}%`,
      `stability=${Math.round(genome.stability * 100)}%`,
    ];
    return parts.join(" • ");
  }, [genome]);

  return (
    <PageShell className="singularity-page">
      <section className="card">
        <h1>Singularity</h1>
        <p className="muted">{headline}</p>
        <div className="lab-grid" style={{ marginTop: 10 }}>
          <div className="card" style={{ padding: 10 }}>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
              env
            </div>
            <div style={{ fontWeight: 800 }}>{launchCfg.envId}</div>
          </div>
          <div className="card" style={{ padding: 10 }}>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
              workers
            </div>
            <div style={{ fontWeight: 800 }}>{launchCfg.workers}</div>
          </div>
          <div className="card" style={{ padding: 10 }}>
            <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
              last intent
            </div>
            <div className="mono" style={{ fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {lastIntent ? lastIntent.kind : "—"}
            </div>
          </div>
        </div>
      </section>

      <div className="lab-grid" style={layout === "list" ? { gridTemplateColumns: "1fr" } : undefined}>
        <section className="card">
          <Evolution onGenome={setGenome} />
        </section>
        <section className="card">
          <SemanticCore envId={launchCfg.envId} defaultWorkers={launchCfg.workers} onIntent={setLastIntent} />
        </section>
        <section className="card">
          <Anticipatory onTelemetry={(t) => setTelemetry(t as any)} />
        </section>
        <section className="card" style={{ gridColumn: layout === "grid" ? "1 / -1" : undefined }}>
          <div className="card-header">
            <div>
              <h3>Holographic Visualization</h3>
              <p className="muted">A lightweight “cognitive field” renderer (seeded + modulated by telemetry).</p>
            </div>
          </div>
          <div className="card-body">
            <HolographicRenderer seed={seed} telemetry={telemetry} />
            <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
              seed: <span className="mono">{seed}</span>
            </div>
          </div>
        </section>
      </div>
    </PageShell>
  );
}

