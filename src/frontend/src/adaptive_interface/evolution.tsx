import { useEffect, useMemo } from "react";
import useLocalStorageState from "../hooks/useLocalStorageState";

export type EvolutionGenome = {
  seed: number;
  complexity: number; // 0..1
  novelty: number; // 0..1
  stability: number; // 0..1
  layout: "grid" | "constellation" | "list";
  palette: "neon" | "mono" | "ember";
};

type EvolutionProps = {
  onGenome?: (genome: EvolutionGenome) => void;
};

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

const defaultGenome = (): EvolutionGenome => ({
  seed: Math.floor(Date.now() / 1000) >>> 0,
  complexity: 0.55,
  novelty: 0.35,
  stability: 0.7,
  layout: "grid",
  palette: "neon",
});

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mutateGenome(prev: EvolutionGenome): EvolutionGenome {
  const rand = mulberry32((prev.seed ^ 0x9e3779b9) >>> 0);
  const jitter = (x: number, scale: number) => clamp01(x + (rand() * 2 - 1) * scale);
  const layouts: EvolutionGenome["layout"][] = ["grid", "constellation", "list"];
  const palettes: EvolutionGenome["palette"][] = ["neon", "mono", "ember"];
  const maybePick = <T,>(xs: T[], cur: T, p: number) => (rand() < p ? xs[Math.floor(rand() * xs.length)] : cur);
  return {
    ...prev,
    seed: (prev.seed + 1 + Math.floor(rand() * 1024)) >>> 0,
    complexity: jitter(prev.complexity, 0.12),
    novelty: jitter(prev.novelty, 0.18),
    stability: jitter(prev.stability, 0.12),
    layout: maybePick(layouts, prev.layout, 0.35),
    palette: maybePick(palettes, prev.palette, 0.25),
  };
}

function phenotype(genome: EvolutionGenome): { label: string; hint: string }[] {
  const density =
    genome.layout === "list" ? 0.55 + 0.35 * genome.complexity : genome.layout === "grid" ? 0.65 + 0.3 * genome.complexity : 0.5 + 0.45 * genome.complexity;
  const exploration = 0.25 + 0.65 * genome.novelty;
  const friction = 0.15 + 0.75 * genome.stability;
  const accent =
    genome.palette === "neon"
      ? "pink/blue"
      : genome.palette === "ember"
        ? "amber/red"
        : "mono";
  return [
    { label: "density", hint: `${Math.round(density * 100)}% (${genome.layout})` },
    { label: "exploration", hint: `${Math.round(exploration * 100)}%` },
    { label: "stability", hint: `${Math.round(friction * 100)}%` },
    { label: "accent", hint: accent },
  ];
}

export default function Evolution({ onGenome }: EvolutionProps) {
  const [genome, setGenome] = useLocalStorageState<EvolutionGenome>("mb:singularity:evolution", defaultGenome());

  useEffect(() => {
    onGenome?.(genome);
  }, [genome, onGenome]);

  const traits = useMemo(() => phenotype(genome), [genome]);

  return (
    <div>
      <div className="card-header">
        <div>
          <h3>Adaptive Evolution</h3>
          <p className="muted">A small “interface genome” you can mutate to steer how the Singularity surface behaves.</p>
        </div>
        <button className="btn btn-ghost" onClick={() => setGenome((g) => mutateGenome(g))}>
          mutate
        </button>
      </div>
      <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="lab-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
          <label>
            <div className="muted" style={{ fontSize: 12 }}>
              complexity
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={genome.complexity}
              onChange={(e) => setGenome((g) => ({ ...g, complexity: clamp01(Number(e.target.value)) }))}
              style={{ width: "100%" }}
            />
          </label>
          <label>
            <div className="muted" style={{ fontSize: 12 }}>
              novelty
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={genome.novelty}
              onChange={(e) => setGenome((g) => ({ ...g, novelty: clamp01(Number(e.target.value)) }))}
              style={{ width: "100%" }}
            />
          </label>
          <label>
            <div className="muted" style={{ fontSize: 12 }}>
              stability
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={genome.stability}
              onChange={(e) => setGenome((g) => ({ ...g, stability: clamp01(Number(e.target.value)) }))}
              style={{ width: "100%" }}
            />
          </label>
          <label>
            <div className="muted" style={{ fontSize: 12 }}>
              layout
            </div>
            <select value={genome.layout} onChange={(e) => setGenome((g) => ({ ...g, layout: e.target.value as any }))}>
              <option value="grid">grid</option>
              <option value="constellation">constellation</option>
              <option value="list">list</option>
            </select>
          </label>
          <label>
            <div className="muted" style={{ fontSize: 12 }}>
              palette
            </div>
            <select value={genome.palette} onChange={(e) => setGenome((g) => ({ ...g, palette: e.target.value as any }))}>
              <option value="neon">neon</option>
              <option value="ember">ember</option>
              <option value="mono">mono</option>
            </select>
          </label>
        </div>

        <div className="lab-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
          {traits.map((t) => (
            <div key={t.label} className="card" style={{ padding: 10 }}>
              <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em" }}>
                {t.label}
              </div>
              <div style={{ fontWeight: 800 }}>{t.hint}</div>
            </div>
          ))}
        </div>

        <div className="muted" style={{ fontSize: 12 }}>
          seed: <span className="mono">{genome.seed}</span>
        </div>
      </div>
    </div>
  );
}

