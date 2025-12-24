import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  fetchSkillEffects,
  fetchSkillPrototypes,
  fetchSkillTimeline,
  fetchSkillToken,
  fetchSkillsAtlas,
  fetchSkillsFiles,
  fetchSkillsSummary,
  generateSkillNames,
  updateSkillName,
  SkillAtlasPoint,
  SkillEffectToken,
  SkillEffectsResponse,
  SkillPrototypesResponse,
  SkillTimelineResponse,
  SkillTokenDetail,
  SkillsAtlasResponse,
  SkillsFilesResponse,
  SkillsSummary,
} from "../api";
import { copyToClipboard } from "../lib/format";

function fmtPct(v: number | null | undefined, digits = 1) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtNum(v: number | null | undefined) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toLocaleString();
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

const hashString = (input: string) => {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

function AtlasStarfield({
  atlas,
  selectedToken,
  onSelect,
}: {
  atlas: SkillsAtlasResponse | null;
  selectedToken: number | null;
  onSelect: (tok: number) => void;
}) {
  const points = useMemo(() => {
    const pts = atlas?.points ?? [];
    return pts.map((p) => {
      const seed = hashString(String(p.token));
      const z = ((seed % 1000) / 1000) * 2 - 1;
      return { ...p, z };
    });
  }, [atlas]);

  const ptsGeom = useMemo(() => {
    const arr = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      arr[i * 3 + 0] = p.x;
      arr[i * 3 + 1] = p.y;
      arr[i * 3 + 2] = p.z;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [points]);

  const RotatingField = () => {
    const ref = useRef<THREE.Points | null>(null);
    useFrame((state) => {
      if (!ref.current) return;
      ref.current.rotation.y = state.clock.getElapsedTime() * 0.15;
    });
    return (
      <points ref={ref} geometry={ptsGeom}>
        <pointsMaterial color="#7bffe6" size={0.02} sizeAttenuation />
      </points>
    );
  };

  return (
    <div className="atlas-starfield">
      <Canvas className="atlas-r3f-canvas" dpr={[1, 2]} camera={{ position: [0, 0, 2.6], fov: 50 }}>
        <color attach="background" args={["#020405"]} />
        <ambientLight intensity={0.8} />
        <RotatingField />
      </Canvas>
      <div className="atlas-starfield-label">Latent Constellation</div>
      <div className="atlas-starfield-hint">navigate clusters · click in 2D atlas to select</div>
    </div>
  );
}

function TokenHelix({ detail }: { detail: SkillTokenDetail }) {
  const seq = detail.decoded_action_seq ?? [];
  const Helix = () => {
    const groupRef = useRef<THREE.Group | null>(null);
    useFrame((state) => {
      if (!groupRef.current) return;
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.4;
    });
    const turns = 6;
    const seqLen = Math.max(seq.length, 120);
    const points = Array.from({ length: seqLen }, (_, i) => {
      const phase = (i / seqLen) * Math.PI * 2 * turns;
      const y = -1 + (i / (seqLen - 1)) * 2;
      return {
        a: new THREE.Vector3(Math.cos(phase) * 0.6, y, Math.sin(phase) * 0.1),
        b: new THREE.Vector3(Math.cos(phase + Math.PI) * 0.6, y, Math.sin(phase + Math.PI) * 0.1),
        hue: (i * 7) % 360,
      };
    });
    return (
      <group ref={groupRef}>
        {points.map((p, i) => (
          <group key={`helix-${i}`}>
            <mesh position={p.a}>
              <sphereGeometry args={[0.04, 10, 10]} />
              <meshStandardMaterial color={`hsl(${p.hue},90%,65%)`} />
            </mesh>
            <mesh position={p.b}>
              <sphereGeometry args={[0.04, 10, 10]} />
              <meshStandardMaterial color={`hsl(${p.hue},90%,65%)`} />
            </mesh>
          </group>
        ))}
      </group>
    );
  };

  return (
    <div className="token-helix">
      <Canvas className="token-helix-canvas" dpr={[1, 2]} camera={{ position: [0, 0, 3], fov: 50 }}>
        <color attach="background" args={["#020405"]} />
        <ambientLight intensity={0.8} />
        <Helix />
      </Canvas>
      <div className="token-helix-label">Token DNA Helix</div>
    </div>
  );
}

function ActionSeqChart({ detail, maxDims = 6 }: { detail: SkillTokenDetail; maxDims?: number }) {
  const seq = detail.decoded_action_seq ?? [];
  if (!seq.length) return <div className="muted">no decoded actions</div>;

  const seqLen = seq.length;
  const actionDim = seq[0]?.length ?? 0;
  const dims = Math.max(1, Math.min(maxDims, actionDim));

  const w = 900;
  const h = 240;
  const pad = 16;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;

  let vmin = Number.POSITIVE_INFINITY;
  let vmax = Number.NEGATIVE_INFINITY;
  for (let t = 0; t < seqLen; t++) {
    for (let d = 0; d < dims; d++) {
      const v = seq[t][d];
      if (!Number.isFinite(v)) continue;
      vmin = Math.min(vmin, v);
      vmax = Math.max(vmax, v);
    }
  }
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) {
    vmin = -1;
    vmax = 1;
  }
  if (Math.abs(vmax - vmin) < 1e-6) {
    vmin -= 1;
    vmax += 1;
  }

  const colors = ["#7dd3fc", "#a78bfa", "#fb7185", "#34d399", "#fbbf24", "#60a5fa", "#f472b6"];

  const polylines = Array.from({ length: dims }, (_, d) => {
    const pts: string[] = [];
    for (let t = 0; t < seqLen; t++) {
      const x = pad + (t / Math.max(1, seqLen - 1)) * innerW;
      const yNorm = (seq[t][d] - vmin) / (vmax - vmin);
      const y = pad + (1 - clamp01(yNorm)) * innerH;
      pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
    }
    return { d, points: pts.join(" "), color: colors[d % colors.length] };
  });

  return (
    <div className="chart">
      <div className="chart-head">
        <div className="muted">Decoded action sequence (first {dims} dims)</div>
        <div className="legend">
          {polylines.map((p) => (
            <span key={p.d} className="legend-item">
              <span className="legend-swatch" style={{ background: p.color }} />
              a{p.d}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg" role="img" aria-label="decoded action chart">
        <rect x="0" y="0" width={w} height={h} rx="10" fill="#050709" stroke="#1b1f22" />
        <line x1={pad} y1={h / 2} x2={w - pad} y2={h / 2} stroke="#1b1f22" />
        {polylines.map((p) => (
          <polyline
            key={p.d}
            points={p.points}
            fill="none"
            stroke={p.color}
            strokeWidth="2"
            opacity={0.95}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
      </svg>
      <div className="muted">
        Range: [{vmin.toFixed(2)}, {vmax.toFixed(2)}] · len={seqLen} · dim={actionDim}
      </div>
    </div>
  );
}

function SummaryKpis({ summary }: { summary: SkillsSummary }) {
  const sv = summary.skill_vqvae ?? { available: false };
  const ds = summary.dataset ?? { available: false };

  return (
    <div className="kpis kpis-wrap">
      <div className="kpi">
        <div className="label">Codes</div>
        <div className="value">{sv.num_codes ?? "—"}</div>
      </div>
      <div className="kpi">
        <div className="label">Seq Len</div>
        <div className="value">{sv.seq_len ?? "—"}</div>
      </div>
      <div className="kpi">
        <div className="label">Action Dim</div>
        <div className="value">{sv.action_dim ?? "—"}</div>
      </div>
      <div className="kpi">
        <div className="label">Utilization</div>
        <div className="value">{fmtPct(sv.codebook_utilization, 1)}</div>
      </div>
      <div className="kpi">
        <div className="label">Entropy</div>
        <div className="value">{sv.usage_entropy?.toFixed?.(2) ?? "—"}</div>
      </div>
      <div className="kpi">
        <div className="label">Steps</div>
        <div className="value">{fmtNum(ds.total_steps)}</div>
      </div>
      <div className="kpi">
        <div className="label">Labeled Files</div>
        <div className="value">{fmtNum(ds.labeled_files)}</div>
      </div>
    </div>
  );
}

function clamp(x: number, a: number, b: number) {
  return Math.max(a, Math.min(b, x));
}

function numOr(v: any, fallback: number) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function skillDotRadius(p: SkillAtlasPoint) {
  const u = p.usage != null && Number.isFinite(p.usage) ? clamp(Number(p.usage), 0, 1) : null;
  const c = p.count != null && Number.isFinite(p.count) ? Math.max(0, Number(p.count)) : null;
  if (u != null) return 2 + Math.sqrt(u) * 7;
  if (c != null) return 2 + Math.sqrt(c) * 0.04;
  return 2.5;
}

function skillDotColor(p: SkillAtlasPoint) {
  const ar = p.avg_reward != null && Number.isFinite(p.avg_reward) ? Number(p.avg_reward) : null;
  if (ar == null) return "rgba(255,255,255,0.38)";
  // Map [-1,1] -> red..green with a bit of neon bias.
  const t = clamp((ar + 1) / 2, 0, 1);
  const r = Math.round(255 * (1 - t) + 45 * t);
  const g = Math.round(60 * (1 - t) + 255 * t);
  const b = Math.round(120 * (1 - t) + 90 * t);
  return `rgba(${r},${g},${b},0.85)`;
}

function SkillAtlas({
  atlas,
  selectedToken,
  cursorToken,
  onSelect,
}: {
  atlas: SkillsAtlasResponse | null;
  selectedToken: number | null;
  cursorToken: number | null;
  onSelect: (tok: number) => void;
}) {
  const pts = atlas?.points ?? [];
  if (!pts.length) return <div className="muted">atlas unavailable</div>;

  return (
    <div>
      <div className="muted">Skill Atlas ({atlas?.method ?? "proj"})</div>
      <svg className="atlas-svg" viewBox="-1.05 -1.05 2.1 2.1" role="img" aria-label="skill atlas">
        <rect x={-1.05} y={-1.05} width={2.1} height={2.1} rx={0.12} fill="rgba(5,7,9,0.9)" stroke="rgba(255,255,255,0.10)" />
        {pts.map((p) => {
          const active = selectedToken === p.token;
          const cursor = cursorToken === p.token;
          const r = skillDotRadius(p);
          const fill = skillDotColor(p);
          const stroke = cursor ? "rgba(255,230,0,0.95)" : active ? "rgba(0,229,255,0.95)" : "rgba(0,0,0,0)";
          const sw = cursor || active ? 0.03 : 0;
          const opacity = cursor || active ? 1 : 0.65;
          return (
            <circle
              key={p.token}
              cx={clamp(p.x, -1.1, 1.1)}
              cy={clamp(p.y, -1.1, 1.1)}
              r={r / 120}
              fill={fill}
              opacity={opacity}
              stroke={stroke}
              strokeWidth={sw}
              onClick={() => onSelect(p.token)}
              style={{ cursor: "pointer" }}
            >
              <title>
                {p.token}
                {p.name ? ` · ${p.name}` : ""} · usage {p.usage != null ? fmtPct(p.usage, 2) : "—"} · count {p.count ?? "—"} · avgR{" "}
                {p.avg_reward != null ? p.avg_reward.toFixed(3) : "—"}
              </title>
            </circle>
          );
        })}
      </svg>
      <div className="muted" style={{ marginTop: 8 }}>
        Click a dot to select · outline = selected/cursor
      </div>
    </div>
  );
}

function TokenTimeline({
  timeline,
  cursor,
  setCursor,
  onSelectToken,
}: {
  timeline: SkillTimelineResponse | null;
  cursor: number;
  setCursor: (i: number) => void;
  onSelectToken: (tok: number) => void;
}) {
  const toks = timeline?.tokens ?? [];
  const rows = timeline?.top_tokens ?? [];
  const N = toks.length;
  if (!N || !rows.length) return <div className="muted">timeline unavailable</div>;

  const rowH = 12;
  const H = rows.length * rowH + 24;
  const W = 860;

  const cursorClamped = clamp(cursor, 0, Math.max(0, N - 1));
  const cursorTok = toks[cursorClamped];

  const segmentsByTok = useMemo(() => {
    const m = new Map<number, Array<[number, number]>>();
    for (const tok of rows) m.set(tok, []);
    let runTok = toks[0];
    let runStart = 0;
    for (let i = 1; i <= N; i++) {
      const t = i < N ? toks[i] : NaN;
      if (t !== runTok) {
        const segs = m.get(runTok);
        if (segs) segs.push([runStart, i]);
        runTok = t as any;
        runStart = i;
      }
    }
    return m;
  }, [rows, toks, N]);

  return (
    <div>
      <div className="row-between">
        <div className="muted">Token Timeline</div>
        <div className="row-inline">
          <span className="badge">
            step {timeline?.step0 ?? 0}…{timeline?.step1 ?? 0} (×{timeline?.stride ?? 1})
          </span>
          <span className="badge">cursor tok {cursorTok}</span>
        </div>
      </div>

      <div className="timeline-scrub">
        <input
          className="input"
          type="range"
          min={0}
          max={Math.max(0, N - 1)}
          value={cursorClamped}
          onChange={(e) => setCursor(Number(e.target.value))}
        />
        <button className="btn btn-ghost" onClick={() => onSelectToken(cursorTok)}>
          select cursor token
        </button>
      </div>

      <svg
        className="timeline-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="token timeline"
        onClick={(e) => {
          const rect = (e.currentTarget as any).getBoundingClientRect?.();
          if (!rect) return;
          const x = clamp((e.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
          setCursor(Math.round(x * (N - 1)));
        }}
        style={{ cursor: "crosshair" }}
      >
        <rect x="0" y="0" width={W} height={H} rx="10" fill="rgba(5,7,9,0.9)" stroke="rgba(255,255,255,0.10)" />
        {rows.map((tok, ri) => {
          const y = 12 + ri * rowH;
          const segs = segmentsByTok.get(tok) ?? [];
          return (
            <g key={tok} onClick={() => onSelectToken(tok)} style={{ cursor: "pointer" }}>
              <text x="10" y={y + 9} fontSize="10" fill="rgba(255,255,255,0.65)">
                {tok}
              </text>
              {segs.map(([a, b], si) => {
                const x = 70 + (a / N) * (W - 86);
                const w = ((b - a) / N) * (W - 86);
                return <rect key={si} x={x} y={y} width={Math.max(1, w)} height={9} rx="3" fill="rgba(0,229,255,0.55)" />;
              })}
            </g>
          );
        })}
        {/* cursor */}
        <line x1={70 + (cursorClamped / N) * (W - 86)} y1={8} x2={70 + (cursorClamped / N) * (W - 86)} y2={H - 8} stroke="rgba(255,230,0,0.85)" />
      </svg>
    </div>
  );
}

function CausalityStrip({
  effects,
  activeToken,
}: {
  effects: SkillEffectsResponse | null;
  activeToken: number | null;
}) {
  const rows = effects?.tokens ?? [];
  const active = activeToken == null ? null : rows.find((r) => r.token === activeToken) ?? null;
  const top = rows.slice(0, 8);

  const pill = (t: SkillEffectToken) => {
    const r = t.reward_mean != null ? t.reward_mean.toFixed(3) : "—";
    const nxt = t.reward_next_sum_mean != null ? t.reward_next_sum_mean.toFixed(2) : "—";
    const dims = t.dominant_dims?.slice?.(0, 3)?.map?.((d) => `a${d}`)?.join?.(", ") ?? "—";
    return (
      <div key={t.token} className={`mini-card ${activeToken === t.token ? "active" : ""}`}>
        <div className="row-between">
          <div className="mono">tok {t.token}</div>
          <span className="badge">{fmtPct(t.pct, 1)}</span>
        </div>
        <div className="muted">actions: {dims}</div>
        <div className="muted">reward: {r} · next {effects?.horizon ?? 0}s: {nxt}</div>
      </div>
    );
  };

  return (
    <div>
      <div className="row-between">
        <div className="muted">Causality Strip</div>
        <span className="badge">token → action → outcome</span>
      </div>
      {!effects ? (
        <div className="muted">effects unavailable</div>
      ) : (
        <>
          {active ? (
            <div className="cause-focus">
              <div className="row-between">
                <h3 style={{ margin: 0 }}>Token {active.token}</h3>
                <span className="badge">count {active.count}</span>
              </div>
              <div className="kv" style={{ marginTop: 8 }}>
                <div className="k">dominant dims</div>
                <div className="v">{active.dominant_dims?.map((d) => `a${d}`).join(", ") ?? "—"}</div>
                <div className="k">reward mean</div>
                <div className="v">{active.reward_mean != null ? active.reward_mean.toFixed(3) : "—"}</div>
                <div className="k">next reward sum</div>
                <div className="v">{active.reward_next_sum_mean != null ? active.reward_next_sum_mean.toFixed(2) : "—"}</div>
              </div>
            </div>
          ) : (
            <div className="muted">select a token to see effect summary</div>
          )}
          <div className="mini-grid" style={{ marginTop: 10 }}>
            {top.map(pill)}
          </div>
        </>
      )}
    </div>
  );
}

function PrototypeGallery({ protos }: { protos: SkillPrototypesResponse | null }) {
  const items = protos?.prototypes ?? [];
  if (!items.length) return <div className="muted">no prototypes found (need `observations` in labeled demos)</div>;
  return (
    <div className="proto-grid">
      {items.slice(0, 12).map((p, i) => {
        const src = p.b64 && p.mime ? `data:${p.mime};base64,${p.b64}` : null;
        return (
          <div key={`${p.file}:${p.index}:${i}`} className="proto">
            {src ? <img src={src} alt={`tok ${protos?.token ?? ""} ex ${i}`} /> : <div className="muted">no image</div>}
            <div className="muted" style={{ marginTop: 6 }}>
              {p.file}#{p.index} {p.reward != null ? `· r=${p.reward.toFixed(3)}` : ""}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Skills() {
  const qc = useQueryClient();
  const summaryQ = useQuery({ queryKey: ["skills", "summary"], queryFn: fetchSkillsSummary, refetchInterval: 3000 });
  const summary = summaryQ.data;
  const tokens = (summary?.dataset?.token_top ?? []).slice(0, 128);
  const [filter, setFilter] = useState("");

  const filesQ = useQuery<SkillsFilesResponse>({ queryKey: ["skills", "files"], queryFn: () => fetchSkillsFiles(40), refetchInterval: 15000 });
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  useEffect(() => {
    if (!selectedFile && filesQ.data?.files?.length) setSelectedFile(filesQ.data.files[0].file);
  }, [selectedFile, filesQ.data]);

  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  useEffect(() => {
    if (selectedToken === null && tokens.length) setSelectedToken(tokens[0].token);
  }, [selectedToken, tokens]);

  const tokenQ = useQuery({
    queryKey: ["skills", "token", selectedToken],
    queryFn: async () => fetchSkillToken(selectedToken as number),
    enabled: selectedToken !== null,
    refetchInterval: 5000,
  });
  const detail = tokenQ.data;
  const atlasQ = useQuery<SkillsAtlasResponse>({ queryKey: ["skills", "atlas"], queryFn: fetchSkillsAtlas, refetchInterval: 30000 });
  const atlas = atlasQ.data ?? null;

  const [timelineWindow, setTimelineWindow] = useState<number>(2000);
  const [timelineStride, setTimelineStride] = useState<number>(2);
  const timelineQ = useQuery<SkillTimelineResponse>({
    queryKey: ["skills", "timeline", selectedFile, timelineWindow, timelineStride],
    queryFn: () => fetchSkillTimeline({ file: selectedFile, window: timelineWindow, stride: timelineStride, topk: 12 }),
    enabled: !!selectedFile,
    refetchInterval: 6000,
  });
  const timeline = timelineQ.data ?? null;
  const [cursor, setCursor] = useState(0);
  useEffect(() => {
    const n = timeline?.tokens?.length ?? 0;
    if (n) setCursor(n - 1);
  }, [timeline?.file, timeline?.tokens?.length]);

  const cursorToken = (() => {
    const arr = timeline?.tokens ?? [];
    if (!arr.length) return null;
    const i = clamp(cursor, 0, arr.length - 1);
    return arr[i];
  })();
  const activeToken = cursorToken ?? selectedToken;

  const effectsQ = useQuery<SkillEffectsResponse>({
    queryKey: ["skills", "effects", selectedFile, activeToken],
    queryFn: () => fetchSkillEffects({ file: selectedFile, tokens: activeToken != null ? [activeToken] : undefined, window: 4000, horizon: 45 }),
    enabled: !!selectedFile && activeToken != null,
    refetchInterval: 8000,
  });
  const effects = effectsQ.data ?? null;

  const protosQ = useQuery<SkillPrototypesResponse>({
    queryKey: ["skills", "prototypes", activeToken],
    queryFn: () => fetchSkillPrototypes(activeToken as number, { limit: 10, size: 112 }),
    enabled: activeToken != null,
    refetchInterval: 30000,
  });
  const protos = protosQ.data ?? null;

  const genNames = useMutation({
    mutationFn: async () => generateSkillNames(32, false),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["skills", "summary"] });
      if (selectedToken !== null) {
        await qc.invalidateQueries({ queryKey: ["skills", "token", selectedToken] });
      }
    },
  });

  const [nameDraft, setNameDraft] = useState("");
  const [subtitleDraft, setSubtitleDraft] = useState("");
  const [tagsDraft, setTagsDraft] = useState("");
  useEffect(() => {
    setNameDraft(detail?.name ?? "");
    setSubtitleDraft(detail?.subtitle ?? "");
    setTagsDraft((detail?.tags ?? []).join(", "));
  }, [detail?.name, detail?.subtitle, detail?.tags, selectedToken]);

  const saveName = useMutation({
    mutationFn: async () => {
      if (selectedToken == null) return null;
      const tags = tagsDraft
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      return updateSkillName({
        token: selectedToken,
        name: nameDraft.trim() || null,
        subtitle: subtitleDraft.trim() || null,
        tags: tags.length ? tags : null,
      });
    },
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["skills", "summary"] });
      if (selectedToken !== null) {
        await qc.invalidateQueries({ queryKey: ["skills", "token", selectedToken] });
      }
    },
  });

  const actionStats = useMemo(() => {
    const m = summary?.dataset?.action_mean;
    const s = summary?.dataset?.action_std;
    if (!m?.length || !s?.length) return null;
    const dims = Math.min(8, Math.min(m.length, s.length));
    return Array.from({ length: dims }, (_, i) => ({ i, mean: m[i], std: s[i] }));
  }, [summary]);

  const filteredTokens = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return tokens;
    return tokens.filter((t) => {
      const hay = `${t.token} ${t.name ?? ""} ${(t.tags ?? []).join(" ")} ${t.subtitle ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [tokens, filter]);

  if (summaryQ.isError) {
    return (
      <section className="card">
        <h2>Skill Spy</h2>
        <div className="muted">Failed to load `/api/skills/summary`.</div>
        <div className="muted">
          {(summaryQ.error as any)?.message ?? String(summaryQ.error)}
        </div>
        <div className="muted">
          Start the orchestrator (`python -m src.orchestrator.main --port 8040`) and ensure the skill artifacts exist.
        </div>
      </section>
    );
  }

  if (!summary) {
    return (
      <section className="card">
        <h2>Skill Spy</h2>
        <div className="muted">loading…</div>
      </section>
    );
  }

  const badge = <span className="badge">live</span>;
  const tokenName = detail?.name ?? tokens.find((t) => t.token === selectedToken)?.name ?? null;
  const tokenSubtitle = detail?.subtitle ?? tokens.find((t) => t.token === selectedToken)?.subtitle ?? null;
  const tokenTags = detail?.tags ?? tokens.find((t) => t.token === selectedToken)?.tags ?? null;

  return (
    <div className="grid page-grid skills-grid">
      <section className="card">
        <div className="row-between">
          <h2>Skill Spy</h2>
          <div className="row-inline">
            {badge}
            <button className="btn" onClick={() => genNames.mutate()} disabled={genNames.isPending || !tokens.length}>
              {genNames.isPending ? "Generating Names…" : "Generate Names"}
            </button>
          </div>
        </div>
        <div className="muted">
          Inspect learned skill tokens, their usage, and decoded action sequences (offline-friendly).
        </div>
        {genNames.isError && <div className="muted">Name generation failed: {(genNames.error as any)?.message ?? String(genNames.error)}</div>}
        <SummaryKpis summary={summary} />
        <div className="split">
          <div>
            <div className="muted">Artifacts</div>
            <div className="kv">
              <div className="k">skill ckpt</div>
              <div className="v">{summary.source.skill_ckpt}</div>
              <div className="k">labeled demos</div>
              <div className="v">{summary.source.labeled_npz_dir}</div>
            </div>
          </div>
          <div>
            <div className="muted">Action stats (dataset)</div>
            {!actionStats ? (
              <div className="muted">unavailable</div>
            ) : (
              <table className="table table-compact">
                <thead>
                  <tr>
                    <th>Dim</th>
                    <th>Mean</th>
                    <th>Std</th>
                  </tr>
                </thead>
                <tbody>
                  {actionStats.map((a) => (
                    <tr key={a.i}>
                      <td>a{a.i}</td>
                      <td>{a.mean.toFixed(3)}</td>
                      <td>{a.std.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
        {(summary.skill_vqvae as any)?.error && <div className="muted">skill_vqvae error: {(summary.skill_vqvae as any).error}</div>}
        {(summary.dataset as any)?.error && <div className="muted">dataset error: {(summary.dataset as any).error}</div>}
      </section>

      <section className="card">
        <div className="row-between">
          <h2>
            Token {selectedToken ?? "—"}
            {tokenName ? ` · ${tokenName}` : ""}
          </h2>
          <div className="row-inline">
            <span className="badge">decoded</span>
            <button
              className="btn btn-ghost"
              onClick={async () => {
                if (selectedToken === null) return;
                await copyToClipboard(String(selectedToken));
              }}
              disabled={selectedToken === null}
            >
              copy token
            </button>
          </div>
        </div>
        {tokenSubtitle && <div className="muted">{tokenSubtitle}</div>}
        {!!tokenTags?.length && (
          <div className="statline">
            {(tokenTags as any[]).slice(0, 10).map((t) => (
              <span key={String(t)} className="chip">
                {String(t)}
              </span>
            ))}
            {(tokenTags as any[]).length > 10 ? <span className="muted">+{(tokenTags as any[]).length - 10} more</span> : null}
          </div>
        )}
        <div className="panel" style={{ marginTop: 10 }}>
          <div className="muted">Naming workflow</div>
          <div className="split" style={{ marginTop: 6 }}>
            <div>
              <label className="muted">Name</label>
              <input className="input" value={nameDraft} onChange={(e) => setNameDraft(e.target.value)} placeholder="e.g. dodge-left microburst" />
            </div>
            <div>
              <label className="muted">Subtitle</label>
              <input className="input" value={subtitleDraft} onChange={(e) => setSubtitleDraft(e.target.value)} placeholder="context notes" />
            </div>
          </div>
          <div style={{ marginTop: 6 }}>
            <label className="muted">Tags</label>
            <input className="input" value={tagsDraft} onChange={(e) => setTagsDraft(e.target.value)} placeholder="comma-separated tags" />
          </div>
          <div className="row-between" style={{ marginTop: 8 }}>
            <span className="muted">Saved to skill_names.json</span>
            <button className="btn btn-ghost" onClick={() => saveName.mutate()} disabled={saveName.isPending || selectedToken == null}>
              {saveName.isPending ? "saving…" : "save name"}
            </button>
          </div>
        </div>

        {tokenQ.isError ? (
          <div className="muted">Failed to decode token {selectedToken ?? "—"}.</div>
        ) : !detail ? (
          <div className="muted">select a token to decode</div>
        ) : (
          <>
            <div className="kpis kpis-wrap">
              <div className="kpi">
                <div className="label">Dataset Count</div>
                <div className="value">{fmtNum(detail.stat?.count)}</div>
              </div>
              <div className="kpi">
                <div className="label">Dataset %</div>
                <div className="value">{fmtPct(detail.stat?.count_pct, 2)}</div>
              </div>
              <div className="kpi">
                <div className="label">Usage %</div>
                <div className="value">{fmtPct(detail.stat?.usage ?? null, 2)}</div>
              </div>
              <div className="kpi">
                <div className="label">Avg Reward</div>
                <div className="value">{detail.stat?.avg_reward?.toFixed?.(3) ?? "—"}</div>
              </div>
            </div>

            <TokenHelix detail={detail} />
            <details className="details">
              <summary className="muted">Decoded action chart</summary>
              <ActionSeqChart detail={detail} />
            </details>
            <details className="details">
              <summary className="muted">Raw decoded actions</summary>
              <pre className="code">{JSON.stringify(detail.decoded_action_seq.slice(0, 8), null, 2)}</pre>
            </details>
          </>
        )}
      </section>

      <section className="card">
        <div className="row-between">
          <h2>Atlas + Timeline</h2>
          <div className="row-inline">
            <span className="badge">{selectedFile ?? "no file"}</span>
            <select className="input" value={selectedFile ?? ""} onChange={(e) => setSelectedFile(e.target.value || null)}>
              {(filesQ.data?.files ?? []).slice(0, 40).map((f) => (
                <option key={f.file} value={f.file}>
                  {f.file} {f.steps != null ? `(${f.steps})` : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
        <AtlasStarfield atlas={atlas ?? null} selectedToken={selectedToken} onSelect={(tok) => setSelectedToken(tok)} />
        <div className="split" style={{ marginTop: 10 }}>
          <div>
            <SkillAtlas
              atlas={atlas ?? null}
              selectedToken={selectedToken}
              cursorToken={cursorToken}
              onSelect={(tok) => {
                setSelectedToken(tok);
              }}
            />
          </div>
          <div>
            <div className="row-inline" style={{ marginBottom: 8 }}>
              <span className="muted">window</span>
              <input
                className="input"
                type="number"
                value={timelineWindow}
                onChange={(e) => setTimelineWindow(clamp(numOr(e.target.value, 2000), 128, 20000))}
                style={{ width: 120 }}
              />
              <span className="muted">stride</span>
              <input
                className="input"
                type="number"
                value={timelineStride}
                onChange={(e) => setTimelineStride(clamp(numOr(e.target.value, 2), 1, 16))}
                style={{ width: 100 }}
              />
            </div>
            <TokenTimeline
              timeline={timeline ?? null}
              cursor={cursor}
              setCursor={setCursor}
              onSelectToken={(tok) => {
                setSelectedToken(tok);
              }}
            />
          </div>
        </div>
      </section>

      <section className="card">
        <h2>
          Effect + Prototypes {activeToken != null ? `· tok ${activeToken}` : ""}
        </h2>
        <div className="muted">Turns tokens into “something you can feel”: actions + next-horizon outcome, plus real frames when available.</div>
        <div className="split" style={{ marginTop: 10 }}>
          <div>
            <CausalityStrip effects={effects} activeToken={activeToken} />
          </div>
          <div>
            <div className="muted">Prototype Gallery</div>
            <PrototypeGallery protos={protos} />
          </div>
        </div>
      </section>

      <section className="card skills-table">
        <div className="row-between">
          <h2>Top Tokens</h2>
          <div className="row-inline">
            <input className="input" placeholder="filter tokens by id/name/tag…" value={filter} onChange={(e) => setFilter(e.target.value)} />
            <span className="badge">{filteredTokens.length}</span>
          </div>
        </div>
        <div className="muted" style={{ marginTop: 6 }}>
          click a row to inspect
        </div>
        <div className="table-viewport" style={{ marginTop: 10 }}>
          <table className="table table-hover">
            <thead>
              <tr>
                <th>Token</th>
                <th>Name</th>
                <th>Tags</th>
                <th>Count</th>
                <th>Count %</th>
                <th>Usage %</th>
                <th>Avg Reward</th>
              </tr>
            </thead>
            <tbody>
              {filteredTokens.map((t) => {
                const active = selectedToken === t.token;
                return (
                  <tr
                    key={t.token}
                    className={active ? "active" : ""}
                    onClick={() => setSelectedToken(t.token)}
                    style={{ cursor: "pointer" }}
                  >
                    <td>{t.token}</td>
                    <td>{t.name ?? "—"}</td>
                    <td className="muted">{t.tags?.slice?.(0, 2)?.join?.(", ") ?? "—"}</td>
                    <td>{fmtNum(t.count)}</td>
                    <td>{fmtPct(t.count_pct, 2)}</td>
                    <td>{fmtPct(t.usage ?? null, 2)}</td>
                    <td>{t.avg_reward.toFixed(3)}</td>
                  </tr>
                );
              })}
              {!filteredTokens.length && (
                <tr>
                  <td colSpan={7} className="muted">
                    no labeled skill tokens found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
