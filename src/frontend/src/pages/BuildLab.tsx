import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import cytoscape, { Core } from "cytoscape";
import { BuildLabExamplesResponse, Heartbeat, fetchBuildLabExamples, fetchWorkers } from "../api";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";

type Metric = "lift" | "deltaScore" | "pTop";
type ViewMode = "web" | "tree" | "forge3d";

type BuildItem = {
  key: string;
  label: string;
  kind: string;
  tier?: number | null;
  rarity?: string | null;
};

type PairStat = {
  a: string;
  b: string;
  support: number;
  supportRate: number;
  lift: number | null;
  deltaScore: number | null;
  pTop: number | null;
  confidence: number | null;
  logOdds: number | null;
};

type ItemsetStat = {
  key: string;
  items: string[];
  support: number;
  lift: number | null;
  deltaScore: number | null;
  pTop: number | null;
  parent: string | null;
  depth: number;
};

type TreeNode = ItemsetStat & {
  label: string;
  children: TreeNode[];
};

type CrossLink = {
  a: string;
  b: string;
  score: number;
};

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function scoreOf(hb: Heartbeat | null | undefined): number {
  const v = Number((hb as any)?.steam_score ?? (hb as any)?.hype_score ?? 0);
  if (!Number.isFinite(v)) return 0;
  return v;
}

function normKey(s: string) {
  return String(s ?? "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[^a-z0-9 _:-]/g, "");
}

function parseBuildItems(hb: Heartbeat): BuildItem[] {
  const raw = (hb as any)?.inventory_items;
  if (!Array.isArray(raw)) return [];
  const out: BuildItem[] = [];
  for (const it of raw) {
    if (!it || typeof it !== "object") continue;
    const label =
      String((it as any).label ?? (it as any).name ?? (it as any).item ?? (it as any).id ?? (it as any).key ?? "").trim() || "—";
    const kind = String((it as any).kind ?? (it as any).type ?? (it as any).category ?? "item").trim() || "item";
    const tierRaw = (it as any).tier ?? (it as any).level ?? (it as any).rank ?? null;
    const tier = tierRaw == null ? null : Number(tierRaw);
    const rarity = (it as any).rarity != null ? String((it as any).rarity) : null;
    const key = `${normKey(kind)}:${normKey(label)}`;
    if (!key || key.endsWith(":")) continue;
    out.push({ key, label, kind, tier: Number.isFinite(tier as any) ? (tier as number) : null, rarity });
  }
  return out;
}

function mean(xs: number[]) {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function buildPairStats(params: { workers: Heartbeat[]; minSupport: number; topFrac: number }): { items: Record<string, BuildItem>; pairs: PairStat[] } {
  const { workers, minSupport, topFrac } = params;
  const itemsByKey: Record<string, BuildItem> = {};
  const itemCounts: Record<string, number> = {};
  const pairCounts: Record<string, number> = {};
  const allScores: number[] = [];
  const scoresByPair: Record<string, number[]> = {};
  const workerItems: Array<{ keys: string[]; score: number }> = [];

  for (const w of workers) {
    const score = scoreOf(w);
    allScores.push(score);
    const items = parseBuildItems(w);
    for (const it of items) itemsByKey[it.key] = it;
    const keys = Array.from(new Set(items.map((it) => it.key))).sort();
    workerItems.push({ keys, score });
    for (const k of keys) itemCounts[k] = (itemCounts[k] ?? 0) + 1;
    for (let i = 0; i < keys.length; i++) {
      for (let j = i + 1; j < keys.length; j++) {
        const p = `${keys[i]}|${keys[j]}`;
        pairCounts[p] = (pairCounts[p] ?? 0) + 1;
        (scoresByPair[p] ??= []).push(score);
      }
    }
  }

  const n = workers.length || 1;
  const overallMean = mean(allScores);
  const sortedScores = [...allScores].sort((a, b) => b - a);
  const topN = Math.max(1, Math.round(sortedScores.length * clamp01(topFrac)));
  const cutoff = sortedScores[topN - 1] ?? Number.POSITIVE_INFINITY;

  const topSets = workerItems.map((wi) => ({ ...wi, top: wi.score >= cutoff }));

  const pairs: PairStat[] = [];
  for (const [key, support] of Object.entries(pairCounts)) {
    if (support < minSupport) continue;
    const [a, b] = key.split("|");
    if (!a || !b) continue;
    const pAB = support / n;
    const countA = itemCounts[a] ?? 0;
    const countB = itemCounts[b] ?? 0;
    const pA = countA / n;
    const pB = countB / n;
    const lift = pA > 0 && pB > 0 ? pAB / (pA * pB) : null;
    const avgPair = mean(scoresByPair[key] ?? []);
    const deltaScore = Number.isFinite(avgPair) ? avgPair - overallMean : null;
    let topHit = 0;
    for (const wi of topSets) {
      if (!wi.top) continue;
      if (wi.keys.includes(a) && wi.keys.includes(b)) topHit++;
    }
    const pTop = topN > 0 ? topHit / topN : null;
    const confA = countA > 0 ? support / countA : null;
    const confB = countB > 0 ? support / countB : null;
    const confidence = confA == null ? confB : confB == null ? confA : Math.max(confA, confB);
    const smooth = 0.5;
    const aOnly = Math.max(0, countA - support);
    const bOnly = Math.max(0, countB - support);
    const neither = Math.max(0, n - countA - countB + support);
    const logOdds = Math.log(((support + smooth) * (neither + smooth)) / ((aOnly + smooth) * (bOnly + smooth)));
    pairs.push({ a, b, support, supportRate: pAB, lift, deltaScore, pTop, confidence, logOdds });
  }

  return { items: itemsByKey, pairs };
}

function fmtPct(x: number | null) {
  if (x == null || !Number.isFinite(x)) return "—";
  return `${Math.round(x * 100)}%`;
}

function fmtNum(x: number | null, digits = 2) {
  if (x == null || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function metricValue(p: PairStat, m: Metric): number | null {
  if (m === "lift") return p.lift;
  if (m === "deltaScore") return p.deltaScore;
  return p.pTop;
}

function metricValueSet(p: ItemsetStat, m: Metric): number | null {
  if (m === "lift") return p.lift;
  if (m === "deltaScore") return p.deltaScore;
  return p.pTop;
}

function colorForKind(kind: string) {
  const k = normKey(kind);
  if (k.includes("weapon")) return "rgba(0,229,255,.9)";
  if (k.includes("tome")) return "rgba(255,214,10,.9)";
  if (k.includes("skill")) return "rgba(255,45,141,.9)";
  if (k.includes("interact")) return "rgba(52,211,153,.9)";
  return "rgba(255,255,255,.85)";
}

function colorForMetric(metric: Metric, v: number | null): string {
  if (v == null || !Number.isFinite(v)) return "rgba(148,163,184,.7)";
  if (metric === "lift") {
    const t = clamp01((v - 0.9) / 2.4);
    const hue = 10 + t * 110;
    return `hsl(${hue}, 85%, 58%)`;
  }
  if (metric === "deltaScore") {
    const t = clamp01((v + 2.5) / 5.0);
    const hue = 0 + t * 120;
    return `hsl(${hue}, 90%, 58%)`;
  }
  const hue = 210 - clamp01(v) * 140;
  return `hsl(${hue}, 90%, 60%)`;
}

function buildItemsetTree(params: {
  workers: Heartbeat[];
  minSupport: number;
  topFrac: number;
  maxDepth: number;
  maxChildren: number;
  metric: Metric;
}): { root: TreeNode; nodes: TreeNode[]; byKey: Record<string, TreeNode> } {
  const { workers, minSupport, topFrac, maxDepth, maxChildren, metric } = params;
  const itemsByKey: Record<string, BuildItem> = {};
  const itemCounts: Record<string, number> = {};
  const setCounts: Record<string, number> = {};
  const scoresBySet: Record<string, number[]> = {};
  const allScores: number[] = [];
  const workerSets: Array<{ score: number; sets: string[] }> = [];

  const addSet = (key: string, score: number) => {
    setCounts[key] = (setCounts[key] ?? 0) + 1;
    (scoresBySet[key] ??= []).push(score);
  };

  const genCombos = (keys: string[], maxK: number, out: string[]) => {
    const n = keys.length;
    const combo: string[] = [];
    const dfs = (start: number, k: number) => {
      if (combo.length === k) {
        out.push(combo.join("|"));
        return;
      }
      for (let i = start; i < n; i++) {
        combo.push(keys[i]);
        dfs(i + 1, k);
        combo.pop();
      }
    };
    for (let k = 1; k <= maxK; k++) dfs(0, k);
  };

  for (const w of workers) {
    const score = scoreOf(w);
    allScores.push(score);
    const items = parseBuildItems(w);
    for (const it of items) itemsByKey[it.key] = it;
    const keys = Array.from(new Set(items.map((it) => it.key))).sort();
    if (!keys.length) continue;
    for (const k of keys) itemCounts[k] = (itemCounts[k] ?? 0) + 1;
    const sets: string[] = [];
    genCombos(keys, Math.min(maxDepth, keys.length), sets);
    workerSets.push({ score, sets });
    for (const s of sets) addSet(s, score);
  }

  const n = workers.length || 1;
  const overallMean = mean(allScores);
  const sortedScores = [...allScores].sort((a, b) => b - a);
  const topN = Math.max(1, Math.round(sortedScores.length * clamp01(topFrac)));
  const cutoff = sortedScores[topN - 1] ?? Number.POSITIVE_INFINITY;
  const topCounts: Record<string, number> = {};
  for (const w of workerSets) {
    if (w.score < cutoff) continue;
    for (const s of w.sets) topCounts[s] = (topCounts[s] ?? 0) + 1;
  }

  const stats: ItemsetStat[] = [];
  const statsByKey: Record<string, ItemsetStat> = {};
  for (const [key, support] of Object.entries(setCounts)) {
    if (support < minSupport) continue;
    const items = key.split("|").filter(Boolean);
    if (!items.length || items.length > maxDepth) continue;
    const pS = support / n;
    let denom = 1;
    for (const it of items) denom *= (itemCounts[it] ?? 0) / n || 0;
    const lift = denom > 0 ? pS / denom : null;
    const avgSet = mean(scoresBySet[key] ?? []);
    const deltaScore = Number.isFinite(avgSet) ? avgSet - overallMean : null;
    const pTop = topN > 0 ? (topCounts[key] ?? 0) / topN : null;
    const stat: ItemsetStat = { key, items, support, lift, deltaScore, pTop, parent: null, depth: items.length };
    stats.push(stat);
    statsByKey[key] = stat;
  }

  for (const stat of stats) {
    if (stat.items.length <= 1) continue;
    let best: ItemsetStat | null = null;
    for (let i = 0; i < stat.items.length; i++) {
      const subset = [...stat.items.slice(0, i), ...stat.items.slice(i + 1)];
      const key = subset.join("|");
      const cand = statsByKey[key];
      if (!cand) continue;
      if (!best || cand.support > best.support) best = cand;
    }
    stat.parent = best ? best.key : null;
  }

  const childrenByParent: Record<string, ItemsetStat[]> = {};
  for (const stat of stats) {
    const p = stat.parent;
    if (!p) continue;
    (childrenByParent[p] ??= []).push(stat);
  }
  for (const [parent, kids] of Object.entries(childrenByParent)) {
    kids.sort((a, b) => {
      const av = metricValueSet(a, metric) ?? -Infinity;
      const bv = metricValueSet(b, metric) ?? -Infinity;
      if (bv !== av) return bv - av;
      return a.key.localeCompare(b.key);
    });
    childrenByParent[parent] = kids.slice(0, Math.max(1, maxChildren));
  }

  const labelFor = (stat: ItemsetStat) => {
    if (!stat.parent) {
      if (stat.items.length === 1) return itemsByKey[stat.items[0]]?.label ?? stat.items[0];
      return stat.items.map((it) => itemsByKey[it]?.label ?? it).join(" + ");
    }
    const parent = statsByKey[stat.parent];
    if (!parent) return stat.items.map((it) => itemsByKey[it]?.label ?? it).join(" + ");
    const parentSet = new Set(parent.items);
    const added = stat.items.find((it) => !parentSet.has(it));
    return added ? itemsByKey[added]?.label ?? added : stat.items.join(" + ");
  };

  const nodesByKey: Record<string, TreeNode> = {};
  for (const stat of stats) {
    nodesByKey[stat.key] = { ...stat, label: labelFor(stat), children: [] };
  }

  const root: TreeNode = {
    key: "root",
    items: [],
    support: workers.length,
    lift: null,
    deltaScore: null,
    pTop: null,
    parent: null,
    depth: 0,
    label: "START",
    children: [],
  };

  for (const stat of stats) {
    if (stat.depth !== 1) continue;
    root.children.push(nodesByKey[stat.key]);
  }
  root.children.sort((a, b) => (b.support - a.support) || a.label.localeCompare(b.label));
  root.children = root.children.slice(0, Math.max(1, maxChildren * 3));

  for (const [parentKey, kids] of Object.entries(childrenByParent)) {
    const parent = nodesByKey[parentKey];
    if (!parent) continue;
    parent.children = kids.map((k) => nodesByKey[k.key]).filter(Boolean);
  }

  const allNodes = [root, ...Object.values(nodesByKey)];
  return { root, nodes: allNodes, byKey: nodesByKey };
}

function layoutRadialTree(root: TreeNode, w: number, h: number, ringGap: number) {
  const cx = w / 2;
  const cy = h / 2;
  const leaves = new Map<string, number>();
  const collectLeaves = (n: TreeNode): number => {
    if (!n.children.length) {
      leaves.set(n.key, 1);
      return 1;
    }
    let sum = 0;
    for (const c of n.children) sum += collectLeaves(c);
    leaves.set(n.key, Math.max(1, sum));
    return Math.max(1, sum);
  };
  collectLeaves(root);
  const positions: Record<string, { x: number; y: number; angle: number; r: number }> = {};
  const assign = (node: TreeNode, start: number, end: number, depth: number) => {
    const angle = (start + end) / 2;
    const r = depth * ringGap;
    positions[node.key] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), angle, r };
    let cur = start;
    const total = leaves.get(node.key) ?? 1;
    for (const child of node.children) {
      const span = ((leaves.get(child.key) ?? 1) / total) * (end - start);
      assign(child, cur, cur + span, depth + 1);
      cur += span;
    }
  };
  assign(root, -Math.PI / 2, (Math.PI * 3) / 2, 0);
  return { positions, center: { x: cx, y: cy } };
}

export default function BuildLab() {
  const workersQ = useQuery({ queryKey: ["workers"], queryFn: fetchWorkers, refetchInterval: 1500 });
  const [metric, setMetric] = useState<Metric>("lift");
  const [viewMode, setViewMode] = useState<ViewMode>("web");
  const [minSupport, setMinSupport] = useState(2);
  const [topFrac, setTopFrac] = useState(0.2);
  const [maxDepth, setMaxDepth] = useState(4);
  const [maxChildren, setMaxChildren] = useState(4);
  const [policy, setPolicy] = useState<string>("all");
  const [onlyTopRuns, setOnlyTopRuns] = useState(false);
  const [lockLayout, setLockLayout] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedTree, setSelectedTree] = useState<string | null>(null);
  const [spotlightItems, setSpotlightItems] = useState<string[] | null>(null);
  const [showCrossLinks, setShowCrossLinks] = useState(true);
  const [whatIfInput, setWhatIfInput] = useState("");
  const pinnedRef = useRef<Set<string>>(new Set());
  const treeWrapRef = useRef<HTMLDivElement | null>(null);
  const [treeSize, setTreeSize] = useState({ w: 900, h: 680 });

  const allWorkers = useMemo(() => Object.values(workersQ.data ?? {}), [workersQ.data]);
  const policies = useMemo(() => {
    const s = new Set<string>();
    for (const w of allWorkers) if (w?.policy_name) s.add(String(w.policy_name));
    return Array.from(s).sort();
  }, [allWorkers]);

  const filteredWorkers = useMemo(() => {
    let ws = allWorkers.filter(Boolean) as Heartbeat[];
    if (policy !== "all") ws = ws.filter((w) => String(w.policy_name ?? "") === policy);
    if (!ws.length) return ws;
    if (!onlyTopRuns) return ws;
    const scores = ws.map((w) => scoreOf(w)).sort((a, b) => b - a);
    const topN = Math.max(1, Math.round(scores.length * clamp01(topFrac)));
    const cutoff = scores[topN - 1] ?? Number.POSITIVE_INFINITY;
    return ws.filter((w) => scoreOf(w) >= cutoff);
  }, [allWorkers, policy, onlyTopRuns, topFrac]);

  const { items, pairs } = useMemo(() => buildPairStats({ workers: filteredWorkers, minSupport, topFrac }), [filteredWorkers, minSupport, topFrac]);
  const itemsByLabel = useMemo(() => {
    const map = new Map<string, string>();
    Object.values(items).forEach((it) => {
      map.set(normKey(it.label), it.key);
    });
    return map;
  }, [items]);

  const whatIfKeys = useMemo(() => {
    const raw = whatIfInput
      .split(",")
      .map((t) => normKey(t))
      .filter(Boolean);
    const out = new Set<string>();
    raw.forEach((label) => {
      const key = itemsByLabel.get(label);
      if (key) out.add(key);
    });
    return Array.from(out.values());
  }, [whatIfInput, itemsByLabel]);

  const whatIfRecs = useMemo(() => {
    if (!whatIfKeys.length) return [];
    const scores = new Map<string, number>();
    const metrics: Record<string, { lift: number | null; deltaScore: number | null; pTop: number | null }> = {};
    for (const p of pairs) {
      const aIn = whatIfKeys.includes(p.a);
      const bIn = whatIfKeys.includes(p.b);
      if (aIn === bIn) continue;
      const cand = aIn ? p.b : p.a;
      const v = metricValue(p, metric) ?? 0;
      const prev = scores.get(cand) ?? -Infinity;
      if (v > prev) {
        scores.set(cand, v);
        metrics[cand] = { lift: p.lift, deltaScore: p.deltaScore, pTop: p.pTop };
      }
    }
    const out = Array.from(scores.entries())
      .filter(([k]) => !whatIfKeys.includes(k))
      .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
      .slice(0, 8)
      .map(([k, v]) => ({ key: k, score: v, metrics: metrics[k] }));
    return out;
  }, [pairs, metric, whatIfKeys]);

  const antiSynergy = useMemo(() => {
    return pairs
      .filter((p) => (p.deltaScore ?? 0) < -0.5)
      .sort((a, b) => (a.deltaScore ?? 0) - (b.deltaScore ?? 0))
      .slice(0, 8);
  }, [pairs]);

  const rankedPairs = useMemo(() => {
    const xs = pairs
      .map((p) => ({ p, v: metricValue(p, metric) }))
      .filter((x) => x.v != null && Number.isFinite(x.v as any));
    xs.sort((a, b) => {
      if (metric === "deltaScore") return (b.v as number) - (a.v as number);
      if (metric === "lift") return (b.v as number) - (a.v as number);
      return (b.v as number) - (a.v as number);
    });
    return xs.slice(0, 90).map((x) => x.p);
  }, [pairs, metric]);

  const itemCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const w of filteredWorkers) {
      for (const it of parseBuildItems(w)) counts[it.key] = (counts[it.key] ?? 0) + 1;
    }
    return counts;
  }, [filteredWorkers]);

  const forceGraphData = useMemo(() => {
    const nodes = Object.values(items).map((it) => ({
      id: it.key,
      name: it.label,
      kind: it.kind,
      size: Math.max(1, Math.min(14, itemCounts[it.key] ?? 1)),
      color: colorForKind(it.kind),
    }));
    const links = rankedPairs.slice(0, 160).map((p) => ({
      source: p.a,
      target: p.b,
      weight: metricValue(p, metric) ?? 0.1,
    }));
    return { nodes, links };
  }, [items, rankedPairs, itemCounts, metric]);

  const selectedPartners = useMemo(() => {
    if (!selected) return [];
    const rel = pairs.filter((p) => p.a === selected || p.b === selected);
    const mapped = rel
      .map((p) => {
        const other = p.a === selected ? p.b : p.a;
        return { other, stat: p, v: metricValue(p, metric) };
      })
      .filter((x) => x.v != null && Number.isFinite(x.v as any));
    mapped.sort((a, b) => (b.v as number) - (a.v as number));
    return mapped.slice(0, 12);
  }, [pairs, selected, metric]);

  const cyRef = useRef<Core | null>(null);
  const graphElRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!graphElRef.current) return;
    if (cyRef.current) return;
    const cy = cytoscape({
      container: graphElRef.current,
      elements: [],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            color: "rgba(255,255,255,.92)",
            "text-outline-color": "rgba(0,0,0,.7)",
            "text-outline-width": 2,
            "font-size": 10,
            "text-max-width": 90,
            "text-wrap": "ellipsis",
            "background-color": "data(color)",
            width: "mapData(size, 1, 12, 14, 34)",
            height: "mapData(size, 1, 12, 14, 34)",
            "border-width": 1,
            "border-color": "rgba(255,255,255,.18)",
          },
        },
        {
          selector: "edge",
          style: {
            width: "mapData(weight, 0, 3, 1, 4)",
            "line-color": "rgba(255,255,255,.25)",
            "curve-style": "bezier",
            opacity: 0.75,
          },
        },
        {
          selector: ".pinned",
          style: {
            "border-width": 2,
            "border-color": "rgba(255,214,10,.9)",
          },
        },
        {
          selector: ".faded",
          style: {
            opacity: 0.12,
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 2,
            "border-color": "rgba(0,229,255,.9)",
          },
        },
      ],
      layout: { name: "cose", animate: false },
      userZoomingEnabled: true,
      wheelSensitivity: 0.18,
    });

    cy.on("tap", "node", (evt) => {
      const id = String(evt.target.id());
      setSelected(id || null);
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) setSelected(null);
    });

    cyRef.current = cy;
    return () => {
      try {
        cy.destroy();
      } catch {}
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.autoungrabify(lockLayout);
  }, [lockLayout]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const els: any[] = [];
    const prevPos = new Map<string, { x: number; y: number }>();
    cy.nodes().forEach((n) => prevPos.set(n.id(), n.position()));
    const prevKeys = Array.from(prevPos.keys());
    let centroid = { x: 0, y: 0 };
    if (prevKeys.length) {
      const sum = prevKeys.reduce(
        (acc, id) => {
          const p = prevPos.get(id)!;
          acc.x += p.x;
          acc.y += p.y;
          return acc;
        },
        { x: 0, y: 0 },
      );
      centroid = { x: sum.x / prevKeys.length, y: sum.y / prevKeys.length };
    }

    const sizeByKey: Record<string, number> = {};
    for (const w of filteredWorkers) {
      for (const it of parseBuildItems(w)) sizeByKey[it.key] = (sizeByKey[it.key] ?? 0) + 1;
    }
    const keys = Object.keys(items);
    for (const k of keys) {
      const it = items[k];
      const size = Math.max(1, Math.min(12, sizeByKey[k] ?? 1));
      els.push({
        data: {
          id: it.key,
          label: it.label,
          color: colorForKind(it.kind),
          kind: it.kind,
          tier: it.tier ?? null,
          rarity: it.rarity ?? null,
          size,
        },
      });
    }

    const edges = rankedPairs
      .map((p) => {
        const w = metricValue(p, metric);
        if (w == null || !Number.isFinite(w)) return null;
        const weight = metric === "deltaScore" ? Math.max(0, w) : Math.max(0, w);
        return { data: { id: `${p.a}->${p.b}`, source: p.a, target: p.b, weight } };
      })
      .filter(Boolean) as any[];

    els.push(...edges);

    cy.batch(() => {
      cy.elements().remove();
      cy.add(els);
      for (const id of pinnedRef.current) {
        const n = cy.getElementById(id);
        if (n && n.length) {
          n.addClass("pinned");
          n.lock();
        }
      }
      if (lockLayout) {
        cy.nodes().forEach((n) => {
          const pos = prevPos.get(n.id());
          if (pos) {
            n.position(pos);
          } else {
            n.position({
              x: centroid.x + (Math.random() - 0.5) * 120,
              y: centroid.y + (Math.random() - 0.5) * 120,
            });
          }
        });
      }
    });

    if (!lockLayout) {
      const layout = cy.layout({ name: "cose", animate: false, fit: true, padding: 24 });
      layout.run();
    }
  }, [items, rankedPairs, metric, filteredWorkers, lockLayout]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.elements().removeClass("faded");
      if (spotlightItems && spotlightItems.length) {
        const keepNodes = new Set(spotlightItems);
        const keep = cy.collection(
          cy.nodes()
            .filter((n) => keepNodes.has(n.id()))
            .map((n) => n),
        );
        const keepEdges = cy.edges().filter((e) => keepNodes.has(e.data("source")) && keepNodes.has(e.data("target")));
        cy.elements().not(keep.union(keepEdges)).addClass("faded");
      }
      if (!selected) return;
      const sel = cy.getElementById(selected);
      if (!sel || !sel.length) return;
      const neigh = sel.closedNeighborhood();
      cy.elements().not(neigh).addClass("faded");
      sel.select();
    });
  }, [selected, spotlightItems]);

  const selectedItem = selected ? items[selected] ?? null : null;
  const selectedPinned = selected ? pinnedRef.current.has(selected) : false;

  useEffect(() => {
    const el = treeWrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect();
      const w = Math.max(420, Math.round(rect.width));
      const h = Math.max(420, Math.round(rect.height));
      setTreeSize({ w, h });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (viewMode !== "tree") setSelectedTree(null);
  }, [viewMode]);

  const treeData = useMemo(() => {
    return buildItemsetTree({
      workers: filteredWorkers,
      minSupport,
      topFrac,
      maxDepth,
      maxChildren,
      metric,
    });
  }, [filteredWorkers, minSupport, topFrac, maxDepth, maxChildren, metric]);

  const treeSelectedNode = useMemo(() => {
    if (!selectedTree) return null;
    return treeData.byKey[selectedTree] ?? null;
  }, [selectedTree, treeData.byKey]);

  const selectedComboItems = useMemo(() => {
    if (viewMode === "tree" && treeSelectedNode?.items?.length) return treeSelectedNode.items;
    if (selected) return [selected];
    return [];
  }, [viewMode, treeSelectedNode, selected]);

  const examplesQ = useQuery<BuildLabExamplesResponse>({
    queryKey: ["buildlabExamples", selectedComboItems.join("|")],
    queryFn: () => fetchBuildLabExamples(selectedComboItems, 6, false),
    enabled: selectedComboItems.length > 0,
    refetchInterval: 8000,
  });
  const predictedScore = useMemo(() => {
    const ex = examplesQ.data?.examples ?? [];
    if (!ex.length) return null;
    const scores = ex.map((e) => Number(e.final_score ?? 0)).filter((v) => Number.isFinite(v));
    if (!scores.length) return null;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  }, [examplesQ.data]);

  const treeLayout = useMemo(() => {
    return layoutRadialTree(treeData.root, treeSize.w, treeSize.h, Math.min(treeSize.w, treeSize.h) / (maxDepth + 1.8));
  }, [treeData.root, treeSize, maxDepth]);

  const treeCrossLinks = useMemo(() => {
    if (!showCrossLinks) return [];
    const nodes = treeData.nodes.filter((n) => n.depth >= 2);
    const seen = new Set<string>();
    const links: CrossLink[] = [];
    const maxPerNode = 3;
    const maxTotal = 220;
    for (const node of nodes) {
      let count = 0;
      for (let i = 0; i < node.items.length; i++) {
        const base = [...node.items.slice(0, i), ...node.items.slice(i + 1)];
        for (const cand of nodes) {
          if (cand.key === node.key || cand.depth !== node.depth) continue;
          const set = new Set(cand.items);
          let shared = 0;
          for (const it of base) if (set.has(it)) shared++;
          if (shared !== base.length) continue;
          const pairKey = [node.key, cand.key].sort().join("::");
          if (seen.has(pairKey)) continue;
          seen.add(pairKey);
          const scoreA = metricValueSet(node, metric) ?? -Infinity;
          const scoreB = metricValueSet(cand, metric) ?? -Infinity;
          const score = Math.max(scoreA, scoreB);
          links.push({ a: node.key, b: cand.key, score });
          count++;
          if (count >= maxPerNode) break;
        }
        if (count >= maxPerNode) break;
      }
      if (links.length >= maxTotal) break;
    }
    links.sort((a, b) => b.score - a.score);
    return links.slice(0, maxTotal);
  }, [treeData.nodes, metric, showCrossLinks]);

  const selectedWorkers = useMemo(() => {
    if (!selected) return [];
    const xs: Array<{ iid: string; score: number; policy: string; ageS: number }> = [];
    const now = Date.now();
    for (const w of filteredWorkers) {
      const keys = new Set(parseBuildItems(w).map((it) => it.key));
      if (!keys.has(selected)) continue;
      const rawTs = Number((w as any)?.ts ?? 0);
      const tsMs = rawTs > 10_000_000_000 ? rawTs : rawTs > 0 ? rawTs * 1000 : now;
      const ageS = Math.max(0, (now - tsMs) / 1000);
      xs.push({ iid: String(w.instance_id), score: scoreOf(w), policy: String(w.policy_name ?? "—"), ageS });
    }
    xs.sort((a, b) => b.score - a.score);
    return xs.slice(0, 8);
  }, [selected, filteredWorkers]);

  const topPairsForTable = useMemo(() => {
    const xs = pairs
      .map((p) => ({ p, v: metricValue(p, metric) }))
      .filter((x) => x.v != null && Number.isFinite(x.v as any));
    xs.sort((a, b) => (b.v as number) - (a.v as number));
    const list = xs.map((x) => x.p);
    if (spotlightItems && spotlightItems.length) {
      const set = new Set(spotlightItems);
      return list.filter((p) => set.has(p.a) && set.has(p.b)).slice(0, 20);
    }
    return list.slice(0, 20);
  }, [pairs, metric, spotlightItems]);

  const onPinSelected = () => {
    const cy = cyRef.current;
    if (!cy || !selected) return;
    const next = new Set(pinnedRef.current);
    const n = cy.getElementById(selected);
    if (next.has(selected)) {
      next.delete(selected);
      n.removeClass("pinned");
      n.unlock();
    } else {
      next.add(selected);
      n.addClass("pinned");
      n.lock();
    }
    pinnedRef.current = next;
  };

  const onSpotlightBranch = (node: TreeNode) => {
    if (!node.items.length) return;
    setSpotlightItems([...node.items]);
    setViewMode("web");
    setSelected(node.items[0] ?? null);
  };

  const onClearSpotlight = () => {
    setSpotlightItems(null);
  };

  const onHighlightPair = (p: PairStat) => {
    const cy = cyRef.current;
    if (!cy) return;
    setSelected(p.a);
    cy.batch(() => {
      cy.elements().removeClass("faded");
      const a = cy.getElementById(p.a);
      const b = cy.getElementById(p.b);
      const edge = cy.getElementById(`${p.a}->${p.b}`);
      const keep = a.union(b).union(edge).closedNeighborhood();
      cy.elements().not(keep).addClass("faded");
      a.select();
      b.select();
    });
    cy.fit(cy.getElementById(p.a).union(cy.getElementById(p.b)), 60);
  };

  const hasAnyItems = Object.keys(items).length > 0;
  const workersTotal = allWorkers.length;
  const workersWithInventory = useMemo(() => allWorkers.filter((w) => parseBuildItems(w).length > 0).length, [allWorkers]);
  const workersMissingInventory = useMemo(() => {
    return allWorkers.filter((w) => parseBuildItems(w).length === 0).map((w) => String((w as any)?.instance_id ?? "")).filter(Boolean);
  }, [allWorkers]);
  const lastUpdateAgeS = useMemo(() => {
    const now = Date.now();
    let lastMs = 0;
    for (const w of allWorkers) {
      const raw = Number((w as any)?.ts ?? 0);
      const tsMs = raw > 10_000_000_000 ? raw : raw > 0 ? raw * 1000 : 0;
      if (tsMs > lastMs) lastMs = tsMs;
    }
    if (!lastMs) return null;
    return Math.max(0, (now - lastMs) / 1000);
  }, [allWorkers]);

  const noWorkers = !workersQ.isLoading && workersTotal === 0;
  const noInventory = workersTotal > 0 && workersWithInventory === 0;
  const filtersExclude = workersTotal > 0 && filteredWorkers.length === 0;
  const noCombos = !hasAnyItems && filteredWorkers.length > 0;

  return (
    <div className="page buildlab">
      <section className="forge-hero">
        <div>
          <div className="forge-title">Synergy Forge</div>
          <div className="muted">Drag items into the crucible to reveal compatible neighbors.</div>
          <div className="forge-kpis">
            <div>
              <span className="label">Active workers</span>
              <strong>{workersTotal}</strong>
            </div>
            <div>
              <span className="label">Known items</span>
              <strong>{Object.keys(items).length}</strong>
            </div>
            <div>
              <span className="label">Live edges</span>
              <strong>{rankedPairs.length}</strong>
            </div>
          </div>
        </div>
        <div className="forge-crucible">
          <div className="forge-crucible-title">Crucible</div>
          {selectedComboItems.length ? (
            <div className="forge-crucible-items">
              {selectedComboItems.map((it) => (
                <span key={it} className="chip chip-glow">
                  {it}
                </span>
              ))}
            </div>
          ) : (
            <div className="muted">Drop an item from the graph to begin forging.</div>
          )}
          <div className="forge-gauge">
            <span>Predicted reward</span>
            <strong>{predictedScore == null ? "—" : predictedScore.toFixed(2)}</strong>
          </div>
        </div>
      </section>
      <div className="row-between" style={{ alignItems: "baseline" }}>
        <h1 style={{ margin: 0 }}>Build Lab</h1>
        <div className="muted">
          {viewMode === "web"
            ? "Synergy Web • combos • build intuition"
            : viewMode === "forge3d"
              ? "Synergy Forge • 3D molecular graph"
              : "Refinement Tree • build stages • next picks"}
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="row-between" style={{ gap: 10, flexWrap: "wrap" }}>
          <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
            <label className="row" style={{ gap: 6 }}>
              <span className="muted">View</span>
              <select value={viewMode} onChange={(e) => setViewMode(e.target.value as ViewMode)}>
                <option value="web">Synergy web</option>
                <option value="forge3d">Forge 3D</option>
                <option value="tree">Refinement tree</option>
              </select>
            </label>
            <label className="row" style={{ gap: 6 }}>
              <span className="muted">Metric</span>
              <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
                <option value="lift">Lift (PMI-ish)</option>
                <option value="deltaScore">ΔScore</option>
                <option value="pTop">P(top)</option>
              </select>
            </label>
            <label className="row" style={{ gap: 6 }}>
              <span className="muted">Policy</span>
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                <option value="all">All</option>
                {policies.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label className="row" style={{ gap: 6 }}>
              <span className="muted">Min support</span>
              <input
                type="number"
                min={1}
                max={20}
                value={minSupport}
                onChange={(e) => setMinSupport(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                style={{ width: 84 }}
              />
            </label>
            <label className="row" style={{ gap: 6 }}>
              <span className="muted">Top%</span>
              <input
                type="number"
                min={5}
                max={100}
                value={Math.round(topFrac * 100)}
                onChange={(e) => setTopFrac(clamp01((Number(e.target.value) || 20) / 100))}
                style={{ width: 72 }}
              />
            </label>
            <label className="row" style={{ gap: 6 }}>
              <input type="checkbox" checked={onlyTopRuns} onChange={(e) => setOnlyTopRuns(e.target.checked)} />
              <span className="muted">only top runs</span>
            </label>
            {viewMode === "tree" ? (
              <>
                <label className="row" style={{ gap: 6 }}>
                  <span className="muted">Depth</span>
                  <input
                    type="number"
                    min={2}
                    max={6}
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(Math.max(2, Math.min(6, Number(e.target.value) || 2)))}
                    style={{ width: 64 }}
                  />
                </label>
                <label className="row" style={{ gap: 6 }}>
                  <span className="muted">Top kids</span>
                  <input
                    type="number"
                    min={2}
                    max={8}
                    value={maxChildren}
                    onChange={(e) => setMaxChildren(Math.max(2, Math.min(8, Number(e.target.value) || 2)))}
                    style={{ width: 64 }}
                  />
                </label>
                <label className="row" style={{ gap: 6 }}>
                  <input type="checkbox" checked={showCrossLinks} onChange={(e) => setShowCrossLinks(e.target.checked)} />
                  <span className="muted">cross-links</span>
                </label>
              </>
            ) : null}
          </div>
          <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
            <label className="row" style={{ gap: 6 }}>
              <input type="checkbox" checked={lockLayout} onChange={(e) => setLockLayout(e.target.checked)} />
              <span className="muted">lock layout</span>
            </label>
            <span className="muted">
              {filteredWorkers.length} workers • {Object.keys(items).length} nodes • {rankedPairs.length} edges
            </span>
            {spotlightItems ? (
              <button className="btn btn-ghost" onClick={onClearSpotlight}>
                Clear spotlight
              </button>
            ) : null}
          </div>
        </div>
        <div className="statline" style={{ marginTop: 8 }}>
          <span className="pill">
            inventory {workersWithInventory}/{workersTotal || 0}
          </span>
          <span className="pill">transactions {filteredWorkers.length}</span>
          <span className="pill">unique items {Object.keys(items).length}</span>
          <span className="pill">edges {rankedPairs.length}</span>
          <span className="pill">last update {lastUpdateAgeS == null ? "—" : `${lastUpdateAgeS.toFixed(1)}s`}</span>
        </div>
        <div className="panel" style={{ marginTop: 10 }}>
          <div className="row-between">
            <div className="muted">Data readiness</div>
            <span className="badge">{workersWithInventory}/{workersTotal || 0} reporting</span>
          </div>
          <div className="statline">
            <span className="pill">
              coverage {workersTotal ? Math.round((workersWithInventory / workersTotal) * 100) : 0}%
            </span>
            <span className={`pill ${workersWithInventory === 0 ? "pill-missing" : "pill-ok"}`}>
              {workersWithInventory === 0 ? "inventory feed missing" : "inventory feed ok"}
            </span>
          </div>
          {workersMissingInventory.length ? (
            <div className="muted" style={{ marginTop: 6 }}>
              missing: {workersMissingInventory.slice(0, 8).join(", ")}
              {workersMissingInventory.length > 8 ? ` +${workersMissingInventory.length - 8} more` : ""}
            </div>
          ) : (
            <div className="muted" style={{ marginTop: 6 }}>All live workers reporting inventory.</div>
          )}
        </div>
      </div>

      {!workersQ.isLoading && (noWorkers || filtersExclude) ? (
        <div className="card" style={{ marginTop: 12, padding: 14 }}>
          <div className="stream-highlight-placeholder" style={{ minHeight: 80 }}>
            {noWorkers ? "No workers connected yet — start a worker to populate builds." : "No workers match filters yet — relax filters or switch policy."}
            <div className="row" style={{ gap: 10, marginTop: 10, flexWrap: "wrap" }}>
              <Link className="btn" to="/instances">
                Go to Instances
              </Link>
              <Link className="btn btn-ghost" to="/spy">
                Open Spy
              </Link>
            </div>
          </div>
        </div>
      ) : null}

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(360px, 0.8fr)", marginTop: 12 }}>
        <div className="card" style={{ minHeight: 520, position: "relative" }}>
          <div className="row-between" style={{ marginBottom: 8 }}>
            <div className="kpi">
              <div className="label">
                {viewMode === "web" ? "Synergy Web" : viewMode === "forge3d" ? "Synergy Forge" : "Refinement Tree"}
              </div>
              <div className="value">{hasAnyItems ? "LIVE" : "EMPTY"}</div>
            </div>
            <div className="muted" style={{ textAlign: "right" }}>
              {viewMode === "web"
                ? "click node → partners • pin nodes for screenshots"
                : viewMode === "forge3d"
                  ? "drag node → realign • click → set crucible"
                  : "click node → next picks • rings = build stages"}
            </div>
          </div>
          {viewMode === "web" ? (
            <div className="buildlab-graph" ref={graphElRef} />
          ) : viewMode === "forge3d" ? (
            <div className="forge3d-wrap">
              <ForceGraph3D
                graphData={forceGraphData}
                backgroundColor="#050709"
                nodeRelSize={4}
                linkOpacity={0.35}
                linkWidth={(link: any) => Math.max(0.5, Math.min(2.5, (link.weight ?? 0.2) * 2))}
                nodeColor={(node: any) => node.color}
                nodeLabel={(node: any) => `${node.name ?? node.id}`}
                onNodeClick={(node: any) => {
                  setSelected(String(node.id));
                  setSpotlightItems([String(node.id)]);
                }}
                onBackgroundClick={() => {
                  setSelected(null);
                  setSpotlightItems(null);
                }}
                nodeThreeObject={(node: any) => {
                  const geom = new THREE.SphereGeometry(0.6 + (node.size ?? 1) * 0.05, 16, 16);
                  const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(node.color ?? "#7bffe6") });
                  return new THREE.Mesh(geom, mat);
                }}
              />
            </div>
          ) : (
            <div className="buildlab-tree" ref={treeWrapRef}>
              <svg className="buildlab-tree-svg" width={treeSize.w} height={treeSize.h} viewBox={`0 0 ${treeSize.w} ${treeSize.h}`}>
                <defs>
                  <radialGradient id="treeGlow" cx="50%" cy="50%" r="60%">
                    <stop offset="0%" stopColor="rgba(255,255,255,.08)" />
                    <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                  </radialGradient>
                </defs>
                <circle cx={treeLayout.center.x} cy={treeLayout.center.y} r={Math.min(treeSize.w, treeSize.h) * 0.45} fill="url(#treeGlow)" />
                {Array.from({ length: maxDepth }).map((_, i) => {
                  const r = (i + 1) * Math.min(treeSize.w, treeSize.h) / (maxDepth + 1.8);
                  return <circle key={`ring-${i}`} cx={treeLayout.center.x} cy={treeLayout.center.y} r={r} className="tree-ring" />;
                })}
                {treeData.nodes
                  .filter((n) => n.parent)
                  .map((n) => {
                    const parent = treeLayout.positions[n.parent as string];
                    const child = treeLayout.positions[n.key];
                    if (!parent || !child) return null;
                    return <line key={`edge-${n.key}`} x1={parent.x} y1={parent.y} x2={child.x} y2={child.y} className="tree-link" />;
                  })}
                {treeCrossLinks.map((link) => {
                  const a = treeLayout.positions[link.a];
                  const b = treeLayout.positions[link.b];
                  if (!a || !b) return null;
                  const cx = treeLayout.center.x;
                  const cy = treeLayout.center.y;
                  const path = `M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`;
                  const stroke = colorForMetric(metric, link.score);
                  return <path key={`xlink-${link.a}-${link.b}`} d={path} className="tree-xlink" stroke={stroke} />;
                })}
                {treeData.nodes.map((n) => {
                  const pos = treeLayout.positions[n.key];
                  if (!pos) return null;
                  const v = metricValueSet(n, metric);
                  const size = n.key === "root" ? 20 : 6 + Math.min(14, Math.sqrt(Math.max(1, n.support)) * 1.6);
                  const fill = n.key === "root" ? "rgba(255,255,255,.9)" : colorForMetric(metric, v);
                  const isSelected = selectedTree === n.key;
                  return (
                    <g key={`node-${n.key}`} className={`tree-node ${isSelected ? "active" : ""}`} onClick={() => setSelectedTree(n.key)}>
                      <circle cx={pos.x} cy={pos.y} r={size} fill={fill} stroke="rgba(0,0,0,.65)" strokeWidth={2} />
                      {n.key !== "root" ? (
                        <text x={pos.x} y={pos.y - size - 6} className="tree-label">
                          {n.label}
                        </text>
                      ) : (
                        <text x={pos.x} y={pos.y - size - 6} className="tree-label tree-label-root">
                          {n.label}
                        </text>
                      )}
                      <title>
                        {n.items.length ? `${n.items.length} items • support ${n.support}` : "start"}
                        {n.lift != null ? ` • lift ${fmtNum(n.lift, 2)}` : ""}
                        {n.deltaScore != null ? ` • Δ ${fmtNum(n.deltaScore, 2)}` : ""}
                        {n.pTop != null ? ` • pTop ${fmtPct(n.pTop)}` : ""}
                      </title>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}
          {!hasAnyItems ? (
            <div className="mse-overlay">
              <div className="mse-overlay-inner">
                <div className="mse-overlay-title">
                  {noWorkers
                    ? "NO WORKERS CONNECTED"
                    : noInventory
                      ? "INVENTORY NOT YET REPORTED"
                      : filtersExclude
                        ? "NO WORKERS MATCH FILTERS"
                        : noCombos
                          ? "NO COMBOS MATCH FILTERS"
                          : "NO BUILD DATA"}
                </div>
                <div className="mse-overlay-sub muted">
                  {noWorkers
                    ? "Start a worker to populate builds."
                    : noInventory
                      ? "Workers are live, but heartbeats lack inventory_items."
                      : filtersExclude
                        ? "Relax filters or switch policy."
                        : noCombos
                          ? "Lower min support or Top% to reveal combos."
                          : "Waiting for inventory_items in worker heartbeats…"}
                </div>
                <div className="row" style={{ gap: 10, marginTop: 10, justifyContent: "center", flexWrap: "wrap" }}>
                  <Link className="btn" to="/instances">
                    Go to Instances
                  </Link>
                  <Link className="btn btn-ghost" to="/spy">
                    Open Spy
                  </Link>
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="card">
            <div className="stream-card-title">What-if Recommender</div>
            <div className="muted">Type current items (comma-separated) to rank next picks.</div>
            <input
              className="input"
              placeholder="e.g., tome, dagger, borgar"
              value={whatIfInput}
              onChange={(e) => setWhatIfInput(e.target.value)}
              style={{ marginTop: 8 }}
            />
            <div style={{ marginTop: 10 }}>
              {whatIfKeys.length ? (
                <div className="statline">
                  {whatIfKeys.map((k) => (
                    <span key={k} className="pill">{items[k]?.label ?? k}</span>
                  ))}
                </div>
              ) : (
                <div className="muted">No items matched yet.</div>
              )}
            </div>
            <div style={{ marginTop: 10 }}>
              <div className="badge">BEST NEXT PICKS</div>
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                {whatIfRecs.length ? (
                  whatIfRecs.map((r) => (
                    <div key={r.key} className="row-between" style={{ gap: 10 }}>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 800, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {items[r.key]?.label ?? r.key}
                        </div>
                        <div className="muted">{items[r.key]?.kind ?? "item"}</div>
                      </div>
                      <div className="numeric muted">
                        {metric === "lift" ? fmtNum(r.metrics?.lift ?? null, 2) : metric === "deltaScore" ? fmtNum(r.metrics?.deltaScore ?? null, 2) : fmtPct(r.metrics?.pTop ?? null)}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="muted">No recommendations yet (need more pair stats).</div>
                )}
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              <div className="badge">AVOID / ANTI-SYNERGY</div>
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                {antiSynergy.length ? (
                  antiSynergy.map((p) => (
                    <div key={`${p.a}|${p.b}`} className="row-between" style={{ gap: 10 }}>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 800, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {items[p.a]?.label ?? p.a} + {items[p.b]?.label ?? p.b}
                        </div>
                        <div className="muted">Δ {fmtNum(p.deltaScore, 2)}</div>
                      </div>
                      <span className="pill pill-warn">avoid</span>
                    </div>
                  ))
                ) : (
                  <div className="muted">No strong anti-synergy pairs yet.</div>
                )}
              </div>
            </div>
          </div>

          <div className="card" style={{ minHeight: 520 }}>
            <div className="stream-card-title">Detail</div>
            {viewMode === "tree" ? (
              treeSelectedNode ? (
                <div style={{ padding: 12 }}>
                  <div className="row-between" style={{ alignItems: "baseline", gap: 10 }}>
                    <div>
                      <div style={{ fontWeight: 900, letterSpacing: 0.4 }}>{treeSelectedNode.label}</div>
                      <div className="muted">
                        stage {treeSelectedNode.items.length || 0}
                        {treeSelectedNode.items.length ? ` • support ${treeSelectedNode.support}` : ""}
                      </div>
                    </div>
                  </div>
                  {treeSelectedNode.items.length ? (
                    <div style={{ marginTop: 10 }} className="muted">
                      {treeSelectedNode.items.map((it) => items[it]?.label ?? it).join(" + ")}
                    </div>
                  ) : null}
                  <div style={{ marginTop: 12 }}>
                    <div className="badge">METRICS</div>
                    <div className="row" style={{ gap: 10, marginTop: 8, flexWrap: "wrap" }}>
                      <span className="pill">support {treeSelectedNode.support}</span>
                      <span className="pill">lift {fmtNum(treeSelectedNode.lift, 2)}</span>
                      <span className="pill">Δscore {fmtNum(treeSelectedNode.deltaScore, 2)}</span>
                      <span className="pill">p(top) {fmtPct(treeSelectedNode.pTop)}</span>
                    </div>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <div className="badge">BEST NEXT PICKS</div>
                    <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                      {treeSelectedNode.children.length ? (
                        treeSelectedNode.children.map((c) => (
                          <button
                            key={c.key}
                            className="btn btn-ghost"
                            style={{ justifyContent: "space-between", gap: 10 }}
                            onClick={() => setSelectedTree(c.key)}
                          >
                            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.label}</span>
                            <span className="numeric muted">
                              {metric === "lift" ? fmtNum(c.lift, 2) : metric === "deltaScore" ? fmtNum(c.deltaScore, 2) : fmtPct(c.pTop)}
                            </span>
                          </button>
                        ))
                      ) : (
                        <div className="muted">No further refinements at this depth.</div>
                      )}
                    </div>
                  </div>
                  <div style={{ marginTop: 12 }}>
                    <button className="btn" onClick={() => onSpotlightBranch(treeSelectedNode)}>
                      Spotlight branch in web
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ padding: 12 }} className="muted">
                  Click a node in the tree to inspect build steps and next picks.
                </div>
              )
            ) : selectedItem ? (
            <div style={{ padding: 12 }}>
              <div className="row-between" style={{ alignItems: "baseline", gap: 10 }}>
                <div>
                  <div style={{ fontWeight: 900, letterSpacing: 0.4 }}>{selectedItem.label}</div>
                  <div className="muted">
                    {selectedItem.kind}
                    {selectedItem.tier != null ? ` • tier ${selectedItem.tier}` : ""}
                    {selectedItem.rarity ? ` • ${selectedItem.rarity}` : ""}
                  </div>
                </div>
                <button className={`btn btn-ghost ${selectedPinned ? "active" : ""}`} onClick={onPinSelected} title="pin/lock node">
                  {selectedPinned ? "PINNED" : "PIN"}
                </button>
              </div>

              <div style={{ marginTop: 10 }}>
                <div className="badge">BEST PARTNERS</div>
                <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                  {selectedPartners.length ? (
                    selectedPartners.map((p) => (
                      <button
                        key={p.other}
                        className="btn btn-ghost"
                        style={{ justifyContent: "space-between", gap: 10 }}
                        onClick={() => onHighlightPair(p.stat)}
                      >
                        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{items[p.other]?.label ?? p.other}</span>
                        <span className="numeric muted">
                          {metric === "lift" ? fmtNum(p.stat.lift, 2) : metric === "deltaScore" ? fmtNum(p.stat.deltaScore, 2) : fmtPct(p.stat.pTop)}
                        </span>
                      </button>
                    ))
                  ) : (
                    <div className="muted">No pair stats yet (raise worker count or lower min support).</div>
                  )}
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <div className="badge">EXAMPLE RUNS</div>
                <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
                  {(examplesQ.data?.examples ?? []).length ? (
                    (examplesQ.data?.examples ?? []).map((ex) => (
                      <div key={ex.run_id} className="row-between" style={{ gap: 10 }}>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 900, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {ex.run_id}
                          </div>
                          <div className="muted">{ex.worker_id ?? "archived run"}</div>
                          {ex.clip_url ? (
                            <a className="btn btn-ghost btn-compact" href={ex.clip_url} target="_blank" rel="noreferrer">
                              clip
                            </a>
                          ) : null}
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div className="numeric">{ex.final_score == null ? "—" : fmtNum(ex.final_score, 2)}</div>
                          <div className="muted">{ex.is_verified ? "verified" : "unverified"}</div>
                        </div>
                      </div>
                    ))
                  ) : selectedWorkers.length ? (
                    selectedWorkers.map((r) => (
                      <div key={r.iid} className="row-between" style={{ gap: 10 }}>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 900, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.iid}</div>
                          <div className="muted">{r.policy}</div>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div className="numeric">{fmtNum(r.score, 2)}</div>
                          <div className="muted numeric">{fmtNum(r.ageS, 1)}s ago</div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="muted">No archived or live examples yet for this combo.</div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: 12 }} className="muted">
              Click a node in the graph to inspect partners, anti-synergies, and example runs.
            </div>
          )}
        </div>
      </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="stream-card-title">Combo Leaderboard (Pairs)</div>
        <div className="muted" style={{ marginTop: 4 }}>
          Click a row to spotlight the subgraph.
        </div>
        <div className="lb-table" style={{ marginTop: 10 }}>
          <div className="lb-row lb-head">
            <div>#</div>
            <div />
            <div>Combo</div>
            <div>Support</div>
            <div>{metric === "lift" ? "Lift" : metric === "deltaScore" ? "ΔScore" : "P(top)"}</div>
            <div />
            <div className="lb-extra">Support%</div>
            <div className="lb-extra">Conf</div>
            <div className="lb-spark lb-extra">Log-Odds</div>
          </div>
          <div className="lb-list" style={{ ["--lb-row-count" as any]: topPairsForTable.length } as any}>
            {topPairsForTable.map((p, i) => (
              <div
                key={`${p.a}|${p.b}`}
                className="lb-row lb-item"
                style={{ ["--lb-row-i" as any]: i } as any}
                onClick={() => onHighlightPair(p)}
                role="button"
                tabIndex={0}
              >
                <div className="muted">{i + 1}</div>
                <div className="lb-lane">
                  <span className="lb-lane-mark idle">•</span>
                </div>
                <div className="lb-agent">
                  <span className="lb-agent-name">
                    {items[p.a]?.label ?? p.a} + {items[p.b]?.label ?? p.b}
                  </span>
                </div>
                <div className="numeric">{p.support}</div>
                <div className="numeric">
                  {metric === "lift" ? fmtNum(p.lift, 2) : metric === "deltaScore" ? fmtNum(p.deltaScore, 2) : fmtPct(p.pTop)}
                </div>
                <div className="muted">—</div>
                <div className="lb-extra numeric">{fmtPct(p.supportRate)}</div>
                <div className="lb-extra numeric">{fmtPct(p.confidence)}</div>
                <div className="lb-spark lb-extra">
                  <span className="numeric">{fmtNum(p.logOdds, 2)}</span>
                </div>
                <div className="lbChipRail" />
              </div>
            ))}
          </div>
          {!topPairsForTable.length ? <div className="muted">No combos yet — waiting for build data and more workers.</div> : null}
        </div>
      </div>
    </div>
  );
}
