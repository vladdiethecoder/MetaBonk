import { useLocation, useNavigate } from "react-router-dom";

export type ContextFilters = {
  run: string;
  policy: string;
  window: string;
  env: string;
  seed: string;
};

export const CONTEXT_WINDOW_OPTIONS = [
  { value: "all", label: "All time" },
  { value: "5m", label: "Last 5m" },
  { value: "1h", label: "Last 1h" },
  { value: "24h", label: "Last 24h" },
];

const DEFAULTS: ContextFilters = {
  run: "all",
  policy: "all",
  window: "all",
  env: "all",
  seed: "all",
};

const PARAMS: Record<keyof ContextFilters, string> = {
  run: "ctxRun",
  policy: "ctxPolicy",
  window: "ctxWindow",
  env: "ctxEnv",
  seed: "ctxSeed",
};

const normalize = (value: string | null | undefined) => {
  if (!value) return "all";
  const v = String(value).trim();
  return v ? v : "all";
};

export const parseWindowSeconds = (value: string): number | null => {
  const v = String(value || "all").toLowerCase();
  if (v === "all") return null;
  if (v.endsWith("m")) {
    const n = Number(v.slice(0, -1));
    return Number.isFinite(n) ? n * 60 : null;
  }
  if (v.endsWith("h")) {
    const n = Number(v.slice(0, -1));
    return Number.isFinite(n) ? n * 3600 : null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

export default function useContextFilters() {
  const loc = useLocation();
  const nav = useNavigate();
  const params = new URLSearchParams(loc.search);
  const ctx: ContextFilters = {
    run: normalize(params.get(PARAMS.run)),
    policy: normalize(params.get(PARAMS.policy)),
    window: normalize(params.get(PARAMS.window)),
    env: normalize(params.get(PARAMS.env)),
    seed: normalize(params.get(PARAMS.seed)),
  };

  const setCtx = (key: keyof ContextFilters, value: string) => {
    const next = new URLSearchParams(loc.search);
    const v = normalize(value);
    if (v === "all") next.delete(PARAMS[key]);
    else next.set(PARAMS[key], v);
    nav({ pathname: loc.pathname, search: next.toString() }, { replace: true });
  };

  const clearAll = () => {
    const next = new URLSearchParams(loc.search);
    (Object.values(PARAMS) as string[]).forEach((k) => next.delete(k));
    nav({ pathname: loc.pathname, search: next.toString() }, { replace: true });
  };

  return {
    ctx,
    setCtx,
    clearAll,
    windowSeconds: parseWindowSeconds(ctx.window),
  };
}
