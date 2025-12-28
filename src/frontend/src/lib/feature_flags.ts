export type FeatureFlag = "betting" | "poll" | "ghosts";

export type FeatureFlags = Record<FeatureFlag, boolean>;

const DEFAULTS: FeatureFlags = {
  betting: false,
  poll: false,
  ghosts: false,
};

const parseCsv = (raw: unknown): Set<string> => {
  if (raw == null) return new Set();
  const s = String(raw).trim();
  if (!s) return new Set();
  return new Set(
    s
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean)
      .map((x) => x.toLowerCase()),
  );
};

export function getFeatureFlags(qs?: URLSearchParams): FeatureFlags {
  const flags: FeatureFlags = { ...DEFAULTS };
  if (typeof window === "undefined") return flags;

  const q = qs ?? new URLSearchParams(window.location.search);
  const env = (import.meta as any)?.env ?? {};
  const w = window as any;

  const enabled = new Set<string>();
  const disabled = new Set<string>();

  for (const x of parseCsv(env.VITE_FEATURE_FLAGS)) enabled.add(x);
  for (const x of parseCsv(w.__MB_FEATURE_FLAGS__)) enabled.add(x);
  for (const x of parseCsv(q.get("ff"))) enabled.add(x);
  for (const x of parseCsv(q.get("noff"))) disabled.add(x);

  if (enabled.has("all")) {
    enabled.delete("all");
    enabled.add("betting");
    enabled.add("poll");
    enabled.add("ghosts");
  }

  for (const k of Object.keys(flags) as FeatureFlag[]) {
    if (disabled.has(k)) flags[k] = false;
    if (enabled.has(k)) flags[k] = true;
  }

  return flags;
}

