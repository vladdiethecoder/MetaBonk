export function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export function fmtNum(n: number | null | undefined) {
  if (n === null || n === undefined) return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString();
}

export function fmtFixed(n: number | null | undefined, digits = 2) {
  if (n === null || n === undefined) return "—";
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

export function fmtPct01(v01: number | null | undefined, digits = 0) {
  if (v01 === null || v01 === undefined) return "—";
  const v = Number(v01);
  if (!Number.isFinite(v)) return "—";
  return `${Math.round(clamp01(v) * 100 * Math.pow(10, digits)) / Math.pow(10, digits)}%`;
}

export function timeAgo(tsSeconds: number | null | undefined) {
  if (!tsSeconds) return "—";
  const ms = Number(tsSeconds) * 1000;
  if (!Number.isFinite(ms)) return "—";
  const d = Date.now() - ms;
  if (!Number.isFinite(d)) return "—";
  const s = Math.max(0, Math.round(d / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  const days = Math.round(h / 24);
  return `${days}d ago`;
}

export async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.top = "-1000px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

