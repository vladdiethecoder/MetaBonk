import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchOverviewIssues } from "../api";
import type { Heartbeat, OverviewIssue } from "../api";
import { HEARTBEAT_SCHEMA_VERSION } from "../lib/schema";

export type IssueSeverity = "low" | "medium" | "high";

export type IssueItem = {
  id: string;
  code?: string;
  label: string;
  severity: IssueSeverity;
  count: number;
  instances: string[];
  firstSeen?: number | null;
  lastSeen?: number | null;
  hint?: string | null;
  evidence?: Array<{ kind: string; url: string; label?: string | null }>;
  acknowledged?: boolean;
  muted?: boolean;
};

const ISSUE_HINTS: Record<string, string> = {
  STREAM_MISSING_NO_PIPEWIRE: "Check PipeWire + streamer service.",
  STREAM_MISSING_SESSION: "PipeWire session manager missing (WirePlumber).",
  STREAM_MISSING_NO_URL: "Streamer not reporting URL.",
  STREAM_STALE_NO_FRAMES: "Encoder stalled; restart stream pipeline.",
  STREAM_NO_KEYFRAME: "No keyframes; encoder stalled or GOP mis-set.",
  STREAM_MAX_CLIENTS: "Too many clients; close extra viewers.",
  WORKER_OFFLINE: "Worker heartbeat stopped.",
  WORKER_CRASHED: "Check last crash logs.",
  INVENTORY_EMPTY: "Bridge inventory feed missing.",
  HEARTBEAT_SCHEMA_MISMATCH: "Heartbeat schema mismatch; update worker or UI.",
};

const severityFor = (id: string): IssueSeverity => {
  if (id.includes("CRASH") || id.includes("PIPEWIRE")) return "high";
  if (id.includes("MISSING") || id.includes("STUCK")) return "medium";
  return "low";
};

export const deriveReasonCode = (hb?: Heartbeat | null): string | null => {
  if (!hb) return null;
  if (hb.schema_version != null && Number(hb.schema_version) !== HEARTBEAT_SCHEMA_VERSION) return "HEARTBEAT_SCHEMA_MISMATCH";
  const status = String(hb.status || "").toLowerCase();
  if (status && status !== "running") {
    if (status.includes("crash")) return "WORKER_CRASHED";
    if (status.includes("offline")) return "WORKER_OFFLINE";
    return `WORKER_${status.toUpperCase()}`;
  }
  if (hb.pipewire_ok === false || hb.pipewire_node_ok === false || status.includes("no_pipewire")) return "STREAM_MISSING_NO_PIPEWIRE";
  if (hb.pipewire_session_ok === false) return "STREAM_MISSING_SESSION";
  if (!hb.stream_url) return "STREAM_MISSING_NO_URL";
  if (hb.stream_ok === false) return "STREAM_STALE_NO_FRAMES";
  if (hb.stream_ok && hb.stream_last_frame_ts != null && hb.stream_keyframe_ts == null) return "STREAM_NO_KEYFRAME";
  if (hb.stream_max_clients && hb.stream_active_clients != null && hb.stream_active_clients >= hb.stream_max_clients) {
    return "STREAM_MAX_CLIENTS";
  }
  if (hb.inventory_items && Array.isArray(hb.inventory_items) && hb.inventory_items.length === 0 && hb.step > 50) {
    return "INVENTORY_EMPTY";
  }
  return null;
};

export default function useIssues(limit = 240) {
  const issuesQ = useQuery({
    queryKey: ["overviewIssues", limit],
    queryFn: () => fetchOverviewIssues(Math.max(120, limit)),
    refetchInterval: 5000,
  });

  return useMemo(() => {
    const raw = (issuesQ.data ?? []) as OverviewIssue[];
    return raw.map((issue) => ({
      id: issue.id,
      code: issue.code ?? issue.id,
      label: issue.label,
      severity: issue.severity,
      count: issue.count,
      instances: issue.instances ?? [],
      firstSeen: (issue as any).first_seen ?? (issue as any).firstSeen ?? null,
      lastSeen: (issue as any).last_seen ?? (issue as any).lastSeen ?? null,
      hint: issue.hint ?? ISSUE_HINTS[issue.code ?? issue.label.replace(/ /g, "_")] ?? null,
      evidence: (issue as any).evidence ?? [],
      acknowledged: (issue as any).acknowledged ?? false,
      muted: (issue as any).muted ?? false,
    })) as IssueItem[];
  }, [issuesQ.data]);
}
