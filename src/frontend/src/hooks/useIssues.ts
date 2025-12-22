import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchInstances } from "../api";
import type { Event, Heartbeat, InstanceView } from "../api";
import { useEventStream } from "./useEventStream";

export type IssueSeverity = "low" | "medium" | "high";

export type IssueItem = {
  id: string;
  label: string;
  severity: IssueSeverity;
  count: number;
  instances: string[];
  firstSeen?: number | null;
  lastSeen?: number | null;
  hint?: string | null;
};

const ISSUE_HINTS: Record<string, string> = {
  STREAM_MISSING_NO_PIPEWIRE: "Check PipeWire + streamer service.",
  STREAM_MISSING_NO_URL: "Streamer not reporting URL.",
  STREAM_STALE_NO_FRAMES: "Encoder stalled; restart stream pipeline.",
  STREAM_MAX_CLIENTS: "Too many clients; close extra viewers.",
  WORKER_OFFLINE: "Worker heartbeat stopped.",
  WORKER_CRASHED: "Check last crash logs.",
  INVENTORY_EMPTY: "Bridge inventory feed missing.",
};

const severityFor = (id: string): IssueSeverity => {
  if (id.includes("CRASH") || id.includes("PIPEWIRE")) return "high";
  if (id.includes("MISSING") || id.includes("STUCK")) return "medium";
  return "low";
};

export const deriveReasonCode = (hb?: Heartbeat | null): string | null => {
  if (!hb) return null;
  const status = String(hb.status || "").toLowerCase();
  if (status && status !== "running") {
    if (status.includes("crash")) return "WORKER_CRASHED";
    if (status.includes("offline")) return "WORKER_OFFLINE";
    return `WORKER_${status.toUpperCase()}`;
  }
  if (hb.pipewire_node_ok === false || status.includes("no_pipewire")) return "STREAM_MISSING_NO_PIPEWIRE";
  if (!hb.stream_url) return "STREAM_MISSING_NO_URL";
  if (hb.stream_ok === false) return "STREAM_STALE_NO_FRAMES";
  if (hb.stream_max_clients && hb.stream_active_clients != null && hb.stream_active_clients >= hb.stream_max_clients) {
    return "STREAM_MAX_CLIENTS";
  }
  if (hb.inventory_items && Array.isArray(hb.inventory_items) && hb.inventory_items.length === 0 && hb.step > 50) {
    return "INVENTORY_EMPTY";
  }
  return null;
};

const classifyEventIssue = (ev: Event): string | null => {
  const type = String(ev.event_type || "").toLowerCase();
  const msg = String(ev.message || "").toLowerCase();
  if (type.includes("error") || msg.includes("error")) return "EVENT_ERRORS";
  if (type.includes("stuck") || msg.includes("stuck")) return "STUCK_DETECTED";
  if (type.includes("stream") && (msg.includes("missing") || msg.includes("stale"))) return "STREAM_GLITCHES";
  if (type.includes("menu") || msg.includes("menu")) return "MENU_ISSUES";
  return null;
};

export default function useIssues(limit = 240) {
  const events = useEventStream(limit);
  const instQ = useQuery({ queryKey: ["instances"], queryFn: fetchInstances, refetchInterval: 2500 });

  return useMemo(() => {
    const issues = new Map<string, IssueItem>();

    const pushIssue = (id: string, instanceId?: string | null, ts?: number | null) => {
      const current = issues.get(id) ?? {
        id,
        label: id.replace(/_/g, " "),
        severity: severityFor(id),
        count: 0,
        instances: [],
        hint: ISSUE_HINTS[id] ?? null,
        firstSeen: ts ?? null,
        lastSeen: ts ?? null,
      };
      current.count += 1;
      if (instanceId && !current.instances.includes(instanceId)) current.instances.push(instanceId);
      if (ts != null) {
        current.firstSeen = current.firstSeen == null ? ts : Math.min(current.firstSeen, ts);
        current.lastSeen = current.lastSeen == null ? ts : Math.max(current.lastSeen, ts);
      }
      issues.set(id, current);
    };

    const instanceMap = (instQ.data ?? {}) as Record<string, InstanceView>;
    Object.values(instanceMap).forEach((view) => {
      const hb = view.heartbeat;
      const reason = deriveReasonCode(hb);
      if (reason) pushIssue(reason, hb.instance_id, hb.ts);
    });

    events.forEach((ev) => {
      const issue = classifyEventIssue(ev);
      if (!issue) return;
      pushIssue(issue, ev.instance_id ?? null, ev.ts ?? null);
    });

    return Array.from(issues.values())
      .sort((a, b) => {
        const sev = { high: 3, medium: 2, low: 1 } as const;
        if (sev[b.severity] !== sev[a.severity]) return sev[b.severity] - sev[a.severity];
        return b.count - a.count;
      })
      .slice(0, 12);
  }, [events, instQ.data]);
}
