import { useCallback } from "react";

export type ContextDrawerPayload = {
  title: string;
  kind?: "metric" | "event" | "instance" | "run" | "stream" | "other";
  instanceId?: string | null;
  runId?: string | null;
  ts?: number | null;
  details?: Record<string, any> | null;
};

const EVENT_NAME = "mb:context-drawer";

export function useContextDrawer() {
  return useCallback((payload: ContextDrawerPayload) => {
    if (typeof window === "undefined") return;
    try {
      const url = new URL(window.location.href);
      if (payload.instanceId) url.searchParams.set("instance_id", String(payload.instanceId));
      if (payload.runId) url.searchParams.set("run_id", String(payload.runId));
      if (payload.ts != null) url.searchParams.set("ts", String(payload.ts));
      const step = (payload.details as any)?.step;
      if (step != null) url.searchParams.set("step", String(step));
      window.history.replaceState({}, "", url.toString());
    } catch {
      // ignore
    }
    window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: payload }));
  }, []);
}

export const contextDrawerEventName = EVENT_NAME;
