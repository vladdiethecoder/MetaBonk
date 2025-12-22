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
    window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: payload }));
  }, []);
}

export const contextDrawerEventName = EVENT_NAME;
