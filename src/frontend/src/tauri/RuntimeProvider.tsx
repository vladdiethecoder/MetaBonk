import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { isTauri } from "../lib/tauri";
import { tauriInvoke, tauriListen } from "../lib/tauri_api";

export type LogLine = {
  ts: number;
  stream: "stdout" | "stderr";
  line: string;
};

type RuntimeContextValue = {
  tauriReady: boolean;
  omegaRunning: boolean;
  discoveryRunning: boolean;
  omegaLogs: LogLine[];
  discoveryLogs: LogLine[];
  lastError: string | null;
  clearOmegaLogs: () => void;
  clearDiscoveryLogs: () => void;
};

const Ctx = createContext<RuntimeContextValue | null>(null);

export function useTauriRuntime() {
  const v = useContext(Ctx);
  if (!v) {
    return {
      tauriReady: false,
      omegaRunning: false,
      discoveryRunning: false,
      omegaLogs: [],
      discoveryLogs: [],
      lastError: null,
      clearOmegaLogs: () => {},
      clearDiscoveryLogs: () => {},
    } as RuntimeContextValue;
  }
  return v;
}

export default function RuntimeProvider({ children }: { children: ReactNode }) {
  const [tauriReady, setTauriReady] = useState(false);
  const [omegaRunning, setOmegaRunning] = useState(false);
  const [discoveryRunning, setDiscoveryRunning] = useState(false);
  const [omegaLogs, setOmegaLogs] = useState<LogLine[]>([]);
  const [discoveryLogs, setDiscoveryLogs] = useState<LogLine[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    const ok = isTauri();
    setTauriReady(ok);
  }, []);

  useEffect(() => {
    if (!tauriReady) return;
    let cancelled = false;
    let unlistenOut: null | (() => void) = null;
    let unlistenErr: null | (() => void) = null;
    let unlistenExit: null | (() => void) = null;
    let unlistenDiscOut: null | (() => void) = null;
    let unlistenDiscErr: null | (() => void) = null;
    let unlistenDiscExit: null | (() => void) = null;

    const init = async () => {
      try {
        unlistenOut = await tauriListen<string>("omega-stdout", (line) => {
          setOmegaLogs((prev) => [...prev.slice(-600), { ts: Date.now(), stream: "stdout", line }]);
        });
        unlistenErr = await tauriListen<string>("omega-stderr", (line) => {
          setOmegaLogs((prev) => [...prev.slice(-600), { ts: Date.now(), stream: "stderr", line }]);
        });
        unlistenExit = await tauriListen<boolean>("omega-exit", () => {
          setOmegaRunning(false);
          setOmegaLogs((prev) => [
            ...prev.slice(-600),
            { ts: Date.now(), stream: "stderr", line: "[tauri] omega exited" },
          ]);
        });

        unlistenDiscOut = await tauriListen<string>("discovery-stdout", (line) => {
          setDiscoveryLogs((prev) => [...prev.slice(-500), { ts: Date.now(), stream: "stdout", line }]);
        });
        unlistenDiscErr = await tauriListen<string>("discovery-stderr", (line) => {
          setDiscoveryLogs((prev) => [...prev.slice(-500), { ts: Date.now(), stream: "stderr", line }]);
        });
        unlistenDiscExit = await tauriListen<boolean>("discovery-exit", () => {
          setDiscoveryRunning(false);
          setDiscoveryLogs((prev) => [
            ...prev.slice(-500),
            { ts: Date.now(), stream: "stderr", line: "[tauri] discovery exited" },
          ]);
        });

        const [o, d] = await Promise.all([
          tauriInvoke<boolean>("omega_running"),
          tauriInvoke<boolean>("discovery_running"),
        ]);
        if (!cancelled) {
          setOmegaRunning(Boolean(o));
          setDiscoveryRunning(Boolean(d));
        }
      } catch (e: any) {
        if (!cancelled) setLastError(String(e?.message ?? e));
      }
    };

    init();
    return () => {
      cancelled = true;
      try {
        unlistenOut?.();
      } catch {}
      try {
        unlistenErr?.();
      } catch {}
      try {
        unlistenExit?.();
      } catch {}
      try {
        unlistenDiscOut?.();
      } catch {}
      try {
        unlistenDiscErr?.();
      } catch {}
      try {
        unlistenDiscExit?.();
      } catch {}
    };
  }, [tauriReady]);

  useEffect(() => {
    if (!tauriReady) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const [o, d] = await Promise.all([
          tauriInvoke<boolean>("omega_running"),
          tauriInvoke<boolean>("discovery_running"),
        ]);
        if (!cancelled) {
          setOmegaRunning(Boolean(o));
          setDiscoveryRunning(Boolean(d));
        }
      } catch {
        // best-effort
      }
    };
    poll();
    const t = window.setInterval(poll, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [tauriReady]);

  const value = useMemo<RuntimeContextValue>(
    () => ({
      tauriReady,
      omegaRunning,
      discoveryRunning,
      omegaLogs,
      discoveryLogs,
      lastError,
      clearOmegaLogs: () => setOmegaLogs([]),
      clearDiscoveryLogs: () => setDiscoveryLogs([]),
    }),
    [tauriReady, omegaRunning, discoveryRunning, omegaLogs, discoveryLogs, lastError],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

