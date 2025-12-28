import { useEffect, useRef } from "react";
import { isTauri } from "../lib/tauri";

export default function useTauriEvent<T>(eventName: string, handler: (payload: T) => void) {
  const handlerRef = useRef(handler);
  useEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: null | (() => void) = null;
    (async () => {
      try {
        const { listen } = await import("@tauri-apps/api/event");
        unlisten = await listen<T>(eventName, (event) => {
          handlerRef.current(event.payload as T);
        });
      } catch {
        // ignore
      }
    })();
    return () => {
      if (unlisten) unlisten();
    };
  }, [eventName]);
}
