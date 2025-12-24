import { useEffect } from "react";

export default function useActivationResizeKick(active: boolean) {
  useEffect(() => {
    if (!active) return;
    const id = window.requestAnimationFrame(() => {
      window.dispatchEvent(new Event("resize"));
    });
    return () => window.cancelAnimationFrame(id);
  }, [active]);
}
