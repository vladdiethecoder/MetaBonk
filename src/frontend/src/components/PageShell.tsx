import type { CSSProperties, ReactNode } from "react";
import { useLayoutEffect, useRef } from "react";

type PageShellProps = {
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
};

export default function PageShell({ className = "", style, children }: PageShellProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    if (!import.meta.env.DEV) return;
    const node = ref.current;
    if (!node) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      (window as any).__mbPageSize = { width, height, ts: Date.now() };
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={ref} className={`page page-shell ${className}`.trim()} style={style}>
      {children}
    </div>
  );
}
