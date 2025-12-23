import { useEffect, useRef, useState } from "react";
import type { RefObject, UIEvent } from "react";

type VirtualRows = {
  viewportRef: RefObject<HTMLDivElement>;
  onScroll: (ev: UIEvent<HTMLDivElement>) => void;
  start: number;
  end: number;
  offsetTop: number;
  offsetBottom: number;
  viewportHeight: number;
};

export default function useVirtualRows(total: number, rowHeight: number, overscan = 6): VirtualRows {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(360);

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const update = () => setViewportHeight(el.clientHeight || 360);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const onScroll = (ev: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(ev.currentTarget.scrollTop);
  };

  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(total, Math.ceil((scrollTop + viewportHeight) / rowHeight) + overscan);
  const offsetTop = start * rowHeight;
  const offsetBottom = Math.max(0, (total - end) * rowHeight);

  return { viewportRef, onScroll, start, end, offsetTop, offsetBottom, viewportHeight };
}
