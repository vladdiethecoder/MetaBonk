import { useRef } from "react";
import useElementSize from "../hooks/useElementSize";

type ElementSizerProps = {
  className?: string;
  children: (size: { width: number; height: number }) => React.ReactNode;
};

export default function ElementSizer({ className, children }: ElementSizerProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const size = useElementSize(ref);
  return (
    <div ref={ref} className={className} style={{ width: "100%", height: "100%" }}>
      {size.width > 0 && size.height > 0 ? children(size) : null}
    </div>
  );
}
