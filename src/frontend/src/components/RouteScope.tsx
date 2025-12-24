import { forwardRef } from "react";
import type { CSSProperties, ReactNode } from "react";

type RouteScopeProps = {
  uiScale?: number;
  tier?: string | null;
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
};

const RouteScope = forwardRef<HTMLDivElement, RouteScopeProps>(({ uiScale, tier, className, style, children }, ref) => {
  const scopeStyle: CSSProperties = {
    ...style,
    ...(uiScale != null ? { ["--ui" as any]: uiScale } : null),
  };

  return (
    <div ref={ref} className={className} style={scopeStyle} data-tier={tier ?? undefined}>
      {children}
    </div>
  );
});

RouteScope.displayName = "RouteScope";

export default RouteScope;
