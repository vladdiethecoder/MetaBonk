import type { ReactNode } from "react";
import { useEffect } from "react";

type QueryLike = {
  isLoading?: boolean;
  isError?: boolean;
  error?: unknown;
};

type QueryStateGateProps = {
  label: string;
  queries?: QueryLike[];
  children: ReactNode;
};

export default function QueryStateGate({ label, queries, children }: QueryStateGateProps) {
  const list = Array.isArray(queries) ? queries : [];
  const loading = list.some((q) => q.isLoading);
  const errorQuery = list.find((q) => q.isError);
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const store = ((window as any).__mbQueryStates ||= {});
    store[label] = {
      loading,
      error: errorQuery ? (errorQuery.error as any)?.message ?? String(errorQuery.error ?? "Unknown error") : null,
      updatedAt: Date.now(),
    };
  }, [label, loading, errorQuery]);
  if (loading) {
    return (
      <div className="card error-card">
        <div className="row-between">
          <h2>{label}</h2>
          <span className="badge">loading</span>
        </div>
        <div className="muted">Fetching data…</div>
      </div>
    );
  }
  if (errorQuery) {
    const msg = (errorQuery.error as any)?.message ?? String(errorQuery.error ?? "Unknown error");
    return (
      <div className="card error-card">
        <div className="row-between">
          <h2>{label}</h2>
          <span className="badge">error</span>
        </div>
        <div className="muted">{msg}</div>
      </div>
    );
  }
  return <>{children}</>;
}
