import { useMemo, useState } from "react";
import useContextFilters, { CONTEXT_WINDOW_OPTIONS } from "../hooks/useContextFilters";
import useSavedViews from "../hooks/useSavedViews";

export default function ContextBar() {
  const { ctx, setCtx, clearAll } = useContextFilters();
  const { views, saveView, removeView } = useSavedViews();
  const [viewName, setViewName] = useState("");
  const [selectedViewId, setSelectedViewId] = useState<string>("");
  const hasFilters =
    ctx.run !== "all" || ctx.policy !== "all" || ctx.window !== "all" || ctx.env !== "all" || ctx.seed !== "all";
  const selectedView = useMemo(() => views.find((v) => v.id === selectedViewId) ?? null, [views, selectedViewId]);

  const applyView = (id: string) => {
    const view = views.find((v) => v.id === id);
    if (!view) return;
    setCtx("run", view.filters.run);
    setCtx("policy", view.filters.policy);
    setCtx("window", view.filters.window);
    setCtx("env", view.filters.env);
    setCtx("seed", view.filters.seed);
  };

  return (
    <section className="context-bar">
      <div className="context-title">Run Context</div>
      <div className="context-fields">
        <label className="context-field">
          <span className="context-label">Run</span>
          <input
            className="input input-compact"
            placeholder="run id"
            value={ctx.run === "all" ? "" : ctx.run}
            onChange={(e) => setCtx("run", e.target.value)}
          />
        </label>
        <label className="context-field">
          <span className="context-label">Policy</span>
          <input
            className="input input-compact"
            placeholder="policy name"
            value={ctx.policy === "all" ? "" : ctx.policy}
            onChange={(e) => setCtx("policy", e.target.value)}
          />
        </label>
        <label className="context-field">
          <span className="context-label">Time</span>
          <select className="select select-compact" value={ctx.window} onChange={(e) => setCtx("window", e.target.value)}>
            {CONTEXT_WINDOW_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <label className="context-field">
          <span className="context-label">Env</span>
          <input
            className="input input-compact"
            placeholder="env build/hash"
            value={ctx.env === "all" ? "" : ctx.env}
            onChange={(e) => setCtx("env", e.target.value)}
          />
        </label>
        <label className="context-field">
          <span className="context-label">Seed</span>
          <input
            className="input input-compact"
            placeholder="seed set"
            value={ctx.seed === "all" ? "" : ctx.seed}
            onChange={(e) => setCtx("seed", e.target.value)}
          />
        </label>
      </div>
      <div className="context-actions">
        {hasFilters ? (
          <>
            <div className="context-chips">
              {ctx.run !== "all" ? <span className="chip chip-compact">run:{ctx.run}</span> : null}
              {ctx.policy !== "all" ? <span className="chip chip-compact">policy:{ctx.policy}</span> : null}
              {ctx.window !== "all" ? <span className="chip chip-compact">time:{ctx.window}</span> : null}
              {ctx.env !== "all" ? <span className="chip chip-compact">env:{ctx.env}</span> : null}
              {ctx.seed !== "all" ? <span className="chip chip-compact">seed:{ctx.seed}</span> : null}
            </div>
            <button className="btn btn-ghost btn-compact" onClick={clearAll}>
              clear
            </button>
          </>
        ) : (
          <span className="muted">context: all</span>
        )}
      </div>
      <div className="context-saved">
        <div className="context-saved-row">
          <input
            className="input input-compact"
            placeholder="save view name"
            value={viewName}
            onChange={(e) => setViewName(e.target.value)}
          />
          <button
            className="btn btn-ghost btn-compact"
            onClick={() => {
              if (!viewName.trim()) return;
              saveView(viewName.trim(), ctx);
              setViewName("");
            }}
            disabled={!viewName.trim()}
          >
            save view
          </button>
        </div>
        <div className="context-saved-row">
          <select
            className="select select-compact"
            value={selectedViewId}
            onChange={(e) => {
              const id = e.target.value;
              setSelectedViewId(id);
              if (id) applyView(id);
            }}
          >
            <option value="">saved views</option>
            {views.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
          <button
            className="btn btn-ghost btn-compact"
            onClick={() => {
              if (!selectedView) return;
              removeView(selectedView.id);
              setSelectedViewId("");
            }}
            disabled={!selectedView}
          >
            delete
          </button>
        </div>
      </div>
    </section>
  );
}
