import { useCallback, useEffect, useMemo, useState } from "react";
import type { ContextFilters } from "./useContextFilters";

export type SavedView = {
  id: string;
  name: string;
  createdTs: number;
  filters: ContextFilters;
  note?: string | null;
};

const STORAGE_KEY = "mb_saved_views_v1";

const normalize = (value: string) => {
  const v = String(value || "all").trim();
  return v ? v : "all";
};

const normalizeFilters = (filters: ContextFilters): ContextFilters => ({
  run: normalize(filters.run),
  policy: normalize(filters.policy),
  window: normalize(filters.window),
  env: normalize(filters.env),
  seed: normalize(filters.seed),
});

const loadSavedViews = (): SavedView[] => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as SavedView[];
    return Array.isArray(parsed)
      ? parsed.map((v) => ({ ...v, filters: normalizeFilters(v.filters) }))
      : [];
  } catch {
    return [];
  }
};

const persistSavedViews = (views: SavedView[]) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(views));
};

export default function useSavedViews() {
  const [views, setViews] = useState<SavedView[]>(() => loadSavedViews());

  useEffect(() => {
    persistSavedViews(views);
  }, [views]);

  const saveView = useCallback((name: string, filters: ContextFilters, note?: string) => {
    const trimmed = name.trim();
    if (!trimmed) return;
    const normalized = normalizeFilters(filters);
    setViews((prev) => {
      const next = prev.filter((v) => v.name.toLowerCase() !== trimmed.toLowerCase());
      next.unshift({
        id: `view-${Date.now()}`,
        name: trimmed,
        createdTs: Date.now(),
        filters: normalized,
        note: note ?? null,
      });
      return next.slice(0, 20);
    });
  }, []);

  const removeView = useCallback((id: string) => {
    setViews((prev) => prev.filter((v) => v.id !== id));
  }, []);

  const getViewById = useCallback(
    (id: string) => views.find((v) => v.id === id) ?? null,
    [views]
  );

  const list = useMemo(() => views.slice().sort((a, b) => b.createdTs - a.createdTs), [views]);

  return {
    views: list,
    saveView,
    removeView,
    getViewById,
  };
}
