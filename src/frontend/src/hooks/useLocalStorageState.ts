import { useCallback, useEffect, useMemo, useState } from "react";

type Options<T> = {
  serialize?: (value: T) => string;
  deserialize?: (raw: string) => T;
};

const EVENT_NAME = "mb:local-storage";

export default function useLocalStorageState<T>(key: string, initialValue: T, options: Options<T> = {}) {
  const serialize = useMemo(() => options.serialize ?? ((v: T) => JSON.stringify(v)), [options.serialize]);
  const deserialize = useMemo(
    () => options.deserialize ?? ((raw: string) => JSON.parse(raw) as T),
    [options.deserialize],
  );

  const read = useCallback((): T => {
    if (typeof window === "undefined") return initialValue;
    try {
      const raw = window.localStorage.getItem(key);
      if (raw == null) return initialValue;
      return deserialize(raw);
    } catch {
      return initialValue;
    }
  }, [deserialize, initialValue, key]);

  const [value, setValue] = useState<T>(() => read());

  useEffect(() => {
    setValue(read());
  }, [read]);

  const setAndPersist = useCallback(
    (next: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const resolved = typeof next === "function" ? (next as (p: T) => T)(prev) : next;
        if (typeof window === "undefined") return resolved;
        try {
          window.localStorage.setItem(key, serialize(resolved));
          window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: { key } }));
        } catch {
          // ignore
        }
        return resolved;
      });
    },
    [key, serialize],
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onStorage = (evt: StorageEvent) => {
      if (evt.storageArea !== window.localStorage) return;
      if (evt.key !== key) return;
      setValue(read());
    };
    const onCustom = (evt: Event) => {
      const detail = (evt as CustomEvent)?.detail as any;
      if (detail?.key !== key) return;
      setValue(read());
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(EVENT_NAME, onCustom);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(EVENT_NAME, onCustom);
    };
  }, [key, read]);

  return [value, setAndPersist] as const;
}

