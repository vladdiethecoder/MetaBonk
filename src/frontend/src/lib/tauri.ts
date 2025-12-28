export function isTauri() {
  if (typeof window === "undefined") return false;
  const w = window as any;
  return Boolean(
    w.__TAURI__ ||
      w.__TAURI_INTERNALS__ ||
      w.__TAURI_IPC__ ||
      w.__TAURI_METADATA__ ||
      w.__TAURI__?.invoke ||
      w.__TAURI_INTERNALS__?.invoke ||
      w.__TAURI_IPC__?.invoke,
  );
}
