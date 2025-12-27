export function isTauri() {
  return typeof window !== "undefined" && "__TAURI__" in window;
}

