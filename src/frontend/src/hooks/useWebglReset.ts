import { useSyncExternalStore } from "react";

let nonce = 0;
const subs = new Set<() => void>();

export function bumpWebglReset() {
  nonce += 1;
  for (const cb of subs) cb();
}

export function useWebglResetNonce() {
  return useSyncExternalStore(
    (cb) => {
      subs.add(cb);
      return () => subs.delete(cb);
    },
    () => nonce,
    () => nonce,
  );
}
