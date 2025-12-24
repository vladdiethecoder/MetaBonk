import { useEffect, useState } from "react";

let activeWebgl = 0;
let webglLostCount = 0;
let lastLostAt = 0;
const listeners = new Set<(count: number) => void>();

export function bumpWebglCount(delta: number) {
  activeWebgl = Math.max(0, activeWebgl + delta);
  listeners.forEach((fn) => fn(activeWebgl));
}

export function reportWebglLost() {
  webglLostCount += 1;
  lastLostAt = Date.now();
}

export function useWebglCounter() {
  const [count, setCount] = useState(activeWebgl);
  useEffect(() => {
    listeners.add(setCount);
    setCount(activeWebgl);
    return () => {
      listeners.delete(setCount);
    };
  }, []);
  return count;
}

export function getWebglLostStats() {
  return { count: webglLostCount, lastLostAt };
}
