import { isTauri } from "./tauri";

let coreMod: Promise<typeof import("@tauri-apps/api/core")> | null = null;
let eventMod: Promise<typeof import("@tauri-apps/api/event")> | null = null;

export type OmegaMode = "train" | "play" | "dream";
export type ObsBackend = "" | "detections" | "pixels" | "hybrid";
export type DiscoveryAdapter = "mock" | "synthetic-eye";

export type DiscoveryStatus = {
  env_id: string;
  cache_dir: string;
  phase_0_complete: boolean;
  phase_1_complete: boolean;
  phase_2_complete: boolean;
  phase_3_complete: boolean;
  ready_for_training: boolean;
  artifacts?: Record<string, string>;
  has_suggested_env?: boolean;
  has_ppo_config?: boolean;
};

export type BenchResult = {
  ok?: boolean;
  fps?: number;
  per_frame_ms?: { mean?: number; p50?: number; p95?: number };
  frames?: number;
  elapsed_s?: number;
  mode?: string;
  audit_log?: string;
  sock?: string;
};

export type StartOmegaArgs = {
  mode: OmegaMode;
  workers: number;
  env_id?: string | null;
  synthetic_eye?: boolean | null;
  synthetic_eye_lockstep?: boolean | null;
  obs_backend?: ObsBackend | null;
  use_discovered_actions?: boolean | null;
  silicon_cortex?: boolean | null;
};

export type StartDiscoveryArgs = {
  env_id: string;
  env_adapter?: DiscoveryAdapter | null;
  exploration_budget?: number | null;
  hold_frames?: number | null;
};

function requireTauri() {
  if (!isTauri()) {
    throw new Error("Tauri API is not available in browser mode");
  }
}

async function getCore() {
  requireTauri();
  if (!coreMod) coreMod = import("@tauri-apps/api/core");
  return coreMod;
}

async function getEvent() {
  requireTauri();
  if (!eventMod) eventMod = import("@tauri-apps/api/event");
  return eventMod;
}

export async function tauriInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await getCore();
  return invoke<T>(cmd, args);
}

export async function tauriListen<T>(event: string, handler: (payload: T) => void): Promise<() => void> {
  const { listen } = await getEvent();
  const unlisten = await listen<T>(event, (evt) => handler(evt.payload));
  return () => {
    try {
      unlisten();
    } catch {
      // ignore
    }
  };
}

