import useLocalStorageState from "./useLocalStorageState";

export type LaunchMode = "train" | "play" | "dream";
export type ObsBackend = "" | "detections" | "pixels" | "hybrid";

export type LaunchConfig = {
  mode: LaunchMode;
  workers: number;
  envId: string;
  syntheticEye: boolean;
  syntheticEyeLockstep: boolean;
  obsBackend: ObsBackend;
  useDiscoveredActions: boolean;
  siliconCortex: boolean;
};

const STORAGE_KEY = "mb:launchConfig:v1";

const defaults: LaunchConfig = {
  mode: "train",
  workers: 1,
  envId: "MegaBonk",
  syntheticEye: true,
  syntheticEyeLockstep: false,
  obsBackend: "",
  useDiscoveredActions: true,
  siliconCortex: false,
};

const normalize = (value: LaunchConfig): LaunchConfig => {
  const workers = Math.max(1, Math.min(64, Number(value.workers || 1)));
  const envId = String(value.envId || "MegaBonk").trim() || "MegaBonk";
  const mode: LaunchMode = value.mode === "play" || value.mode === "dream" ? value.mode : "train";
  const obsBackend: ObsBackend =
    value.obsBackend === "detections" || value.obsBackend === "pixels" || value.obsBackend === "hybrid"
      ? value.obsBackend
      : "";
  return {
    mode,
    workers,
    envId,
    syntheticEye: Boolean(value.syntheticEye),
    syntheticEyeLockstep: Boolean(value.syntheticEyeLockstep),
    obsBackend,
    useDiscoveredActions: Boolean(value.useDiscoveredActions),
    siliconCortex: Boolean(value.siliconCortex),
  };
};

export default function useLaunchConfig() {
  const [cfg, setCfg] = useLocalStorageState<LaunchConfig>(STORAGE_KEY, defaults);

  const setNormalized = (next: LaunchConfig | ((prev: LaunchConfig) => LaunchConfig)) => {
    setCfg((prev) => {
      const base = normalize(prev);
      const resolved = typeof next === "function" ? (next as (p: LaunchConfig) => LaunchConfig)(base) : next;
      return normalize(resolved);
    });
  };

  return [normalize(cfg), setNormalized] as const;
}
