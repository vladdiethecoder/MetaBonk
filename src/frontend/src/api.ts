export type Heartbeat = {
  run_id?: string | null;
  instance_id: string;
  policy_name?: string | null;
  policy_version?: number | null;
  step: number;
  reward?: number | null;
  steam_score?: number | null;
  hype_score?: number | null;
  hype_label?: string | null;
  shame_score?: number | null;
  shame_label?: string | null;
  survival_prob?: number | null;
  danger_level?: number | null;
  episode_idx?: number | null;
  episode_t?: number | null;
  luck_mult?: number | null;
  luck_label?: string | null;
  luck_drop_count?: number | null;
  luck_legendary_count?: number | null;
  borgar_count?: number | null;
  borgar_label?: string | null;
  enemy_count?: number | null;
  incoming_dps?: number | null;
  clearing_dps?: number | null;
  dps_pressure?: number | null;
  overrun?: boolean | null;
  inventory_items?: any[] | null;
  synergy_edges?: any[] | null;
  evolution_recipes?: any[] | null;
  status: string;
  stream_url?: string | null;
  stream_type?: string | null;
  stream_ok?: boolean | null;
  stream_last_frame_ts?: number | null;
  stream_error?: string | null;
  streamer_last_error?: string | null;
  stream_backend?: string | null;
  fifo_stream_enabled?: boolean | null;
  fifo_stream_path?: string | null;
  fifo_stream_last_error?: string | null;
  go2rtc_stream_name?: string | null;
  go2rtc_base_url?: string | null;
  pipewire_node_ok?: boolean | null;
  worker_device?: string | null;
  vision_device?: string | null;
  learned_reward_device?: string | null;
  reward_device?: string | null;
  control_url?: string | null;
  featured_slot?: string | null;
  featured_role?: string | null;
  display_name?: string | null;
  sponsor_user?: string | null;
  sponsor_user_id?: string | null;
  sponsor_avatar_url?: string | null;
  ts: number;
};

export type FeaturedResponse = {
  ts: number;
  slots: Record<string, string | null>;
  featured_ids: string[];
};

export type HofEntry = {
  ts: number;
  run_id?: string | null;
  instance_id: string;
  episode_idx?: number | null;
  duration_s: number;
  score: number;
};

export type Run = {
  run_id: string;
  experiment_id: string;
  status: string;
  best_reward: number;
  last_reward: number;
  last_step: number;
  created_ts: number;
  updated_ts: number;
  config?: Record<string, unknown>;
};

export type InstanceView = {
  heartbeat: Heartbeat;
  config?: {
    instance_id: string;
    display?: string | null;
    policy_name?: string | null;
    hparams?: Record<string, unknown>;
    eval_mode?: boolean | null;
    eval_seed?: number | null;
  } | null;
};

export type Event = {
  event_id: string;
  run_id?: string | null;
  instance_id?: string | null;
  event_type: string;
  message: string;
  ts: number;
  payload?: Record<string, unknown>;
};

export type BettingRoundState = {
  active: boolean;
  round_id?: number;
  question?: string;
  status?: string;
  total_pool?: number;
  bets_count?: number;
  options?: Array<{ name: string; pool: number; odds: string }>;
  believer_ratio?: { ratio: number; believers: number; doubters: number; color: string };
};

export type Whale = {
  rank: number;
  username: string;
  balance: number;
  win_rate: number;
  all_time_high: number;
};

export type BettingState = {
  round: BettingRoundState;
  whales: Whale[];
};

export type PollState = {
  active: boolean;
  poll_id?: string;
  question?: string;
  options?: string[];
  votes?: number[];
  starts_ts?: number;
  ends_ts?: number;
};

export type TimelinePin = {
  pin_id: string;
  ts: number;
  instance_id?: string | null;
  run_id?: string | null;
  episode_idx?: number | null;
  episode_t?: number | null;
  title: string;
  note?: string;
  created_by?: string | null;
  kind?: string;
};

export type TimelineBounty = {
  bounty_id: string;
  created_ts: number;
  title: string;
  kind: string;
  threshold: number;
  pot_total: number;
  active: boolean;
  claimed_ts?: number | null;
  claimed_by_instance?: string | null;
  payouts?: Array<{ user_id: string; amount: number }>;
};

export type TimelineState = {
  pins: TimelinePin[];
  bounties: TimelineBounty[];
  agent_names: Record<string, string>;
  run_names: Record<string, string>;
};

export type FuseSegment = {
  ts: number;
  run_id?: string | null;
  instance_id: string;
  duration_s: number;
  score: number;
  result?: string | null;
  stage?: number | null;
  biome?: string | null;
};

export type EraMarker = {
  ts: number;
  label: string;
  color?: string | null;
};

export type TimelineFuse = {
  now_ts: number;
  world_record_duration_s: number;
  genesis_ts?: number | null;
  total_events?: number | null;
  total_runs?: number | null;
  segments: FuseSegment[]; // newest-first
  eras: EraMarker[];
};

export type AttractClip = {
  ts: number;
  run_id?: string | null;
  instance_id: string;
  duration_s: number;
  score: number;
  clip_url: string;
};

export type AttractHighlights = {
  fame: AttractClip | null;
  shame: AttractClip | null;
  timestamp: number;
};

export type HistoricLeaderboardEntry = {
  instance_id: string;
  display_name?: string | null;
  policy_name?: string | null;
  last_ts?: number | null;
  last_score?: number | null;
  last_step?: number | null;
  best_score?: number | null;
  best_score_ts?: number | null;
  best_step?: number | null;
  best_step_ts?: number | null;
};

export type SkillTokenStat = {
  token: number;
  count: number;
  count_pct: number;
  avg_reward: number;
  usage?: number | null;
  name?: string | null;
  subtitle?: string | null;
  tags?: string[] | null;
};

export type SkillsSummary = {
  timestamp: number;
  source: {
    skill_ckpt: string;
    labeled_npz_dir: string;
  };
  skill_vqvae: {
    available: boolean;
    num_codes?: number;
    seq_len?: number;
    action_dim?: number;
    codebook_utilization?: number;
    usage_entropy?: number;
    error?: string;
  };
  dataset: {
    available: boolean;
    labeled_files?: number;
    total_steps?: number;
    token_top?: SkillTokenStat[];
    token_count_full?: Array<number | null> | null;
    token_avg_reward_full?: Array<number | null> | null;
    action_mean?: number[] | null;
    action_std?: number[] | null;
    error?: string;
  };
};

export type SkillTokenDetail = {
  token: number;
  decoded_action_seq: number[][];
  stat?: SkillTokenStat | null;
  name?: string | null;
  subtitle?: string | null;
  tags?: string[] | null;
  timestamp: number;
};

export type SkillFileInfo = {
  file: string;
  steps?: number | null;
  mtime: number;
};

export type SkillsFilesResponse = {
  timestamp: number;
  files: SkillFileInfo[];
};

export type SkillAtlasPoint = {
  token: number;
  x: number;
  y: number;
  usage?: number | null;
  count?: number | null;
  avg_reward?: number | null;
  name?: string | null;
  subtitle?: string | null;
  tags?: string[] | null;
};

export type SkillsAtlasResponse = {
  timestamp: number;
  method: string;
  points: SkillAtlasPoint[];
};

export type SkillTimelineResponse = {
  timestamp: number;
  file: string;
  step0: number;
  step1: number;
  stride: number;
  total_steps: number;
  tokens: number[];
  rewards?: number[] | null;
  top_tokens: number[];
};

export type SkillEffectToken = {
  token: number;
  count: number;
  pct: number;
  reward_mean?: number;
  reward_next_sum_mean?: number;
  action_mean?: number[];
  action_std?: number[];
  dominant_dims?: number[];
};

export type SkillEffectsResponse = {
  timestamp: number;
  file: string;
  step0: number;
  step1: number;
  total_steps: number;
  horizon: number;
  tokens: SkillEffectToken[];
};

export type SkillPrototype = {
  file: string;
  index: number;
  reward?: number;
  mime?: string;
  w?: number;
  h?: number;
  b64?: string;
};

export type SkillPrototypesResponse = {
  timestamp: number;
  token: number;
  prototypes: SkillPrototype[];
};

export type PretrainArtifact = {
  path: string;
  exists: boolean;
  meta?: Record<string, unknown>;
};

export type PretrainStatus = {
  timestamp: number;
  datasets: {
    video_demos_dir: string;
    video_demos_npz: number | null;
    video_labeled_dir: string;
    video_labeled_npz: number | null;
    video_rollouts_pt_dir: string;
    video_rollouts_pt: number | null;
  };
  artifacts: {
    idm_ckpt: PretrainArtifact;
    reward_ckpt: PretrainArtifact;
    skill_ckpt: PretrainArtifact;
    world_model_ckpt: PretrainArtifact;
    dream_policy_ckpt: PretrainArtifact;
    skill_names: { path: string; exists: boolean };
  };
};

export type OrchestratorStatus = {
  workers: number;
  policies: string[];
  timestamp: number;
};

export type EvalScore = {
  mean_return?: number;
  mean_length?: number;
  eval_seed?: number | null;
  episodes?: number;
  last_eval_ts?: number;
  window?: number;
};

export type PolicyRecord = {
  policy_name?: string | null;
  policy_version?: number | null;
  steam_score?: number | null;
  eval_score?: number | null;
  score_source?: string | null;
  last_update_ts?: number | null;
  last_eval_ts?: number | null;
  last_mutation_ts?: number | null;
  generation?: number | null;
  pbt_muted?: boolean | null;
  active_instances?: string[];
  assigned_instances?: string[];
  eval?: EvalScore | null;
};

export type PoliciesResponse = Record<string, PolicyRecord>;

export type PbtMuteState = {
  muted: boolean;
  policies: Record<string, boolean>;
};

const ORCH = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    throw new Error(`${r.status} ${r.statusText}`);
  }
  return (await r.json()) as T;
}

export async function fetchStatus(): Promise<OrchestratorStatus> {
  return fetchJson<OrchestratorStatus>(`${ORCH}/status`);
}

export async function fetchWorkers(): Promise<Record<string, Heartbeat>> {
  return fetchJson(`${ORCH}/workers`);
}

export async function fetchPolicies(): Promise<PoliciesResponse> {
  return fetchJson(`${ORCH}/policies`);
}

export async function fetchPbtMute(): Promise<PbtMuteState> {
  return fetchJson(`${ORCH}/pbt/mute`);
}

export async function setPbtMute(payload: { muted: boolean; policy_name?: string }): Promise<PbtMuteState> {
  return fetchJson(`${ORCH}/pbt/mute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchFeatured(): Promise<FeaturedResponse> {
  return fetchJson(`${ORCH}/featured`);
}

export async function fetchRuns(): Promise<Run[]> {
  return fetchJson(`${ORCH}/runs`);
}

export async function fetchInstances() {
  return fetchJson<Record<string, InstanceView>>(`${ORCH}/instances`);
}

export async function fetchEvents(limit = 200): Promise<Event[]> {
  return fetchJson(`${ORCH}/events?limit=${limit}`);
}

export async function fetchSkillsSummary(): Promise<SkillsSummary> {
  return fetchJson(`${ORCH}/skills/summary`);
}

export async function fetchSkillToken(tokenId: number): Promise<SkillTokenDetail> {
  return fetchJson(`${ORCH}/skills/token/${tokenId}`);
}

export async function fetchSkillsFiles(limit = 40): Promise<SkillsFilesResponse> {
  return fetchJson(`${ORCH}/skills/files?limit=${encodeURIComponent(limit)}`);
}

export async function fetchSkillsAtlas(): Promise<SkillsAtlasResponse> {
  return fetchJson(`${ORCH}/skills/atlas`);
}

export async function fetchSkillTimeline(params: { file?: string | null; window?: number; stride?: number; topk?: number; start?: number | null }): Promise<SkillTimelineResponse> {
  const sp = new URLSearchParams();
  if (params.file) sp.set("file", params.file);
  if (params.window != null) sp.set("window", String(params.window));
  if (params.stride != null) sp.set("stride", String(params.stride));
  if (params.topk != null) sp.set("topk", String(params.topk));
  if (params.start != null) sp.set("start", String(params.start));
  const qs = sp.toString();
  return fetchJson(`${ORCH}/skills/timeline${qs ? `?${qs}` : ""}`);
}

export async function fetchSkillEffects(params: { file?: string | null; tokens?: number[]; window?: number; horizon?: number; start?: number | null }): Promise<SkillEffectsResponse> {
  const sp = new URLSearchParams();
  if (params.file) sp.set("file", params.file);
  if (params.tokens?.length) sp.set("tokens", params.tokens.join(","));
  if (params.window != null) sp.set("window", String(params.window));
  if (params.horizon != null) sp.set("horizon", String(params.horizon));
  if (params.start != null) sp.set("start", String(params.start));
  const qs = sp.toString();
  return fetchJson(`${ORCH}/skills/effects${qs ? `?${qs}` : ""}`);
}

export async function fetchSkillPrototypes(tokenId: number, params: { limit?: number; size?: number } = {}): Promise<SkillPrototypesResponse> {
  const sp = new URLSearchParams();
  if (params.limit != null) sp.set("limit", String(params.limit));
  if (params.size != null) sp.set("size", String(params.size));
  const qs = sp.toString();
  return fetchJson(`${ORCH}/skills/token/${tokenId}/prototypes${qs ? `?${qs}` : ""}`);
}

export async function generateSkillNames(topk = 32, force = false): Promise<any> {
  return fetchJson(`${ORCH}/skills/names/generate?topk=${topk}&force=${force ? "1" : "0"}`, { method: "POST" });
}

export async function fetchPretrainStatus(): Promise<PretrainStatus> {
  return fetchJson(`${ORCH}/pretrain/status`);
}

export type PretrainJob = {
  job_id: string;
  kind: string;
  status: "running" | "succeeded" | "failed";
  started_ts: number;
  ended_ts?: number | null;
  returncode?: number | null;
  cmd?: string[] | null;
  log?: string[];
  artifacts?: Record<string, string>;
};

export async function fetchPretrainJobs(): Promise<PretrainJob[]> {
  return fetchJson(`${ORCH}/pretrain/jobs`);
}

export async function buildDreamPolicy(params?: {
  steps?: number;
  batch_obs?: number;
  horizon?: number;
  starts?: number;
  device?: string;
  pt_dir?: string;
  world_model_ckpt?: string;
  out_ckpt?: string;
}): Promise<{ ok: boolean; job_id?: string; dream_policy_ckpt?: string; already_present?: boolean; path?: string }> {
  const qs = new URLSearchParams();
  const p = params ?? {};
  for (const [k, v] of Object.entries(p)) {
    if (v === null || v === undefined) continue;
    qs.set(k, String(v));
  }
  const suf = qs.toString() ? `?${qs.toString()}` : "";
  return fetchJson(`${ORCH}/pretrain/dream/build${suf}`, { method: "POST" });
}

export async function buildWorldModel(params?: {
  epochs?: number;
  device?: string;
  pt_dir?: string;
  out_ckpt?: string;
  max_episodes?: number;
}): Promise<{ ok: boolean; job_id?: string; world_model_ckpt?: string }> {
  const qs = new URLSearchParams();
  const p = params ?? {};
  for (const [k, v] of Object.entries(p)) {
    if (v === null || v === undefined) continue;
    qs.set(k, String(v));
  }
  const suf = qs.toString() ? `?${qs.toString()}` : "";
  return fetchJson(`${ORCH}/pretrain/world_model/build${suf}`, { method: "POST" });
}

export async function fetchBettingState(): Promise<BettingState> {
  return fetchJson(`${ORCH}/betting/state`);
}

export async function fetchPollState(): Promise<PollState> {
  return fetchJson(`${ORCH}/poll/state`);
}

export async function fetchHofTop(limit = 10, sort: "score" | "time" = "score"): Promise<HofEntry[]> {
  return fetchJson(`${ORCH}/hof/top?limit=${encodeURIComponent(limit)}&sort=${encodeURIComponent(sort)}`);
}

export async function fetchTimeline(limitPins = 200): Promise<TimelineState> {
  return fetchJson(`${ORCH}/timeline?limit_pins=${encodeURIComponent(limitPins)}`);
}

export async function fetchTimelineFuse(limit = 5000): Promise<TimelineFuse> {
  return fetchJson(`${ORCH}/timeline/fuse?limit=${encodeURIComponent(limit)}`);
}

export async function fetchAttractHighlights(): Promise<AttractHighlights> {
  return fetchJson(`${ORCH}/attract/highlights`);
}

export async function fetchHistoricLeaderboard(limit = 20, sort: "best_score" | "best_step" | "recent" = "best_score"): Promise<HistoricLeaderboardEntry[]> {
  return fetchJson(`${ORCH}/leaderboard/historic?limit=${encodeURIComponent(limit)}&sort=${encodeURIComponent(sort)}`);
}
