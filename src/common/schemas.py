"""Shared Pydantic models for MetaBonk.

These are inferred from last-known usage in worker/orchestrator code.
The original repository likely evolved these further; this recovery version
aims for compatibility and forwards-extensibility.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

try:
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pydantic is required for MetaBonk schemas; please install dependencies"
    ) from e


class MBBaseModel(BaseModel):
    """Pydantic v1/v2 compatibility shim."""

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        if hasattr(super(), "model_dump"):
            return super().model_dump(*args, **kwargs)  # type: ignore[attr-defined]
        return super().dict(*args, **kwargs)

    class Config:
        extra = "allow"


class TrainerConfig(MBBaseModel):
    lr: float = Field(3e-4, ge=0)
    gamma: float = Field(0.99, ge=0, le=1)
    entropy_coef: float = Field(0.01, ge=0)
    clip_range: float = Field(0.2, ge=0)
    gae_lambda: float = Field(0.95, ge=0, le=1)
    batch_size: int = Field(2048, ge=1)
    minibatch_size: int = Field(256, ge=1)
    epochs: int = Field(4, ge=1)
    use_lstm: Optional[bool] = True
    seq_len: Optional[int] = 32
    burn_in: Optional[int] = 8


class InstanceConfig(MBBaseModel):
    instance_id: str
    display: Optional[Union[str, int]] = None
    display_name: Optional[str] = None
    policy_name: str = "Greed"
    hparams: Dict[str, Any] = Field(default_factory=lambda: TrainerConfig().model_dump())
    # Stream/spectator controls (best-effort; worker may ignore if unsupported).
    capture_enabled: Optional[bool] = None
    featured_slot: Optional[str] = None  # "hype0"|"hype1"|"hype2"|"shame0"|"shame1"
    featured_role: Optional[str] = None  # "featured"|"warming"|"background"
    config_poll_s: Optional[float] = None
    # Evaluation mode (fixed-seed evals, no training updates).
    eval_mode: Optional[bool] = None
    eval_seed: Optional[int] = None


class Heartbeat(MBBaseModel):
    run_id: Optional[str] = None
    instance_id: str
    policy_name: Optional[str] = None
    policy_version: Optional[int] = None
    step: int = 0
    reward: Optional[float] = None
    steam_score: Optional[float] = None
    # Stream "hype" (0..100) used to rank which feeds to show on the Stream HUD.
    hype_score: Optional[float] = None
    hype_label: Optional[str] = None
    # Stream "shame" (0..100) used to pick a rotating "most shamed" feed.
    shame_score: Optional[float] = None
    shame_label: Optional[str] = None
    # Real-time "Choke Meter" (estimated P(survive next horizon)).
    survival_prob: Optional[float] = None
    # Convenience alias (0-1), where 1 is maximum danger (1 - survival_prob).
    danger_level: Optional[float] = None
    status: str = "running"
    # Optional streaming metadata (UI/OBS).
    stream_url: Optional[str] = None
    stream_type: Optional[str] = None  # "mp4" | "mpegts"
    # Whether the worker believes the stream is currently producing frames.
    stream_ok: Optional[bool] = None
    # Last time a frame was observed (unix seconds).
    stream_last_frame_ts: Optional[float] = None
    # Optional stream debug metadata.
    stream_error: Optional[str] = None
    streamer_last_error: Optional[str] = None
    stream_backend: Optional[str] = None
    # Optional FIFO/go2rtc distribution metadata (Metabonk).
    fifo_stream_enabled: Optional[bool] = None
    fifo_stream_path: Optional[str] = None
    fifo_stream_last_error: Optional[str] = None
    go2rtc_stream_name: Optional[str] = None
    go2rtc_base_url: Optional[str] = None
    pipewire_node_ok: Optional[bool] = None
    # Optional control base URL (for clip triggers, etc.)
    control_url: Optional[str] = None
    # Spectator selection (set by orchestrator).
    featured_slot: Optional[str] = None
    featured_role: Optional[str] = None
    # Fun stats (real-data driven via events, surfaced for UI convenience).
    luck_mult: Optional[float] = None
    luck_label: Optional[str] = None
    luck_drop_count: Optional[int] = None
    luck_legendary_count: Optional[int] = None
    borgar_count: Optional[int] = None
    borgar_label: Optional[str] = None
    # Swarm / DPS pressure (best-effort, requires plugin telemetry).
    enemy_count: Optional[int] = None
    incoming_dps: Optional[float] = None
    clearing_dps: Optional[float] = None
    dps_pressure: Optional[float] = None  # incoming / max(clearing, eps)
    overrun: Optional[bool] = None
    # Build / inventory (visual-model derived; do not require memory access).
    inventory_items: Optional[list[dict[str, Any]]] = None
    synergy_edges: Optional[list[dict[str, Any]]] = None
    evolution_recipes: Optional[list[dict[str, Any]]] = None
    # Sponsor overlay (community agency).
    display_name: Optional[str] = None
    sponsor_user: Optional[str] = None
    sponsor_user_id: Optional[str] = None
    sponsor_avatar_url: Optional[str] = None
    # Device hints (best-effort; do not assume CUDA is available).
    worker_device: Optional[str] = None
    vision_device: Optional[str] = None
    learned_reward_device: Optional[str] = None
    reward_device: Optional[str] = None
    ts: float = Field(default_factory=time.time)
    # Episode context (worker-local timer; does not require game memory access).
    episode_idx: Optional[int] = None
    episode_t: Optional[float] = None  # seconds since episode start


# Vision/Learner auxiliary messages (minimal)


class PredictRequest(MBBaseModel):
    """Vision predict request; carries either raw bytes or a shm handle."""

    image_b64: Optional[str] = None
    shm_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class PredictResponse(MBBaseModel):
    detections: Any = None
    latency_ms: Optional[float] = None
    # Optional structured metrics derived from the visual model (no game memory access).
    metrics: Optional[Dict[str, Any]] = None


class WeightsResponse(MBBaseModel):
    policy_name: str
    weights_b64: Optional[str] = None
    version: int = 0


class GradientsRequest(MBBaseModel):
    instance_id: str
    policy_name: str
    gradients_b64: Optional[str] = None
    step: int = 0


class RegisterWorkerRequest(MBBaseModel):
    instance_id: str
    policy_name: str
    capabilities: Dict[str, Any] = Field(default_factory=dict)


class RolloutBatch(MBBaseModel):
    """On-policy rollout batch sent from workers to learner."""

    instance_id: str
    policy_name: str
    # Optional hyperparameters snapshot (for PBT).
    hparams: Optional[Dict[str, Any]] = None
    obs: list[list[float]]
    actions_cont: list[list[float]]
    actions_disc: list[list[int]]
    # Optional per-step invalid-action masks for discrete heads.
    # Shape: [T, discrete_dim] for first discrete branch, where 1=valid, 0=invalid.
    action_masks: Optional[list[list[int]]] = None
    rewards: list[float]
    dones: list[bool]
    log_probs: Optional[list[float]] = None
    values: Optional[list[float]] = None
    # Sequence metadata for variable-length rollouts (optional).
    seq_lens: Optional[list[int]] = None
    truncated: Optional[bool] = None
    # Episode summaries (completed episodes only).
    episode_returns: Optional[list[float]] = None
    episode_lengths: Optional[list[int]] = None
    # Evaluation batches are excluded from training updates.
    eval_mode: Optional[bool] = None
    eval_seed: Optional[int] = None
    eval_clip_url: Optional[str] = None


class DemoBatch(MBBaseModel):
    """Supervised demo batch (e.g., 'watch me play' imitation updates).

    This is intentionally lightweight and storage-minimal: send a small batch of
    (obs, target_action) pairs, apply an online update, then discard.
    """

    policy_name: str
    obs: list[list[float]]
    actions_cont: list[list[float]]
    actions_disc: list[list[int]]
    action_masks: Optional[list[list[int]]] = None


class VisualRolloutBatch(MBBaseModel):
    """Visual-only rollout batch for world model learning.

    Used when the system is *watching* gameplay and learning dynamics/reward from visuals
    without recording the player's inputs.
    """

    policy_name: str
    obs: list[list[float]]
    rewards: Optional[list[float]] = None
    dones: Optional[list[bool]] = None


class CurriculumConfig(MBBaseModel):
    """Curriculum and reward weights."""

    phase: str = "foundation"  # survival, obstacle, physics, grandmaster
    survival_tick: float = 0.001
    xp_reward: float = 0.1
    levelup_reward: float = 1.0
    boss_reward: float = 5.0
    velocity_weight: float = 0.0001
    damage_penalty: float = -1.0
    wall_penalty: float = -0.05


# ---------------------------------------------------------------------------
# Omega / ACE (Agentic Context Engineering)
# ---------------------------------------------------------------------------


class ACEEpisodeReport(MBBaseModel):
    """Post-episode summary for ACE reflection/curation."""

    summary: str
    expected_reward: float
    actual_reward: float
    run_id: Optional[str] = None
    instance_id: Optional[str] = None


class ACERevertRequest(MBBaseModel):
    """Request to revert ACE strategy to a prior version."""

    version_id: Optional[str] = None


class ACEContextResponse(MBBaseModel):
    """Current ACE system prompt + metadata."""

    system_prompt: str
    current_strategy_version: Optional[str] = None
    total_episodes: int = 0
