"""Central Learner service (PPO + SinZero intrinsic rewards).

Workers submit on‑policy rollouts; this service updates per‑policy PPO
learners and serves weights.

SinZero integration:
  - Policy names matching a Sin receive default bias knobs.
  - Lust gets Random Network Distillation (RND) intrinsic reward.
  - Other sins currently use lightweight heuristics (see `_intrinsic_reward`).

This keeps the distributed recovery system runnable while leaving room
for research‑grade upgrades (OPPO, alpha‑divergence, CVaR, MoE).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, HTTPException, Response
import uvicorn

import torch
from pydantic import BaseModel, Field

from src.common.device import resolve_device
from src.common.schemas import (
    RegisterWorkerRequest,
    RolloutBatch,
    WeightsResponse,
    DemoBatch,
    VisualRolloutBatch,
)
from src.common.sins import DEFAULT_SIN_BIASES, Sin
from .ppo import PPOConfig, PolicyLearner, to_tensor
from .rnd import RNDConfig, RNDModule
from .world_model import WorldModel, WorldModelConfig
from .task_vectors import TaskVector, TaskVectorMetadata, TIESMerger
try:
    from .consistency_policy import ConsistencyPolicy, CPQEConfig
except Exception:  # pragma: no cover
    ConsistencyPolicy = None  # type: ignore
    CPQEConfig = None  # type: ignore
try:
    from .diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig
except Exception:  # pragma: no cover
    DiffusionPolicy = None  # type: ignore
    DiffusionPolicyConfig = None  # type: ignore
try:
    from .free_energy import ActiveInferenceAgent
except Exception:  # pragma: no cover
    ActiveInferenceAgent = None  # type: ignore


app = FastAPI(title="MetaBonk Learner Service")

# Per‑policy learners and intrinsic modules.
_learners: Dict[str, PolicyLearner] = {}
_rnd: Dict[str, RNDModule] = {}
_world_models: Dict[str, WorldModel] = {}
_cpqe: Dict[str, Any] = {}
_diffusion_teachers: Dict[str, Any] = {}
_active_inf: Dict[str, Any] = {}

# Latest training metrics snapshot per policy (for RCC / observability).
_last_metrics: Dict[str, Dict[str, Any]] = {}
_last_push_ts: Optional[float] = None
_tps_ema: float = 0.0
_merge_lock = threading.Lock()
_policy_obs_dim: Dict[str, int] = {}
_ppo_ckpt_lock = threading.Lock()
_ppo_last_save_ts: Dict[str, float] = {}
_demo_lock = threading.Lock()
_eval_lock = threading.Lock()


def _eval_history_path() -> Path:
    return Path(os.environ.get("METABONK_EVAL_HISTORY_PATH", "checkpoints/eval_history.json"))


def _eval_best_path() -> Path:
    return Path(os.environ.get("METABONK_EVAL_BEST_PATH", "checkpoints/eval_best.json"))


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return []
    return []


def _record_eval_metrics(metrics: Dict[str, Any]) -> None:
    path = _eval_history_path()
    best_path = _eval_best_path()
    with _eval_lock:
        history = _load_json_list(path)
        history.append(metrics)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history[-500:], indent=2, sort_keys=True))
        except Exception:
            pass

        try:
            best = {}
            if best_path.exists():
                best = json.loads(best_path.read_text())
        except Exception:
            best = {}
        policy = str(metrics.get("policy_name") or "policy")
        score = float(metrics.get("mean_return", 0.0))
        prev = best.get(policy, {})
        prev_score = float(prev.get("mean_return", -1e9)) if isinstance(prev, dict) else -1e9
        if score >= prev_score:
            best[policy] = metrics
            try:
                best_path.parent.mkdir(parents=True, exist_ok=True)
                best_path.write_text(json.dumps(best, indent=2, sort_keys=True))
            except Exception:
                pass


def _eval_metrics_from_batch(batch: RolloutBatch) -> Dict[str, Any]:
    rewards = [float(r) for r in (batch.rewards or [])]
    dones = [bool(d) for d in (batch.dones or [])]
    ep_returns = [float(r) for r in (batch.episode_returns or [])]
    ep_lengths = [int(l) for l in (batch.episode_lengths or [])]
    if not ep_returns:
        cur_ret = 0.0
        cur_len = 0
        for r, d in zip(rewards, dones):
            cur_ret += r
            cur_len += 1
            if d:
                ep_returns.append(cur_ret)
                ep_lengths.append(cur_len)
                cur_ret = 0.0
                cur_len = 0
        if not ep_returns and cur_len > 0:
            ep_returns.append(cur_ret)
            ep_lengths.append(cur_len)
    episodes = len(ep_returns)
    mean_return = sum(ep_returns) / max(1, episodes)
    mean_length = sum(ep_lengths) / max(1, episodes)
    return {
        "policy_name": batch.policy_name,
        "instance_id": batch.instance_id,
        "ts": time.time(),
        "episodes": episodes,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "steps": len(rewards),
        "eval_seed": batch.eval_seed,
        "eval_clip_url": batch.eval_clip_url,
        "truncated": batch.truncated,
    }


def _ppo_ckpt_dir() -> Path:
    return Path(os.environ.get("METABONK_PPO_CKPT_DIR", "checkpoints/ppo"))


def _ppo_ckpt_path(policy_name: str) -> Path:
    safe = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in (policy_name or "policy")])
    return _ppo_ckpt_dir() / f"{safe}.pt"


def _load_state_dict_from_any(obj: Any) -> Optional[dict]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        # Direct state_dict?
        if obj and all(isinstance(k, str) for k in obj.keys()):
            if any(".weight" in k or ".bias" in k or k in ("log_std", "action_log_std") for k in obj.keys()):
                return obj
        for k in ("policy", "model_state_dict", "policy_state_dict", "state_dict"):
            v = obj.get(k)
            if isinstance(v, dict):
                return v
    return None


def _translate_singularity_keys(sd: dict) -> dict:
    """Translate legacy 'singularity' actor keys into ActorCritic key names (best-effort)."""
    out: dict = {}
    for k, v in sd.items():
        nk = k
        if nk == "action_log_std":
            nk = "log_std"
        nk = nk.replace("network.0.", "shared.0.")
        nk = nk.replace("network.2.", "shared.2.")
        nk = nk.replace("action_mean.", "mu.")
        out[nk] = v
    return out


def _filter_state_dict_for_model(model: torch.nn.Module, sd: dict) -> dict:
    """Keep only keys present in model with exactly matching tensor shapes."""
    try:
        msd = model.state_dict()
    except Exception:
        return sd
    out: dict = {}
    for k, v in sd.items():
        if k not in msd:
            continue
        try:
            if hasattr(v, "shape") and hasattr(msd[k], "shape") and tuple(v.shape) != tuple(msd[k].shape):
                continue
        except Exception:
            continue
        out[k] = v
    return out


def _try_load_ppo_weights(learner: PolicyLearner, policy_name: str, obs_dim: int) -> bool:
    """Load policy weights from disk (resume) or from a pretrain init ckpt."""
    device = learner.device
    # 1) Resume from per-policy PPO ckpt.
    ckpt_path = _ppo_ckpt_path(policy_name)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = _load_state_dict_from_any(ckpt)
            if isinstance(sd, dict):
                sd = _filter_state_dict_for_model(learner.net, sd)
                learner.net.load_state_dict(sd, strict=False)
                return True
        except Exception:
            pass

    # 2) Fallback: shared initialization ckpt (e.g., "singularity").
    init_path = Path(os.environ.get("METABONK_POLICY_INIT_CKPT", "checkpoints/singularity.pt"))
    if not init_path.exists():
        return False
    try:
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        sd = _load_state_dict_from_any(ckpt)
        if not isinstance(sd, dict):
            return False
        # Quick sanity check on input dim when available.
        try:
            w0 = sd.get("network.0.weight") or sd.get("shared.0.weight")
            if isinstance(w0, torch.Tensor) and int(w0.shape[-1]) != int(obs_dim):
                return False
        except Exception:
            pass
        sd = _translate_singularity_keys(sd)
        sd = _filter_state_dict_for_model(learner.net, sd)
        learner.net.load_state_dict(sd, strict=False)
        return True
    except Exception:
        return False


def _maybe_save_ppo_weights(learner: PolicyLearner, policy_name: str, obs_dim: int) -> None:
    ttl = float(os.environ.get("METABONK_PPO_SAVE_TTL_S", "10.0"))
    now = time.time()
    last = _ppo_last_save_ts.get(policy_name, 0.0)
    if now - last < ttl:
        return
    path = _ppo_ckpt_path(policy_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    payload = {
        "policy_name": policy_name,
        "obs_dim": int(obs_dim),
        "version": int(getattr(learner, "version", 0)),
        "saved_ts": float(now),
        "state_dict": learner.net.state_dict(),
    }
    with _ppo_ckpt_lock:
        try:
            torch.save(payload, tmp)
            tmp.replace(path)
            _ppo_last_save_ts[policy_name] = now
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


def _iter_pt_episodes(data_dir: str, limit: int = 0):
    """Yield rollout dicts saved as torch .pt files.

    Expected keys: observations [T,obs_dim], actions [T,action_dim], rewards [T], dones [T] (optional)
    """
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"pt_dir not found: {data_dir}")
    n = 0
    for f in sorted(p.glob("*.pt")):
        try:
            ep = torch.load(f, weights_only=False)
        except Exception:
            continue
        if not isinstance(ep, dict):
            continue
        if "observations" not in ep or "actions" not in ep:
            continue
        yield f, ep
        n += 1
        if limit and n >= limit:
            break


def _episode_to_tensors(ep: Dict[str, Any], device: torch.device):
    obs = ep.get("observations")
    actions = ep.get("actions")
    rewards = ep.get("rewards")
    dones = ep.get("dones")
    if obs is None or actions is None:
        raise ValueError("episode missing observations/actions")
    obs_t = obs.to(device) if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, device=device, dtype=torch.float32)
    act_t = actions.to(device) if isinstance(actions, torch.Tensor) else torch.as_tensor(actions, device=device, dtype=torch.float32)
    if rewards is None:
        rewards_t = torch.zeros(obs_t.shape[0], device=device, dtype=torch.float32)
    else:
        rewards_t = rewards.to(device) if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, device=device, dtype=torch.float32)
    dones_t = None
    if dones is not None:
        dones_t = dones.to(device)
        if dones_t.dtype != torch.bool:
            dones_t = dones_t.to(dtype=torch.bool)
    return obs_t, act_t, rewards_t, dones_t


class OfflineWorldModelTrainRequest(BaseModel):
    policy_name: str = Field(default="Greed")
    pt_dir: str = Field(default="rollouts/video_rollouts")
    epochs: int = Field(default=5, ge=1)
    max_episodes: int = Field(default=0, ge=0)
    out_ckpt: Optional[str] = None


class OfflineDreamTrainRequest(BaseModel):
    policy_name: str = Field(default="Greed")
    pt_dir: str = Field(default="rollouts/video_rollouts")
    max_episodes: int = Field(default=0, ge=0)
    steps: int = Field(default=2000, ge=1)
    batch_obs: int = Field(default=256, ge=1)
    horizon: int = Field(default=5, ge=1)
    starts: int = Field(default=8, ge=1)
    gamma: Optional[float] = None
    # Optional world-model checkpoint to load before dreaming.
    world_model_ckpt: Optional[str] = None


def _sanitize_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metrics dict is JSON-serializable."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        try:
            if isinstance(v, torch.Tensor):
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = float(v) if isinstance(v, (int, float)) else v
        except Exception:
            out[k] = v
    return out


def _as_sin(name: str) -> Optional[Sin]:
    if not name:
        return None
    n = name.strip().lower()
    for s in Sin:
        if s.value.lower() == n:
            return s
    return None


def _get_or_create(policy_name: str, obs_dim: int) -> PolicyLearner:
    if policy_name in _learners:
        return _learners[policy_name]

    # Allow forcing learner device to avoid starving the actual game of VRAM (e.g., watch mode).
    want_dev = str(os.environ.get("METABONK_LEARNER_DEVICE") or os.environ.get("METABONK_DEVICE") or "").strip()
    want_dev = resolve_device(want_dev, context="learner")

    sin = _as_sin(policy_name)
    bias = DEFAULT_SIN_BIASES.get(sin) if sin else None
    cfg = PPOConfig(
        entropy_coef=bias.entropy_coef if bias else PPOConfig.entropy_coef,
    )
    try:
        from .ppo import apply_env_overrides

        cfg = apply_env_overrides(cfg)
    except Exception:
        pass
    if os.environ.get("METABONK_PPO_USE_LSTM", "0") in ("1", "true", "True"):
        cfg.use_lstm = True
    try:
        if os.environ.get("METABONK_PPO_SEQ_LEN"):
            cfg.seq_len = int(os.environ.get("METABONK_PPO_SEQ_LEN", str(cfg.seq_len)))
    except Exception:
        pass
    try:
        if os.environ.get("METABONK_PPO_BURN_IN"):
            cfg.burn_in = int(os.environ.get("METABONK_PPO_BURN_IN", str(cfg.burn_in)))
    except Exception:
        pass
    net_cls = None
    # LNN Pilot backend is mandatory for pilot policies.
    n = policy_name.lower()
    if "pilot" in n or n.endswith("_pilot"):
        try:
            from .liquid_policy import LiquidActorCritic

            net_cls = LiquidActorCritic
        except Exception:
            net_cls = None
    if net_cls is not None:
        learner = PolicyLearner(
            obs_dim=obs_dim,
            cfg=cfg,
            device=want_dev,
            net_cls=net_cls,
        )
    else:
        learner = PolicyLearner(
            obs_dim=obs_dim,
            cfg=cfg,
            device=want_dev,
        )
    _learners[policy_name] = learner

    # Best-effort warm-start from saved PPO weights or from a shared init ckpt.
    _try_load_ppo_weights(learner, policy_name=policy_name, obs_dim=obs_dim)

    if sin == Sin.LUST:
        _rnd[policy_name] = RNDModule(RNDConfig(obs_dim=obs_dim), device=str(learner.device))

    return learner


def _get_or_create_world_model(policy_name: str, obs_dim: int, action_dim: int, device: torch.device) -> WorldModel:
    if policy_name in _world_models:
        return _world_models[policy_name]
    cfg = WorldModelConfig(obs_dim=obs_dim, action_dim=action_dim)
    wm = WorldModel(cfg).to(device)
    _world_models[policy_name] = wm
    return wm


def _get_or_create_cpqe(
    policy_name: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> Optional[Any]:
    """Optionally create a CPQE (consistency) motor policy for pilot distillation."""
    if ConsistencyPolicy is None or CPQEConfig is None:
        return None
    if not os.environ.get("METABONK_TRAIN_CPQE", "1") in ("1", "true", "True"):
        return None
    n = (policy_name or "").lower()
    if "pilot" not in n and not n.endswith("_pilot"):
        return None
    if policy_name in _cpqe:
        return _cpqe[policy_name]
    try:
        horizon = int(os.environ.get("METABONK_CPQE_HORIZON", "8"))
        cfg = CPQEConfig(obs_dim=obs_dim, action_dim=action_dim, horizon=horizon)
        cpqe = ConsistencyPolicy(cfg).to(device)
        _cpqe[policy_name] = cpqe
        return cpqe
    except Exception:
        return None


def _get_or_create_diffusion_teacher(
    policy_name: str,
    obs_dim: int,
    action_dim: int,
    horizon: int,
    device: torch.device,
) -> Optional[Any]:
    """Optionally create a diffusion teacher for pilot distillation."""
    if DiffusionPolicy is None or DiffusionPolicyConfig is None:
        return None
    if not os.environ.get("METABONK_TRAIN_DIFFUSION_TEACHER", "1") in ("1", "true", "True"):
        return None
    n = (policy_name or "").lower()
    if "pilot" not in n and not n.endswith("_pilot"):
        return None
    if policy_name in _diffusion_teachers:
        return _diffusion_teachers[policy_name]
    try:
        cfg = DiffusionPolicyConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            obs_horizon=2,
            use_goal_conditioning=False,
        )
        teacher = DiffusionPolicy(cfg).to(device)
        _diffusion_teachers[policy_name] = teacher
        return teacher
    except Exception:
        return None


def _apply_hparams(learner: PolicyLearner, hparams: Dict[str, Any]) -> None:
    """Apply PBT/worker hparams to a learner in-place."""
    cfg = learner.cfg
    # Map common keys.
    if "lr" in hparams:
        cfg.learning_rate = float(hparams["lr"])
        for g in learner.opt.param_groups:
            g["lr"] = cfg.learning_rate
    if "gamma" in hparams:
        cfg.gamma = float(hparams["gamma"])
    if "entropy_coef" in hparams:
        cfg.entropy_coef = float(hparams["entropy_coef"])
    if "clip_range" in hparams:
        cfg.clip_range = float(hparams["clip_range"])
    if "gae_lambda" in hparams:
        cfg.gae_lambda = float(hparams["gae_lambda"])
    if "batch_size" in hparams:
        cfg.batch_size = int(hparams["batch_size"])
    if "minibatch_size" in hparams:
        cfg.minibatch_size = int(hparams["minibatch_size"])
    if "epochs" in hparams:
        cfg.num_epochs = int(hparams["epochs"])
    if "max_grad_norm" in hparams:
        cfg.max_grad_norm = float(hparams["max_grad_norm"])
    if "use_lstm" in hparams:
        cfg.use_lstm = bool(hparams["use_lstm"])
    if "seq_len" in hparams:
        cfg.seq_len = int(hparams["seq_len"])
    if "burn_in" in hparams:
        cfg.burn_in = int(hparams["burn_in"])


def _intrinsic_reward(
    policy_name: str,
    obs_t: torch.Tensor,
    actions_cont: torch.Tensor,
) -> torch.Tensor:
    """Compute per‑step intrinsic reward for a policy."""
    sin = _as_sin(policy_name)
    if sin is None:
        return torch.zeros(obs_t.shape[0], device=obs_t.device)

    if sin == Sin.LUST and policy_name in _rnd:
        return _rnd[policy_name].intrinsic_reward(obs_t)

    return torch.zeros(obs_t.shape[0], device=obs_t.device)


@app.get("/")
async def read_root():
    return {"message": "MetaBonk Learner Service is running"}


@app.post("/register_worker")
async def register_worker(req: RegisterWorkerRequest):
    # Workers can provide obs_dim in capabilities so we can warm-start immediately.
    try:
        od = req.capabilities.get("obs_dim") if isinstance(req.capabilities, dict) else None
        if od is not None:
            _policy_obs_dim[req.policy_name] = int(od)
            _get_or_create(req.policy_name, obs_dim=int(od))
    except Exception:
        pass
    return {"ok": True, "instance_id": req.instance_id, "policy_name": req.policy_name}


@app.get("/get_weights", response_model=WeightsResponse)
async def get_weights(policy_name: str, since_version: int = -1):
    if policy_name not in _learners:
        # Try to warm-start if we know the obs_dim from worker registration.
        od = _policy_obs_dim.get(policy_name)
        if od is None:
            try:
                od = int(os.environ.get("METABONK_DEFAULT_OBS_DIM", "204"))
            except Exception:
                od = None
        if od is None:
            return WeightsResponse(policy_name=policy_name, weights_b64=None, version=0)
        _get_or_create(policy_name, obs_dim=int(od))
    learner = _learners[policy_name]
    try:
        if int(since_version) >= 0 and int(learner.version) <= int(since_version):
            return Response(status_code=204)
    except Exception:
        pass
    return WeightsResponse(
        policy_name=policy_name,
        weights_b64=learner.get_weights_b64(),
        version=learner.version,
    )


@app.get("/get_cpqe_weights", response_model=WeightsResponse)
async def get_cpqe_weights(policy_name: str):
    """Serve CPQE motor weights if trained."""
    if policy_name not in _cpqe:
        return WeightsResponse(policy_name=policy_name, weights_b64=None, version=0)
    cpqe = _cpqe[policy_name]
    try:
        buf = io.BytesIO()
        torch.save(cpqe.state_dict(), buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return WeightsResponse(policy_name=policy_name, weights_b64=b64, version=0)
    except Exception:
        return WeightsResponse(policy_name=policy_name, weights_b64=None, version=0)


@app.post("/push_rollout")
async def push_rollout(batch: RolloutBatch):
    global _last_push_ts, _tps_ema
    if not batch.obs:
        raise HTTPException(status_code=400, detail="empty rollout")
    obs_dim = len(batch.obs[0])
    learner = _get_or_create(batch.policy_name, obs_dim=obs_dim)
    if batch.hparams:
        try:
            _apply_hparams(learner, batch.hparams)
        except Exception:
            pass

    if batch.eval_mode:
        metrics = _eval_metrics_from_batch(batch)
        _record_eval_metrics(metrics)
        _last_metrics[batch.policy_name] = {**_last_metrics.get(batch.policy_name, {}), **_sanitize_metrics(metrics)}
        return {"ok": True, "policy_name": batch.policy_name, "eval": metrics}

    obs_t = to_tensor(batch.obs)
    actions_cont_t = to_tensor(batch.actions_cont)
    actions_disc_t = to_tensor(batch.actions_disc, dtype=torch.long)
    rewards_t = to_tensor(batch.rewards)
    dones_t = to_tensor(batch.dones, dtype=torch.bool)

    action_masks_t = None
    if batch.action_masks:
        action_masks_t = to_tensor(batch.action_masks, dtype=torch.long)

    device = getattr(learner, "device", torch.device("cpu"))
    obs_t = obs_t.to(device)
    actions_cont_t = actions_cont_t.to(device)
    actions_disc_t = actions_disc_t.to(device)
    rewards_t = rewards_t.to(device)
    dones_t = dones_t.to(device)
    if action_masks_t is not None:
        action_masks_t = action_masks_t.to(device)

    actions_flat = torch.cat([actions_cont_t, actions_disc_t.float()], dim=-1)

    # Add intrinsic reward if applicable.
    intr = _intrinsic_reward(batch.policy_name, obs_t, actions_cont_t)
    coef = 0.0
    # Prefer curiosity_beta from hparams if present (matches YAML presets).
    if batch.hparams and isinstance(batch.hparams.get("reward_shaping"), dict):
        try:
            cb = batch.hparams["reward_shaping"].get("curiosity_beta")
            if cb is not None:
                coef = float(cb)
        except Exception:
            coef = 0.0
    if coef == 0.0:
        bias = DEFAULT_SIN_BIASES.get(_as_sin(batch.policy_name))
        coef = bias.intrinsic_coef if bias else 0.0
    rewards_t = rewards_t + coef * intr

    # Update RND online for Lust.
    if batch.policy_name in _rnd:
        try:
            _rnd[batch.policy_name].update(obs_t)
        except Exception:
            pass

    # --- Dreamer-lite auxiliary world model update ---
    try:
        act_dim = actions_flat.shape[-1]
        wm = _get_or_create_world_model(batch.policy_name, obs_dim=obs_dim, action_dim=act_dim, device=obs_t.device)
        wm_losses = wm.update_from_rollout(obs_t, actions_flat, rewards_t, dones=dones_t)
    except Exception:
        wm_losses = {}

    # --- Optional Active-Inference EFE policy update ---
    try:
        if os.environ.get("METABONK_USE_ACTIVE_INFERENCE", "1") in ("1", "true", "True"):
            wm_ai = _world_models.get(batch.policy_name)
            if wm_ai is not None and ActiveInferenceAgent is not None:
                ai = _active_inf.get(batch.policy_name)
                if ai is None or getattr(ai, "world_model", None) is not wm_ai:
                    ai = ActiveInferenceAgent(wm_ai).to(obs_t.device)
                    _active_inf[batch.policy_name] = ai
                efe_losses = ai.update_from_rollout(
                    observations=obs_t.detach(),
                    actions=actions_flat.detach(),
                    rewards=rewards_t.detach(),
                    dones=dones_t.detach(),
                )
            else:
                efe_losses = {}
        else:
            efe_losses = {}
    except Exception:
        efe_losses = {}

    # --- Optional CPQE distillation for pilot policies ---
    try:
        cpqe = _get_or_create_cpqe(
            batch.policy_name,
            obs_dim=obs_dim,
            action_dim=actions_cont_t.shape[-1],
            device=obs_t.device,
        )
        if cpqe is not None:
            horizon = int(getattr(cpqe.cfg, "horizon", 8))
            seq_obs = []
            seq_act = []
            # Build short windows from on-policy rollouts.
            N = obs_t.shape[0]
            for i in range(0, max(1, N - horizon)):
                a_win = actions_cont_t[i : i + horizon]
                if a_win.shape[0] < horizon:
                    pad = a_win[-1:].repeat(horizon - a_win.shape[0], 1)
                    a_win = torch.cat([a_win, pad], dim=0)
                a_win = a_win.unsqueeze(0)  # [1, T, action_dim]
                o_win = obs_t[i : i + 2]
                if o_win.shape[0] < 2:
                    o_win = torch.cat([o_win, o_win[-1:]], dim=0)
                o_win = o_win.unsqueeze(0)  # [1, 2, obs_dim]
                seq_obs.append(o_win)
                seq_act.append(a_win)
                if len(seq_obs) >= 64:
                    break
            if seq_obs and seq_act:
                obs_batch = torch.cat(seq_obs, dim=0)
                act_batch = torch.cat(seq_act, dim=0)
                # Prefer diffusion teacher -> CPQE consistency distillation when available.
                teacher_losses: Dict[str, float] = {}
                teacher = _get_or_create_diffusion_teacher(
                    batch.policy_name,
                    obs_dim=obs_dim,
                    action_dim=actions_cont_t.shape[-1],
                    horizon=horizon,
                    device=obs_t.device,
                )
                if teacher is not None:
                    try:
                        teacher_losses = teacher.update(act_batch, obs_batch)
                        # Teacher generates clean trajectories for distillation.
                        teacher_actions = teacher.sample(obs_batch, goal=None, use_ddim=True).clamp(-1.0, 1.0)
                        cpqe_losses = cpqe.update_policy(teacher_actions, obs_batch)
                        cpqe_losses.update({f"diffusion_{k}": float(v) for k, v in teacher_losses.items()})
                        cpqe_losses["cpqe_distill_source"] = "diffusion_teacher"
                    except Exception:
                        cpqe_losses = cpqe.update_policy(act_batch, obs_batch)
                        cpqe_losses.update({f"diffusion_{k}": float(v) for k, v in teacher_losses.items()})
                        cpqe_losses["cpqe_distill_source"] = "on_policy"
                else:
                    cpqe_losses = cpqe.update_policy(act_batch, obs_batch)
                    cpqe_losses["cpqe_distill_source"] = "on_policy"

                # Optional Q-ensemble TD updates from on-policy windows.
                try:
                    q_updates = int(os.environ.get("METABONK_CPQE_Q_UPDATES", "1"))
                    q_gamma = float(os.environ.get("METABONK_CPQE_Q_GAMMA", "0.99"))
                    if q_updates > 0:
                        q_losses = []
                        for _ in range(q_updates):
                            q_obs = []
                            q_next = []
                            q_rewards = []
                            q_dones = []
                            N = obs_t.shape[0]
                            for i in range(0, max(1, N - 2)):
                                o_win = obs_t[i : i + 2]
                                if o_win.shape[0] < 2:
                                    o_win = torch.cat([o_win, o_win[-1:]], dim=0)
                                o_win = o_win.unsqueeze(0)
                                o_next = obs_t[i + 1 : i + 3]
                                if o_next.shape[0] < 2:
                                    o_next = torch.cat([o_next, o_next[-1:]], dim=0)
                                o_next = o_next.unsqueeze(0)
                                r = rewards_t[i : i + 1]
                                d = dones_t[i : i + 1]
                                q_obs.append(o_win)
                                q_next.append(o_next)
                                q_rewards.append(r)
                                q_dones.append(d)
                                if len(q_obs) >= 64:
                                    break
                            if q_obs:
                                q_obs_b = torch.cat(q_obs, dim=0)
                                q_next_b = torch.cat(q_next, dim=0)
                                q_rewards_b = torch.cat(q_rewards, dim=0).view(-1)
                                q_dones_b = torch.cat(q_dones, dim=0).view(-1)
                                q_actions_b = act_batch[: q_obs_b.shape[0]].detach()
                                q_loss_out = cpqe.update_q(
                                    q_obs_b,
                                    q_actions_b,
                                    q_rewards_b,
                                    q_next_b,
                                    q_dones_b,
                                    gamma=q_gamma,
                                )
                                q_losses.append(q_loss_out.get("q_loss", 0.0))
                        if q_losses:
                            cpqe_losses["cpqe_q_loss"] = float(sum(q_losses) / max(1, len(q_losses)))
                            cpqe_losses["cpqe_q_updates"] = int(q_updates)
                except Exception:
                    pass
            else:
                cpqe_losses = {}
        else:
            cpqe_losses = {}
    except Exception:
        cpqe_losses = {}

    losses = learner.update_from_rollout(
        obs=obs_t,
        actions_cont=actions_cont_t,
        actions_disc=actions_disc_t,
        rewards=rewards_t,
        dones=dones_t,
        action_masks=action_masks_t,
        old_log_probs=to_tensor(batch.log_probs) if batch.log_probs else None,
        old_values=to_tensor(batch.values) if batch.values else None,
        seq_lens=batch.seq_lens,
    )
    losses["policy_name"] = batch.policy_name
    losses["ts"] = time.time()
    losses.update(wm_losses)
    losses.update(efe_losses)
    losses.update(cpqe_losses)

    # --- Optional Dreamer-lite imagination update ---
    try:
        if os.environ.get("METABONK_USE_DREAM_TRAINING", "1") in ("1", "true", "True"):
            wm = _world_models.get(batch.policy_name)
            if wm is not None:
                # Prevent world-model grads from accumulating during policy dream update.
                try:
                    wm.zero_grad(set_to_none=True)
                    wm.opt.zero_grad(set_to_none=True)
                except Exception:
                    pass
                dream_h = int(os.environ.get("METABONK_DREAM_HORIZON", "5"))
                dream_k = int(os.environ.get("METABONK_DREAM_STARTS", "8"))
                dream_gamma = float(os.environ.get("METABONK_DREAM_GAMMA", str(learner.cfg.gamma)))
                dream_losses = learner.dream_update(
                    wm,
                    obs_t.detach(),
                    horizon=dream_h,
                    num_starts=dream_k,
                    gamma=dream_gamma,
                )
                losses.update(dream_losses)
                try:
                    wm.zero_grad(set_to_none=True)
                except Exception:
                    pass
    except Exception:
        pass
    # Throughput estimate (steps/sec) based on rollout cadence.
    now = losses["ts"]
    if _last_push_ts is not None:
        dt = max(1e-6, now - _last_push_ts)
        tps = len(batch.obs) / dt
        _tps_ema = tps if _tps_ema == 0.0 else (0.9 * _tps_ema + 0.1 * tps)
    _last_push_ts = now
    losses["batch_steps"] = len(batch.obs)
    losses["learner_tps"] = _tps_ema

    # Store latest metrics snapshot for dashboard polling.
    _last_metrics[batch.policy_name] = _sanitize_metrics(losses)
    # Persist weights so restarts don't go back to zero knowledge.
    try:
        _maybe_save_ppo_weights(learner, policy_name=batch.policy_name, obs_dim=obs_dim)
    except Exception:
        pass
    return {"ok": True, "losses": losses}


@app.post("/push_demo")
async def push_demo(batch: DemoBatch):
    """Online supervised imitation update from demo samples.

    Intended for "watch me play" training: apply a small gradient step then discard.
    """
    if not batch.obs:
        raise HTTPException(status_code=400, detail="empty demo batch")
    obs_dim = len(batch.obs[0])
    learner = _get_or_create(batch.policy_name, obs_dim=obs_dim)

    obs_t = to_tensor(batch.obs)
    target_cont_t = to_tensor(batch.actions_cont)
    target_disc_t = to_tensor(batch.actions_disc, dtype=torch.long)
    action_masks_t = None
    if batch.action_masks is not None:
        try:
            action_masks_t = to_tensor(batch.action_masks, dtype=torch.long)
        except Exception:
            action_masks_t = None

    cont_w = float(os.environ.get("METABONK_DEMO_CONT_W", "1.0"))
    disc_w = float(os.environ.get("METABONK_DEMO_DISC_W", "1.0"))
    max_grad = float(os.environ.get("METABONK_DEMO_MAX_GRAD_NORM", "1.0"))

    with _demo_lock:
        learner.net.train()
        obs = obs_t.to(learner.device)
        tgt_cont = target_cont_t.to(learner.device)
        tgt_disc = target_disc_t.to(learner.device)
        mask = action_masks_t.to(learner.device) if action_masks_t is not None else None

        mu, _std, logits, _v, _ = learner.net.forward(obs)  # type: ignore[arg-type]

        cont_dim = min(int(getattr(learner.cfg, "continuous_dim", 2)), int(tgt_cont.shape[-1]))
        cont_loss = (
            torch.nn.functional.mse_loss(mu[:, :cont_dim], tgt_cont[:, :cont_dim])
            if cont_dim > 0
            else torch.zeros((), device=learner.device)
        )

        disc_loss = torch.zeros((), device=learner.device)
        try:
            for i, l in enumerate(logits):
                if tgt_disc.numel() == 0 or i >= tgt_disc.shape[-1]:
                    break
                n = int(l.shape[-1])
                target_i = tgt_disc[:, i].clamp(min=0, max=max(0, n - 1))
                if i == 0 and mask is not None:
                    try:
                        if mask.shape == l.shape:
                            l = learner.net._mask_logits(l, mask)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                disc_loss = disc_loss + torch.nn.functional.cross_entropy(l, target_i)
        except Exception:
            pass

        loss = cont_w * cont_loss + disc_w * disc_loss
        learner.opt.zero_grad(set_to_none=True)
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(learner.net.parameters(), max_grad)
        except Exception:
            pass
        learner.opt.step()
        learner.version += 1
        learner.net.eval()

        try:
            _maybe_save_ppo_weights(learner, policy_name=batch.policy_name, obs_dim=obs_dim)
        except Exception:
            pass

        metrics = {
            "demo_loss": float(loss.detach().item()),
            "demo_cont_loss": float(cont_loss.detach().item()),
            "demo_disc_loss": float(disc_loss.detach().item()),
            "demo_batch": int(obs_t.shape[0]),
            "ts": time.time(),
        }
        _last_metrics[batch.policy_name] = {**_last_metrics.get(batch.policy_name, {}), **_sanitize_metrics(metrics)}
        return {"ok": True, "policy_name": batch.policy_name, "metrics": metrics}


@app.post("/push_visual_rollout")
async def push_visual_rollout(batch: VisualRolloutBatch):
    """Visual-only learning: update the world model from observed obs sequences.

    This endpoint intentionally does not accept (or store) player inputs.
    It trains a world model with action_dim=0 (action-free dynamics).
    """
    if not batch.obs:
        raise HTTPException(status_code=400, detail="empty visual rollout")
    obs_dim = len(batch.obs[0])
    want_dev = str(os.environ.get("METABONK_WORLD_MODEL_DEVICE") or os.environ.get("METABONK_DEVICE") or "").strip()
    device = torch.device(resolve_device(want_dev, context="world model"))

    obs_t = to_tensor(batch.obs).to(device)
    T = int(obs_t.shape[0])
    # Action-free: shape [T, 0]
    actions_t = torch.zeros((T, 0), device=device, dtype=torch.float32)
    if batch.rewards is not None and len(batch.rewards) == T:
        rewards_t = to_tensor(batch.rewards).to(device)
    else:
        rewards_t = torch.zeros((T,), device=device, dtype=torch.float32)
    dones_t = None
    if batch.dones is not None and len(batch.dones) == T:
        try:
            dones_t = to_tensor(batch.dones, dtype=torch.bool).to(device)
        except Exception:
            dones_t = None

    wm = _get_or_create_world_model(batch.policy_name, obs_dim=obs_dim, action_dim=0, device=device)
    wm.train()
    losses = wm.update_from_rollout(obs_t, actions_t, rewards_t, dones=dones_t)
    wm.eval()

    metrics = {"policy_name": batch.policy_name, "ts": time.time(), "visual_steps": T, **(losses or {})}
    _last_metrics[batch.policy_name] = {**_last_metrics.get(batch.policy_name, {}), **_sanitize_metrics(metrics)}
    return {"ok": True, "losses": metrics}


@app.get("/metrics")
async def get_metrics(policy_name: Optional[str] = None):
    """Return latest learner metrics for RCC."""
    if policy_name:
        return _last_metrics.get(policy_name, {})
    # Return all policies if not specified.
    return _last_metrics


@app.post("/offline/world_model/train")
async def offline_world_model_train(req: OfflineWorldModelTrainRequest):
    """Train/update the Dreamer-lite world model from offline .pt episodes (e.g., video demos)."""
    # Choose device consistent with learner policies.
    want_dev = str(os.environ.get("METABONK_LEARNER_DEVICE") or os.environ.get("METABONK_DEVICE") or "").strip()
    device = torch.device(resolve_device(want_dev, context="offline world model"))

    # Peek first episode to infer dims.
    first_ep = None
    for _path, ep in _iter_pt_episodes(req.pt_dir, limit=max(1, int(req.max_episodes or 0)) or 1):
        first_ep = ep
        break
    if first_ep is None:
        raise HTTPException(status_code=404, detail=f"no .pt episodes found in {req.pt_dir}")

    obs0 = first_ep.get("observations")
    act0 = first_ep.get("actions")
    if obs0 is None or act0 is None:
        raise HTTPException(status_code=400, detail="invalid episode format (missing observations/actions)")
    obs_dim = int(obs0.shape[-1])
    act_dim = int(act0.shape[-1])

    wm = _get_or_create_world_model(req.policy_name, obs_dim=obs_dim, action_dim=act_dim, device=device)
    wm.train()

    losses_hist: List[Dict[str, float]] = []
    for _epoch in range(int(req.epochs)):
        for _path, ep in _iter_pt_episodes(req.pt_dir, limit=int(req.max_episodes or 0)):
            try:
                obs_t, act_t, rew_t, done_t = _episode_to_tensors(ep, device=device)
                l = wm.update_from_rollout(obs_t, act_t, rew_t, dones=done_t)
                losses_hist.append({k: float(v) for k, v in (l or {}).items()})
            except Exception:
                continue

    # Aggregate.
    agg: Dict[str, float] = {}
    if losses_hist:
        keys = set().union(*[d.keys() for d in losses_hist])
        for k in keys:
            vals = [d.get(k) for d in losses_hist if d.get(k) is not None]
            if vals:
                agg[k] = float(sum(vals) / len(vals))

    if req.out_ckpt:
        out = Path(req.out_ckpt)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": wm.state_dict(), "config": wm.cfg}, out)

    return {"ok": True, "policy_name": req.policy_name, "device": str(device), "avg_losses": agg, "updates": len(losses_hist)}


@app.post("/offline/dream/train")
async def offline_dream_train(req: OfflineDreamTrainRequest):
    """Run Dreamer-lite imagination updates from offline .pt episodes (no workers required)."""
    want_dev = str(os.environ.get("METABONK_LEARNER_DEVICE") or os.environ.get("METABONK_DEVICE") or "").strip()
    device = torch.device(resolve_device(want_dev, context="offline dream"))

    # Load/ensure world model.
    wm = _world_models.get(req.policy_name)
    if req.world_model_ckpt:
        ckpt = torch.load(req.world_model_ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config")
        if cfg is None:
            raise HTTPException(status_code=400, detail="invalid world_model_ckpt (missing config)")
        wm = WorldModel(cfg).to(device)
        wm.load_state_dict(ckpt.get("model_state_dict") or {})
        _world_models[req.policy_name] = wm

    if wm is None:
        # Try to infer dims and create a fresh world model if none exists.
        first_ep = None
        for _path, ep in _iter_pt_episodes(req.pt_dir, limit=1):
            first_ep = ep
            break
        if first_ep is None:
            raise HTTPException(status_code=404, detail=f"no .pt episodes found in {req.pt_dir}")
        obs0 = first_ep.get("observations")
        act0 = first_ep.get("actions")
        if obs0 is None or act0 is None:
            raise HTTPException(status_code=400, detail="invalid episode format (missing observations/actions)")
        wm = _get_or_create_world_model(
            req.policy_name,
            obs_dim=int(obs0.shape[-1]),
            action_dim=int(act0.shape[-1]),
            device=device,
        )

    wm = wm.to(device)
    wm.eval()

    # Create a policy learner (dream actor lives here).
    learner = _learners.get(req.policy_name)
    if learner is None:
        learner = _get_or_create(req.policy_name, obs_dim=int(wm.cfg.obs_dim))
    # Ensure learner matches device.
    try:
        learner.device = device
        learner.net = learner.net.to(device)
        if learner.dream_actor is not None:
            learner.dream_actor = learner.dream_actor.to(device)
    except Exception:
        pass

    # Preload episodes (small list) for sampling.
    episodes: List[Dict[str, Any]] = []
    for _path, ep in _iter_pt_episodes(req.pt_dir, limit=int(req.max_episodes or 0)):
        episodes.append(ep)
    if not episodes:
        raise HTTPException(status_code=404, detail=f"no .pt episodes found in {req.pt_dir}")

    gamma = float(req.gamma if req.gamma is not None else learner.cfg.gamma)

    losses_hist: List[Dict[str, float]] = []
    for _step in range(int(req.steps)):
        # Sample a random episode and random start indices.
        ep = episodes[int(torch.randint(0, len(episodes), (1,), device=device).item())]
        obs = ep.get("observations")
        if obs is None:
            continue
        obs_t = obs.to(device) if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, device=device, dtype=torch.float32)
        if obs_t.dim() != 2 or obs_t.shape[0] < 2:
            continue
        B = min(int(req.batch_obs), int(obs_t.shape[0]))
        idx = torch.randint(0, obs_t.shape[0], (B,), device=device)
        obs_batch = obs_t[idx]
        l = learner.dream_update(
            wm,
            obs_batch,
            horizon=int(req.horizon),
            num_starts=int(req.starts),
            gamma=gamma,
        )
        losses_hist.append({k: float(v) for k, v in (l or {}).items()})

    agg: Dict[str, float] = {}
    if losses_hist:
        keys = set().union(*[d.keys() for d in losses_hist])
        for k in keys:
            vals = [d.get(k) for d in losses_hist if d.get(k) is not None]
            if vals:
                agg[k] = float(sum(vals) / len(vals))

    _last_metrics[req.policy_name] = _sanitize_metrics({**agg, "ts": time.time(), "policy_name": req.policy_name, "offline": True})
    return {"ok": True, "policy_name": req.policy_name, "device": str(device), "avg_losses": agg, "steps": int(req.steps)}


@app.post("/merge_policies")
async def merge_policies(payload: Dict[str, Any]):
    """Federated / Hive Mind merge of multiple policies into a target policy.

    Expected payload:
      {
        "source_policies": ["Scout","Speedrunner",...],
        "target_policy": "God",
        "base_policy": null,          # optional
        "method": "ties"|"weighted",  # optional
        "topk": 0.2,                  # optional (TIES sparsify)
        "role_weights": {"Scout":1.0} # optional (weighted merge)
      }
    """
    source_policies = payload.get("source_policies") or []
    if not isinstance(source_policies, list) or not source_policies:
        raise HTTPException(status_code=400, detail="source_policies must be non-empty list")
    target_policy = str(payload.get("target_policy") or "God")
    base_policy = payload.get("base_policy")
    method = str(payload.get("method") or "ties").lower()
    topk = float(payload.get("topk") or 0.2)
    role_weights = payload.get("role_weights") if isinstance(payload.get("role_weights"), dict) else None
    dry_run = bool(payload.get("dry_run") or False)
    force = bool(payload.get("force") or False) or os.environ.get("METABONK_MERGE_FORCE", "0") in ("1", "true", "True")
    auto_llm = bool(payload.get("auto_llm") or False) or os.environ.get(
        "METABONK_LLM_AUTO_MERGE", "1"
    ) in ("1", "true", "True")

    # Optionally let LLM propose merge weights/method before locking.
    if auto_llm:
        try:
            from src.cognitive.llm_weighting import LLMWeightComposer

            composer = LLMWeightComposer()
            proposal = composer.propose_merge(
                source_policies=list(source_policies),
                target_policy=target_policy,
                metrics_snapshot=_last_metrics,
            )
            # Only override fields not explicitly set.
            if "method" not in payload:
                method = proposal.method
            if "topk" not in payload:
                topk = proposal.topk
            if role_weights is None or "role_weights" not in payload:
                role_weights = proposal.role_weights
        except Exception:
            pass

    # Ensure all sources exist.
    missing = [p for p in source_policies if p not in _learners]
    if missing:
        raise HTTPException(status_code=404, detail=f"missing learners: {missing}")

    with _merge_lock:
        # Determine base state.
        if base_policy and base_policy in _learners:
            base_name = str(base_policy)
        elif target_policy in _learners:
            base_name = target_policy
        else:
            base_name = str(source_policies[0])

        base_learner = _learners[base_name]
        base_state = {k: v.detach().clone() for k, v in base_learner.net.state_dict().items()}

        # Build task vectors for each source relative to base.
        vectors: List[TaskVector] = []
        for sp in source_policies:
            if sp == base_name:
                continue
            st = _learners[sp].net.state_dict()
            # Validate compatibility.
            for k, bv in base_state.items():
                if k not in st or st[k].shape != bv.shape:
                    raise HTTPException(
                        status_code=400,
                        detail=f"incompatible param {k} between {base_name} and {sp}",
                    )
            meta = TaskVectorMetadata(name=str(sp), tags=["hive_mind"])
            vectors.append(TaskVector.from_state_dicts(base_state, st, meta))

        # Merge report / gating stats.
        report: Dict[str, Any] = {
            "sources": [],
            "pairwise": [],
            "method": method,
            "topk": float(topk),
        }
        for vec in vectors:
            report["sources"].append(
                {
                    "name": vec.metadata.name,
                    "magnitude": vec.magnitude(),
                    "param_count": vec.param_count(),
                }
            )
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1 = vectors[i]
                v2 = vectors[j]
                report["pairwise"].append(
                    {
                        "a": v1.metadata.name,
                        "b": v2.metadata.name,
                        "cosine": v1.cosine_similarity(v2),
                        "sign_agreement": v1.sign_agreement(v2),
                    }
                )

        min_cos = float(payload.get("min_cosine") or os.environ.get("METABONK_MERGE_MIN_COSINE", "0.0"))
        min_sign = float(payload.get("min_sign_agreement") or os.environ.get("METABONK_MERGE_MIN_SIGN_AGREEMENT", "0.0"))
        if report["pairwise"]:
            cos_vals = [p["cosine"] for p in report["pairwise"]]
            sign_vals = [p["sign_agreement"] for p in report["pairwise"]]
            report["cosine_min"] = float(min(cos_vals))
            report["cosine_avg"] = float(sum(cos_vals) / max(1, len(cos_vals)))
            report["sign_min"] = float(min(sign_vals))
            report["sign_avg"] = float(sum(sign_vals) / max(1, len(sign_vals)))
        else:
            report["cosine_min"] = 0.0
            report["cosine_avg"] = 0.0
            report["sign_min"] = 0.0
            report["sign_avg"] = 0.0

        if dry_run:
            return {"ok": True, "dry_run": True, "report": report}
        if not force and (report["cosine_min"] < min_cos or report["sign_min"] < min_sign):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "merge_gated",
                    "min_cosine": min_cos,
                    "min_sign_agreement": min_sign,
                    "report": report,
                },
            )

        if not vectors:
            # Nothing to merge; copy base.
            merged_vector = TaskVector({})
        elif method == "weighted" and role_weights:
            # Weighted merge is simple average with role weights.
            total_w = 0.0
            acc: Dict[str, torch.Tensor] = {}
            for vec in vectors:
                w = float(role_weights.get(vec.metadata.name, 1.0))
                total_w += w
                for k, t in vec.vector.items():
                    acc[k] = acc.get(k, torch.zeros_like(t)) + t * w
            merged_vector = TaskVector(
                {k: v / max(1e-8, total_w) for k, v in acc.items()},
                TaskVectorMetadata(name=f"weighted({'+'.join([v.metadata.name for v in vectors])})", tags=["merged"]),
            )
        else:
            merger = TIESMerger(topk=topk)
            merged_vector = merger.merge(vectors)

        new_state = merged_vector.apply_to_state_dict(base_state)

        # Optional: persist merged vector to skill DB.
        if os.environ.get("METABONK_MERGE_SAVE_VECTOR", "0") in ("1", "true", "True"):
            try:
                from src.learner.task_vectors import SkillVectorDatabase

                db_path = os.environ.get("METABONK_SKILL_DB_PATH", "./skill_vectors")
                skill_db = SkillVectorDatabase(db_path)
                merged_name = f"merge_{target_policy}_{int(time.time())}"
                skill_db.store_from_state_dicts(
                    merged_name,
                    base_state,
                    new_state,
                    tags=["merged", method],
                    description=f"merge:{method} base={base_name} sources={list(source_policies)}",
                    performance={},
                )
            except Exception:
                pass

        # Create / update target learner.
        obs_dim = getattr(base_learner.net, "obs_dim", None) or len(next(iter(base_state.values())))
        target_learner = _get_or_create(target_policy, obs_dim=int(obs_dim))
        target_learner.net.load_state_dict(new_state)
        target_learner.version += 1

        # Snapshot metrics for RCC.
        _last_metrics[target_policy] = {
            "policy_name": target_policy,
            "ts": time.time(),
            "merge_method": method,
            "merge_base": base_name,
            "merge_sources": list(source_policies),
            "merge_role_weights": role_weights or {},
            "merge_topk": float(topk),
            "merge_auto_llm": bool(auto_llm),
            "version": target_learner.version,
            "merge_cosine_min": report.get("cosine_min", 0.0),
            "merge_sign_min": report.get("sign_min", 0.0),
        }

    return {
        "ok": True,
        "target_policy": target_policy,
        "base_policy": base_name,
        "version": target_learner.version,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaBonk learner service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8061)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
