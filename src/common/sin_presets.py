"""Load and map SinZero PPO presets from YAML.

The user-provided `configs/sinful_ppo_configs.yaml` is the canonical
hyperparameter source for the Seven Deadly Sins. This module:

1. Loads the YAML with anchors resolved.
2. Normalizes keys to MetaBonk internal hparam names.
3. Exposes helpers for orchestrator/learner.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # PyYAML

from .sins import Sin


DEFAULT_PATH = Path("configs") / "sinful_ppo_configs.yaml"

_CACHE_RAW: Optional[Dict[str, Any]] = None
_CACHE_MAPPED: Optional[Dict[str, Dict[str, Any]]] = None


def load_raw_presets(path: Optional[str] = None) -> Dict[str, Any]:
    """Load presets YAML into a dict."""
    global _CACHE_RAW
    if _CACHE_RAW is not None:
        return _CACHE_RAW
    p = Path(path or os.environ.get("METABONK_SIN_PRESETS", str(DEFAULT_PATH)))
    if not p.exists():
        _CACHE_RAW = {}
        return _CACHE_RAW
    data = yaml.safe_load(p.read_text()) or {}
    _CACHE_RAW = data
    return data


def _map_one(preset: Dict[str, Any]) -> Dict[str, Any]:
    """Map external PPO keys -> internal hparams keys."""
    out: Dict[str, Any] = {}
    if "learning_rate" in preset:
        out["lr"] = float(preset["learning_rate"])
    if "gamma" in preset:
        out["gamma"] = float(preset["gamma"])
    if "ent_coef" in preset:
        out["entropy_coef"] = float(preset["ent_coef"])
    if "clip_coef" in preset:
        out["clip_range"] = float(preset["clip_coef"])
    if "gae_lambda" in preset:
        out["gae_lambda"] = float(preset["gae_lambda"])
    if "num_steps" in preset:
        out["batch_size"] = int(preset["num_steps"])
    if "minibatch_size" in preset:
        out["minibatch_size"] = int(preset["minibatch_size"])
    if "num_epochs" in preset:
        out["epochs"] = int(preset["num_epochs"])
    if "max_grad_norm" in preset:
        out["max_grad_norm"] = float(preset["max_grad_norm"])
    if "target_kl" in preset and preset["target_kl"] is not None:
        out["target_kl"] = float(preset["target_kl"])

    # Pass through custom blocks untouched.
    for k in ("reward_shaping", "aux_loss", "risk_distortion", "architecture"):
        if k in preset:
            out[k] = preset[k]
    return out


def load_mapped_presets(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return mapping {sin_lower: internal_hparams}."""
    global _CACHE_MAPPED
    if _CACHE_MAPPED is not None:
        return _CACHE_MAPPED
    raw = load_raw_presets(path)
    mapped: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        mapped[k.lower()] = _map_one(v)
    _CACHE_MAPPED = mapped
    return mapped


def preset_for_policy(policy_name: str) -> Optional[Dict[str, Any]]:
    presets = load_mapped_presets()
    return presets.get(policy_name.lower())


def preset_for_sin(sin: Sin) -> Optional[Dict[str, Any]]:
    return preset_for_policy(sin.value)

