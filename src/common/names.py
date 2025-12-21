"""Deterministic agent naming utilities.

The original MetaBonk project used a policy‑based naming system so that
workers can be visually tracked in logs and in-game UI. This recovery
implementation keeps that contract:

- Names are drawn from per‑policy pools in `configs/agent_names.json`.
- Generation is deterministic for a given (policy_name, instance_id, seed).
- The resulting name is passed through `MEGABONK_AGENT_NAME`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from random import Random
from typing import Dict, List, Optional


DEFAULT_NAMES: Dict[str, List[str]] = {
    "default": [
        "BonkBot",
        "Phopper",
        "Megachad",
        "Sidecar",
        "Cortex",
    ],
}


def _config_path() -> Path:
    p = os.environ.get("METABONK_AGENT_NAMES")
    if p:
        return Path(p)
    return Path("configs") / "agent_names.json"


_CACHE: Optional[Dict[str, List[str]]] = None


def load_name_pools() -> Dict[str, List[str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = _config_path()
    pools: Dict[str, List[str]] = {}
    if path.exists():
        try:
            pools = json.loads(path.read_text())
        except Exception:
            pools = {}
    # Merge defaults for any missing keys.
    for k, v in DEFAULT_NAMES.items():
        pools.setdefault(k, v)
    _CACHE = pools
    return pools


def generate_display_name(
    policy_name: str,
    instance_id: str,
    master_seed: Optional[int] = None,
) -> str:
    """Generate a deterministic display name for a worker."""
    pools = load_name_pools()
    key = (policy_name or "default").lower()
    pool = pools.get(key) or pools.get("default") or DEFAULT_NAMES["default"]
    # Derive deterministic RNG seed from stable hash.
    seed_src = f"{master_seed or 0}:{key}:{instance_id}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = Random(seed)
    return rng.choice(pool)

