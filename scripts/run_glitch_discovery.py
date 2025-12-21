#!/usr/bin/env python3
"""TAS glitch discovery runner (real environment adapter required).

This is a generic discovery loop that searches for "interesting" save-states
by exploring action sequences and keeping an archive of the best/novel states.

It does not generate synthetic observations. You must provide an adapter that
talks to a real backend (UnityBridge, BonkLink, SHM plugin, etc.).

Adapter interface (duck-typed):
  - reset() -> obs (any)
  - step(action) -> (obs, done) OR (obs, reward, done, info) OR gym-style tuple
  - get_save_state() -> bytes
  - load_save_state(state: bytes) -> None
  - sample_action() -> action   (optional; otherwise uses adapter.action_space.sample())
  - score(obs, info) -> float   (optional; otherwise uses reward accumulation)

Example:
  python scripts/run_glitch_discovery.py --adapter mypkg.adapters:MyAdapter --episodes 200
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _import_obj(spec: str):
    if ":" not in spec:
        raise ValueError("--adapter must be in the form module:ClassName")
    mod, name = spec.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def _parse_step(ret: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Normalize adapter.step return into (obs, reward, done, info)."""
    if isinstance(ret, tuple) and len(ret) == 2:
        obs, done = ret
        return obs, 0.0, bool(done), {}
    if isinstance(ret, tuple) and len(ret) == 3:
        obs, reward, done = ret
        return obs, float(reward), bool(done), {}
    if isinstance(ret, tuple) and len(ret) == 4:
        # Could be (obs, reward, done, info) or gymnasium (obs, reward, terminated, truncated)
        obs, reward, a, b = ret
        if isinstance(b, dict):
            return obs, float(reward), bool(a), b
        return obs, float(reward), bool(a) or bool(b), {}
    if isinstance(ret, tuple) and len(ret) >= 5:
        # gymnasium: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = ret[:5]
        return obs, float(reward), bool(terminated) or bool(truncated), dict(info or {})
    return ret, 0.0, False, {}


@dataclass
class ArchiveEntry:
    key: str
    score: float
    state_b64: str
    meta: Dict[str, Any]


def _select_entry(archive: List[ArchiveEntry]) -> ArchiveEntry:
    # Weighted pick: bias toward higher score, but keep exploration.
    if not archive:
        raise RuntimeError("empty archive")
    scores = [max(0.0, e.score) for e in archive]
    mx = max(scores) if scores else 0.0
    weights = [(0.1 + (s / (mx + 1e-6))) for s in scores]
    return random.choices(archive, weights=weights, k=1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaBonk TAS glitch discovery (adapter required)")
    parser.add_argument("--adapter", required=True, help="Python path module:ClassName")
    parser.add_argument("--adapter-kwargs", default="{}", help="JSON kwargs passed to adapter constructor")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=300, help="Max steps per rollout from an archive state")
    parser.add_argument("--stagnation", type=int, default=50, help="Early stop if no improvement for N steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--archive", default="game_states/glitch_archive.jsonl")
    parser.add_argument("--max-archive", type=int, default=500)
    args = parser.parse_args()

    random.seed(int(args.seed))

    AdapterCls = _import_obj(args.adapter)
    try:
        kwargs = json.loads(args.adapter_kwargs or "{}")
    except Exception as e:
        raise SystemExit(f"invalid --adapter-kwargs JSON: {e}")

    adapter = AdapterCls(**(kwargs if isinstance(kwargs, dict) else {}))

    # Required methods.
    for meth in ("reset", "step", "get_save_state", "load_save_state"):
        if not hasattr(adapter, meth):
            raise SystemExit(f"Adapter missing required method: {meth}()")

    # Optional methods.
    score_fn: Optional[Callable[[Any, Dict[str, Any]], float]] = None
    if hasattr(adapter, "score") and callable(getattr(adapter, "score")):
        score_fn = getattr(adapter, "score")

    archive_path = Path(args.archive)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize archive from reset.
    obs0 = adapter.reset()
    st0 = adapter.get_save_state()
    e0 = ArchiveEntry(
        key=f"seed-{int(args.seed)}",
        score=0.0,
        state_b64=base64.b64encode(st0).decode("ascii"),
        meta={"ts": time.time(), "note": "initial"},
    )
    archive: List[ArchiveEntry] = [e0]

    best_score = 0.0

    for ep in range(int(args.episodes)):
        entry = _select_entry(archive)
        adapter.load_save_state(base64.b64decode(entry.state_b64.encode("ascii")))

        total_reward = 0.0
        local_best = entry.score
        no_improve = 0
        last_info: Dict[str, Any] = {}
        last_obs: Any = None

        for t in range(int(args.horizon)):
            if hasattr(adapter, "sample_action") and callable(getattr(adapter, "sample_action")):
                action = adapter.sample_action()
            elif hasattr(adapter, "action_space") and hasattr(adapter.action_space, "sample"):
                action = adapter.action_space.sample()
            else:
                raise SystemExit("Adapter must provide sample_action() or action_space.sample().")

            ret = adapter.step(action)
            obs, reward, done, info = _parse_step(ret)
            total_reward += float(reward)
            last_obs = obs
            last_info = info

            cur_score = total_reward
            if score_fn is not None:
                try:
                    cur_score = float(score_fn(obs, info))
                except Exception:
                    cur_score = total_reward

            if cur_score > local_best + 1e-6:
                local_best = cur_score
                no_improve = 0
            else:
                no_improve += 1

            if done or no_improve >= int(args.stagnation):
                break

        # Commit best state at the end of the rollout.
        try:
            st = adapter.get_save_state()
        except Exception:
            st = None

        if st is not None and local_best > best_score + 1e-6:
            best_score = local_best
            rec = ArchiveEntry(
                key=f"ep{ep:05d}",
                score=float(local_best),
                state_b64=base64.b64encode(st).decode("ascii"),
                meta={
                    "ts": time.time(),
                    "episode": ep,
                    "seed": int(args.seed),
                    "last_info": last_info,
                    "note": "new_best",
                },
            )
            archive.append(rec)
            archive.sort(key=lambda e: e.score, reverse=True)
            archive = archive[: max(1, int(args.max_archive))]

            with archive_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__) + "\n")

            print(f"[glitch] ep={ep} NEW_BEST score={best_score:.4f} archive={len(archive)}")
        else:
            print(f"[glitch] ep={ep} score={local_best:.4f} best={best_score:.4f} archive={len(archive)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
