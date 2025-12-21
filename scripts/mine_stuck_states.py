#!/usr/bin/env python3
"""Mine "stuck/loop" state clusters from your own rollout buffer.

This is a game-agnostic helper for menu/UI competence:
- detect repeated latent/observation states in your own experience
- export representative reset points (file, index) for replay-heavy training

Inputs:
  - `.pt` episodes (dict with `observations` tensor [T,D]) as produced by
    `scripts/video_pretrain.py --phase export_pt`.

Output:
  - JSON file listing the most frequent repeated-state clusters and example indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch required") from e


def _state_hash(vec: np.ndarray, *, quant: float, dims: int) -> str:
    """Stable hash for a float state vector."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if dims > 0:
        v = v[: int(dims)]
    if quant > 0:
        q = np.clip(np.round(v / float(quant)), -127, 127).astype(np.int8)
        payload = q.tobytes()
    else:
        # Fallback: sign bits.
        bits = (v > 0).astype(np.uint8)
        payload = np.packbits(bits).tobytes()
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", default="rollouts/video_rollouts", help="Directory of .pt episodes")
    ap.add_argument("--out", default="checkpoints/stuck_states.json", help="Output JSON")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--min-repeats", type=int, default=25, help="Minimum repeats of a state hash to keep")
    ap.add_argument("--recent-window", type=int, default=240, help="Lookback window for counting repeats")
    ap.add_argument("--dims", type=int, default=64, help="How many obs dims to hash")
    ap.add_argument("--quant", type=float, default=0.25, help="Quantization step for hashing (0 => sign-bits)")
    args = ap.parse_args()

    pt_dir = Path(args.pt_dir)
    files = sorted(pt_dir.glob("*.pt"))[: max(0, int(args.max_files))]
    if not files:
        raise SystemExit(f"no .pt files found in {pt_dir}")

    min_rep = max(2, int(args.min_repeats))
    lookback = max(10, int(args.recent_window))

    clusters: Dict[str, Dict[str, Any]] = {}

    for fi, p in enumerate(files):
        try:
            ep = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        obs = ep.get("observations")
        if obs is None:
            continue
        try:
            obs_t = obs.detach().to("cpu")
            if obs_t.ndim != 2:
                continue
            arr = obs_t.numpy().astype(np.float32, copy=False)
        except Exception:
            continue

        T = int(arr.shape[0])
        if T < 10:
            continue

        # Track repeats in a rolling window.
        last_seen: Dict[str, int] = {}
        recent_counts: Dict[str, int] = {}

        for t in range(T):
            h = _state_hash(arr[t], quant=float(args.quant), dims=int(args.dims))

            # Decrement counts for items that fall out of the lookback window (approx).
            # (We keep it simple: rely on last_seen time as a proxy.)
            for k, lt in list(last_seen.items()):
                if t - int(lt) > lookback:
                    last_seen.pop(k, None)
                    recent_counts.pop(k, None)

            if h in last_seen:
                recent_counts[h] = int(recent_counts.get(h, 1)) + 1
            else:
                recent_counts[h] = 1
            last_seen[h] = t

            c = int(recent_counts.get(h, 0))
            if c >= min_rep:
                rec = clusters.get(h)
                if rec is None:
                    rec = {"hash": h, "count": 0, "examples": []}
                    clusters[h] = rec
                rec["count"] = int(rec.get("count", 0)) + 1
                ex = {"file": p.name, "index": int(t), "episode_len": int(T)}
                # Keep only a few examples per cluster.
                exs = rec.get("examples") or []
                if isinstance(exs, list) and len(exs) < 6:
                    exs.append(ex)
                    rec["examples"] = exs

        if (fi + 1) % 25 == 0:
            print(f"[mine_stuck] scanned {fi+1}/{len(files)} files, clusters={len(clusters)}")

    # Sort by count.
    items = list(clusters.values())
    items.sort(key=lambda x: int(x.get("count") or 0), reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": float(time.time()),
        "pt_dir": str(pt_dir),
        "max_files": int(args.max_files),
        "min_repeats": int(min_rep),
        "recent_window": int(lookback),
        "hash_dims": int(args.dims),
        "quant": float(args.quant),
        "clusters": items[:200],
        "total_clusters": len(items),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[mine_stuck] wrote {out} (clusters={len(items)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
