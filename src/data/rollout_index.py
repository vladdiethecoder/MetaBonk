"""Rollout indexing utilities.

Production training needs fast dataset introspection:
  - choose curricula based on rollout metadata (T, shapes, reward presence)
  - avoid re-parsing large files when nothing changed

This module builds an index over `.npz` and `.pt` rollouts that exist in the
repo (video demos, on-policy recordings, exported `.pt` rollouts, etc.).

Index format:
  - JSONL (one record per file), written atomically
  - Each record includes a `signature` (size + mtime_ns) for incremental updates
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _safe_np_load(path: Path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)


def _file_signature(path: Path) -> Tuple[int, int]:
    st = path.stat()
    size = int(st.st_size)
    # Prefer nanosecond precision when available.
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    return size, mtime_ns


def iter_rollout_files(roots: Sequence[str | Path], *, recursive: bool = False) -> Iterator[Path]:
    exts = {".npz", ".pt"}
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in exts:
                yield root
            continue

        if recursive:
            # rglob is fine for typical rollout directory sizes.
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    yield p
            continue

        # Non-recursive (fast): scan only this directory.
        try:
            with os.scandir(root) as it:
                for ent in it:
                    if not ent.is_file():
                        continue
                    p = Path(ent.path)
                    if p.suffix.lower() in exts:
                        yield p
        except FileNotFoundError:
            continue


@dataclass(frozen=True)
class RolloutRecord:
    path: str
    kind: str  # npz|pt
    size_bytes: int
    mtime_ns: int
    signature: str
    # Basic shapes (if present).
    steps: Optional[int] = None
    obs_shape: Optional[List[int]] = None
    obs_dtype: Optional[str] = None
    action_shape: Optional[List[int]] = None
    action_dtype: Optional[str] = None
    reward_shape: Optional[List[int]] = None
    reward_dtype: Optional[str] = None
    done_shape: Optional[List[int]] = None
    done_dtype: Optional[str] = None
    # Optional free-form metadata (e.g., `meta` JSON string in NPZ).
    meta: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "path": str(self.path),
            "kind": str(self.kind),
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
            "signature": str(self.signature),
        }
        for k in (
            "steps",
            "obs_shape",
            "obs_dtype",
            "action_shape",
            "action_dtype",
            "reward_shape",
            "reward_dtype",
            "done_shape",
            "done_dtype",
        ):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


class RolloutIndexer:
    """Index `.npz` and `.pt` rollout files."""

    def index_file(self, path: str | Path) -> RolloutRecord:
        p = Path(path)
        size, mtime_ns = _file_signature(p)
        sig = f"{size}:{mtime_ns}"
        kind = p.suffix.lower().lstrip(".")
        if kind == "npz":
            return self._index_npz(p, size=size, mtime_ns=mtime_ns, signature=sig)
        if kind == "pt":
            return self._index_pt(p, size=size, mtime_ns=mtime_ns, signature=sig)
        raise ValueError(f"unsupported rollout file type: {p}")

    def _index_npz(self, path: Path, *, size: int, mtime_ns: int, signature: str) -> RolloutRecord:
        data = _safe_np_load(path)

        def _shape_dtype(key: str) -> Tuple[Optional[List[int]], Optional[str]]:
            if key not in data.files:
                return None, None
            arr = np.asarray(data[key])
            return [int(x) for x in arr.shape], str(arr.dtype)

        obs_shape, obs_dtype = _shape_dtype("observations")
        act_shape, act_dtype = _shape_dtype("actions")
        rew_shape, rew_dtype = _shape_dtype("rewards")
        done_shape, done_dtype = _shape_dtype("dones")

        steps = None
        if obs_shape and len(obs_shape) >= 1:
            steps = int(obs_shape[0])
        elif act_shape and len(act_shape) >= 1:
            steps = int(act_shape[0])

        meta: Dict[str, Any] = {}
        if "meta" in data.files:
            try:
                raw = data["meta"]
                # Common formats: string scalar, 0-d array, bytes, or json string.
                if isinstance(raw, np.ndarray) and raw.shape == ():
                    raw = raw.item()
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "replace")
                if isinstance(raw, str) and raw.strip().startswith("{"):
                    meta = json.loads(raw)
            except Exception:
                meta = {}

        return RolloutRecord(
            path=str(path),
            kind="npz",
            size_bytes=int(size),
            mtime_ns=int(mtime_ns),
            signature=str(signature),
            steps=int(steps) if steps is not None else None,
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            action_shape=act_shape,
            action_dtype=act_dtype,
            reward_shape=rew_shape,
            reward_dtype=rew_dtype,
            done_shape=done_shape,
            done_dtype=done_dtype,
            meta=meta,
        )

    def _index_pt(self, path: Path, *, size: int, mtime_ns: int, signature: str) -> RolloutRecord:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required to index .pt rollouts")
        payload = torch.load(path, map_location="cpu", weights_only=False)

        def _shape_dtype_tensor(x) -> Tuple[Optional[List[int]], Optional[str]]:
            try:
                if hasattr(x, "shape") and hasattr(x, "dtype"):
                    return [int(d) for d in list(x.shape)], str(x.dtype)
            except Exception:
                pass
            return None, None

        obs_shape, obs_dtype = _shape_dtype_tensor(payload.get("observations") if isinstance(payload, dict) else None)
        act_shape, act_dtype = _shape_dtype_tensor(payload.get("actions") if isinstance(payload, dict) else None)
        rew_shape, rew_dtype = _shape_dtype_tensor(payload.get("rewards") if isinstance(payload, dict) else None)
        done_shape, done_dtype = _shape_dtype_tensor(payload.get("dones") if isinstance(payload, dict) else None)

        steps = None
        if obs_shape and len(obs_shape) >= 1:
            steps = int(obs_shape[0])
        elif act_shape and len(act_shape) >= 1:
            steps = int(act_shape[0])

        meta: Dict[str, Any] = {}
        if isinstance(payload, dict):
            # Keep only cheap, JSONable scalar metadata.
            for k in ("episode_id", "episode_idx", "source", "note"):
                v = payload.get(k)
                if v is not None and isinstance(v, (str, int, float, bool)):
                    meta[str(k)] = v

        return RolloutRecord(
            path=str(path),
            kind="pt",
            size_bytes=int(size),
            mtime_ns=int(mtime_ns),
            signature=str(signature),
            steps=int(steps) if steps is not None else None,
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            action_shape=act_shape,
            action_dtype=act_dtype,
            reward_shape=rew_shape,
            reward_dtype=rew_dtype,
            done_shape=done_shape,
            done_dtype=done_dtype,
            meta=meta,
        )


def _load_existing_index(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except Exception:
            continue
        p = str(rec.get("path") or "")
        sig = str(rec.get("signature") or "")
        if p and sig:
            out[p] = dict(rec)
    return out


def build_rollout_index(
    *,
    roots: Sequence[str | Path],
    out_path: str | Path,
    incremental: bool = True,
    recursive: bool = False,
) -> Dict[str, Any]:
    """Build an index and write it to `out_path` (JSONL)."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_index(out_p) if incremental else {}
    indexer = RolloutIndexer()

    records: List[Dict[str, Any]] = []
    reused = 0
    updated = 0
    errors: List[Dict[str, Any]] = []

    for f in sorted(iter_rollout_files(roots, recursive=bool(recursive))):
        try:
            size, mtime_ns = _file_signature(f)
            sig = f"{size}:{mtime_ns}"
            prev = existing.get(str(f))
            if incremental and prev is not None and str(prev.get("signature") or "") == sig:
                records.append(prev)
                reused += 1
                continue

            rec = indexer.index_file(f).to_dict()
            records.append(rec)
            updated += 1
        except Exception as e:
            errors.append({"path": str(f), "error": str(e)})

    tmp = out_p.with_suffix(out_p.suffix + ".tmp")
    txt = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records) + ("\n" if records else "")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(out_p)

    summary = {
        "out_path": str(out_p),
        "roots": [str(Path(r)) for r in roots],
        "recursive": bool(recursive),
        "incremental": bool(incremental),
        "total": int(len(records)),
        "reused": int(reused),
        "updated": int(updated),
        "errors": int(len(errors)),
        "error_samples": errors[:5],
    }
    return summary


__all__ = ["RolloutIndexer", "RolloutRecord", "build_rollout_index", "iter_rollout_files"]

