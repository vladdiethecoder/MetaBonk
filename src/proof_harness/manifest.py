"""Manifest + hashing helpers for proof artifacts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_artifacts(paths: Iterable[Path]) -> dict:
    out = {}
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        out[str(p)] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
    return out


def write_manifest(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
