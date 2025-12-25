#!/usr/bin/env python3
"""Certify Eye orchestration invariants from run logs.

Checks (2-worker baseline, with 1-worker fallback):
  - Worker 0 resolution matches expected featured size (default 1920x1080)
  - If worker_1 logs exist: Worker 1 resolution matches expected background size (default 640x360)
  - If worker_1 logs exist: Worker 1 streaming is disabled (METABONK_STREAMER_ENABLED=0 in isolation log)

Usage:
  python scripts/certify_orchestration.py --run-dir runs/run-omega-XXXX
  python scripts/certify_orchestration.py  # uses latest runs/run-omega-*
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


_RE_DIM = re.compile(r"\bwidth=(\d+)\b.*\bheight=(\d+)\b")
_RE_KV = re.compile(r"^(\w+)=([^\s]+)$")
_RE_VISION_OK = re.compile(r"^\[VISION\].*\bmean=([0-9.]+)\b.*\bstd=([0-9.]+)\b.*\bdevice=cuda", re.M)
_RE_INPUT_AUDIT_END = re.compile(
    r"^\[INPUT\].*audit END.*\bchanged=(\w+)\b.*\bpointer_moved=(\w+)\b.*\bsend_fail=(\w+)\b",
    re.M,
)


def _latest_run_dir(repo_root: Path) -> Optional[Path]:
    runs = repo_root / "runs"
    if not runs.exists():
        return None
    candidates = sorted(runs.glob("run-omega-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_last_dims(dmabuf_log: Path) -> Tuple[int, int]:
    if not dmabuf_log.exists():
        raise FileNotFoundError(str(dmabuf_log))
    last: Optional[Tuple[int, int]] = None
    for line in dmabuf_log.read_text(errors="ignore").splitlines():
        m = _RE_DIM.search(line)
        if m:
            last = (int(m.group(1)), int(m.group(2)))
    if last is None:
        raise RuntimeError(f"no width/height found in {dmabuf_log}")
    return last


def _estimate_fps(dmabuf_log: Path) -> Optional[float]:
    """Estimate fps from successive audit lines.

    Prefer delta(last_frame_id)/delta(ts) (robust to import_fail affecting import_ok),
    fallback to delta(import_ok)/delta(ts) if needed.
    """
    if not dmabuf_log.exists():
        return None
    points = []
    for line in dmabuf_log.read_text(errors="ignore").splitlines():
        ts = None
        ok = None
        frame_id = None
        for tok in line.split():
            m = _RE_KV.match(tok)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            if k == "ts":
                try:
                    ts = int(v)
                except Exception:
                    ts = None
            elif k == "import_ok":
                try:
                    ok = int(v)
                except Exception:
                    ok = None
            elif k == "last_frame_id":
                try:
                    frame_id = int(v)
                except Exception:
                    frame_id = None
        if ts is not None and frame_id is not None:
            points.append((ts, frame_id))
        elif ts is not None and ok is not None:
            points.append((ts, ok))
    if len(points) < 2:
        return None
    (t1, ok1), (t2, ok2) = points[-2], points[-1]
    dt = t2 - t1
    if dt <= 0:
        return None
    return float(ok2 - ok1) / float(dt)


def _read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _has_vision_ok(worker_log: Path) -> bool:
    if not worker_log.exists():
        return False
    txt = worker_log.read_text(errors="ignore")
    return bool(_RE_VISION_OK.search(txt))


def _has_input_audit_ok(worker_log: Path) -> bool:
    if not worker_log.exists():
        return False
    txt = worker_log.read_text(errors="ignore")
    m = _RE_INPUT_AUDIT_END.search(txt)
    if not m:
        return False
    changed, pointer_moved, send_fail = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
    if send_fail not in ("false", "0"):
        return False
    return (changed in ("true", "1")) or (pointer_moved in ("true", "1"))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="", help="Run directory (e.g. runs/run-omega-...)")
    ap.add_argument("--featured-w", type=int, default=1280)
    ap.add_argument("--featured-h", type=int, default=720)
    ap.add_argument("--background-w", type=int, default=640)
    ap.add_argument("--background-h", type=int, default=360)
    ap.add_argument("--require-fps", action="store_true", help="Also require ~60fps for worker0/worker1")
    ap.add_argument("--require-vision", action="store_true", help="Require at least one [VISION] mean/std line")
    ap.add_argument("--require-input-audit", action="store_true", help="Require [INPUT] audit END with effect")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir.strip() else _latest_run_dir(repo_root)
    if run_dir is None or not run_dir.exists():
        print("[certify] ERROR: no run dir found (pass --run-dir or create runs/run-omega-*)", file=sys.stderr)
        return 2

    logs_dir = run_dir / "logs"
    w0_log = logs_dir / "worker_0_dmabuf.log"
    w1_log = logs_dir / "worker_1_dmabuf.log"
    w0_worker_log = logs_dir / "worker_0.log"
    iso0 = logs_dir / "worker_0_isolation.log"
    iso1 = logs_dir / "worker_1_isolation.log"

    w0_dim = _read_last_dims(w0_log)
    has_w1 = w1_log.exists()
    w1_dim = _read_last_dims(w1_log) if has_w1 else None

    ok = True
    print(f"[certify] run_dir={run_dir}")

    exp0 = (int(args.featured_w), int(args.featured_h))
    exp1 = (int(args.background_w), int(args.background_h))
    print(f"[certify] worker_0 dims={w0_dim} expected={exp0}")
    if w0_dim != exp0:
        print("[certify] FAIL: worker_0 resolution mismatch", file=sys.stderr)
        ok = False
    if has_w1:
        assert w1_dim is not None
        print(f"[certify] worker_1 dims={w1_dim} expected={exp1}")
        if w1_dim != exp1:
            print("[certify] FAIL: worker_1 resolution mismatch", file=sys.stderr)
            ok = False
    else:
        print("[certify] worker_1 logs missing; skipping background resolution checks")

    if has_w1:
        iso1_kv = _read_kv(iso1)
        se1 = str(iso1_kv.get("METABONK_STREAMER_ENABLED", "")).strip()
        print(f"[certify] worker_1 METABONK_STREAMER_ENABLED={se1!r} (expected '0')")
        if se1 != "0":
            print(
                "[certify] FAIL: worker_1 streaming not disabled (expected METABONK_STREAMER_ENABLED=0)",
                file=sys.stderr,
            )
            ok = False
    else:
        print("[certify] worker_1 logs missing; skipping background streamer checks")

    if args.require_fps:
        f0 = _estimate_fps(w0_log)
        print(f"[certify] worker_0 fps_est={f0}")
        if f0 is None or f0 < 55.0:
            print("[certify] FAIL: worker_0 fps estimate below threshold (need >=55)", file=sys.stderr)
            ok = False
        if has_w1:
            f1 = _estimate_fps(w1_log)
            print(f"[certify] worker_1 fps_est={f1}")
            if f1 is None or f1 < 55.0:
                print("[certify] FAIL: worker_1 fps estimate below threshold (need >=55)", file=sys.stderr)
                ok = False

    if args.require_vision:
        has_vision = _has_vision_ok(w0_worker_log)
        print(f"[certify] worker_0 vision_ok={has_vision}")
        if not has_vision:
            print("[certify] FAIL: missing [VISION] mean/std line in worker_0.log", file=sys.stderr)
            ok = False

    if args.require_input_audit:
        has_input = _has_input_audit_ok(w0_worker_log)
        print(f"[certify] worker_0 input_audit_ok={has_input}")
        if not has_input:
            print("[certify] FAIL: missing/failed [INPUT] audit END line in worker_0.log", file=sys.stderr)
            ok = False

    if ok:
        print("[certify] OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
