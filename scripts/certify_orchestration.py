#!/usr/bin/env python3
"""Certify Eye orchestration invariants from run logs.

Checks (2-worker baseline):
  - Worker 0 resolution matches expected featured size (default 1920x1080)
  - Worker 1 resolution matches expected background size (default 640x360)
  - Worker 1 streaming is disabled (METABONK_STREAMER_ENABLED=0 in isolation log)

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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="", help="Run directory (e.g. runs/run-omega-...)")
    ap.add_argument("--featured-w", type=int, default=1920)
    ap.add_argument("--featured-h", type=int, default=1080)
    ap.add_argument("--background-w", type=int, default=640)
    ap.add_argument("--background-h", type=int, default=360)
    ap.add_argument("--require-fps", action="store_true", help="Also require ~60fps for worker0/worker1")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir.strip() else _latest_run_dir(repo_root)
    if run_dir is None or not run_dir.exists():
        print("[certify] ERROR: no run dir found (pass --run-dir or create runs/run-omega-*)", file=sys.stderr)
        return 2

    logs_dir = run_dir / "logs"
    w0_log = logs_dir / "worker_0_dmabuf.log"
    w1_log = logs_dir / "worker_1_dmabuf.log"
    iso0 = logs_dir / "worker_0_isolation.log"
    iso1 = logs_dir / "worker_1_isolation.log"

    w0_dim = _read_last_dims(w0_log)
    w1_dim = _read_last_dims(w1_log)

    ok = True
    print(f"[certify] run_dir={run_dir}")

    exp0 = (int(args.featured_w), int(args.featured_h))
    exp1 = (int(args.background_w), int(args.background_h))
    print(f"[certify] worker_0 dims={w0_dim} expected={exp0}")
    print(f"[certify] worker_1 dims={w1_dim} expected={exp1}")
    if w0_dim != exp0:
        print("[certify] FAIL: worker_0 resolution mismatch", file=sys.stderr)
        ok = False
    if w1_dim != exp1:
        print("[certify] FAIL: worker_1 resolution mismatch", file=sys.stderr)
        ok = False

    iso1_kv = _read_kv(iso1)
    se1 = str(iso1_kv.get("METABONK_STREAMER_ENABLED", "")).strip()
    print(f"[certify] worker_1 METABONK_STREAMER_ENABLED={se1!r} (expected '0')")
    if se1 != "0":
        print("[certify] FAIL: worker_1 streaming not disabled (expected METABONK_STREAMER_ENABLED=0)", file=sys.stderr)
        ok = False

    if args.require_fps:
        f0 = _estimate_fps(w0_log)
        f1 = _estimate_fps(w1_log)
        print(f"[certify] worker_0 fps_est={f0}")
        print(f"[certify] worker_1 fps_est={f1}")
        if f0 is None or f1 is None or f0 < 55.0 or f1 < 55.0:
            print("[certify] FAIL: fps estimate below threshold (need >=55)", file=sys.stderr)
            ok = False

    if ok:
        print("[certify] OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
