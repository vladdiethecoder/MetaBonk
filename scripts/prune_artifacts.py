#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class FileRec:
    path: Path
    mtime: float
    size: int


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _iter_video_files(root: Path) -> Iterable[FileRec]:
    if not root.exists():
        return []
    out: list[FileRec] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in (".mp4", ".mkv", ".webm", ".ts"):
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        out.append(FileRec(path=p, mtime=float(st.st_mtime), size=int(st.st_size)))
    return out


def _clip_local_path(highlights_dir: Path, clip_url: str) -> Optional[Path]:
    u = str(clip_url or "").strip()
    if not u:
        return None
    if u.startswith("/api"):
        u = u[len("/api") :]
    if u.startswith("http://") or u.startswith("https://"):
        try:
            # Avoid importing urllib for one-off parsing; split on the first "/" after scheme+host.
            u = "/" + u.split("/", 3)[3]
        except Exception:
            return None
    if u.startswith("/highlights/"):
        rel = u[len("/highlights/") :].lstrip("/")
        return (highlights_dir / rel).resolve()
    return None


def _delete_file(p: Path, *, yes: bool) -> bool:
    if not yes:
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False


def _prune_clips_db(
    db_path: Path,
    *,
    highlights_dir: Path,
    delete_clip_urls: set[str],
    retention_before_ts: Optional[float],
    yes: bool,
) -> dict[str, int]:
    if not db_path.exists():
        return {"deleted": 0, "missing": 0, "retention": 0}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute("SELECT clip_url, timestamp FROM clips;").fetchall()
        except Exception:
            return {"deleted": 0, "missing": 0, "retention": 0}

        to_delete: set[str] = set(delete_clip_urls)
        missing = 0
        retention = 0
        for r in rows:
            cu = str(r["clip_url"] or "")
            ts = float(r["timestamp"] or 0.0)
            lp = _clip_local_path(highlights_dir, cu)
            if lp is not None and not lp.exists():
                to_delete.add(cu)
                missing += 1
            if retention_before_ts is not None and ts > 0 and ts < retention_before_ts:
                to_delete.add(cu)
                retention += 1

        if not to_delete:
            return {"deleted": 0, "missing": missing, "retention": retention}

        if not yes:
            return {"deleted": 0, "missing": missing, "retention": retention}

        with conn:
            for cu in sorted(to_delete):
                conn.execute("DELETE FROM clips WHERE clip_url = ?;", (cu,))
        return {"deleted": len(to_delete), "missing": missing, "retention": retention}
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune MetaBonk run artifacts (highlights + manifest DB).")
    ap.add_argument("--yes", action="store_true", help="Actually delete files/rows (default: dry-run).")
    ap.add_argument("--highlights-dir", default=os.environ.get("METABONK_HIGHLIGHTS_DIR", "highlights"))
    ap.add_argument("--db", default=os.environ.get("METABONK_BUILD_RUNS_DB", "checkpoints/build_runs.db"))
    ap.add_argument(
        "--retention-days",
        type=float,
        default=_env_float("METABONK_HIGHLIGHTS_RETENTION_DAYS", 14.0),
        help="Delete highlight clips older than this (days). Set 0 to disable.",
    )
    ap.add_argument(
        "--max-gb",
        type=float,
        default=_env_float("METABONK_HIGHLIGHTS_MAX_GB", 0.0),
        help="Cap highlight directory size (GiB). Set 0 to disable.",
    )
    args = ap.parse_args()

    yes = bool(args.yes)
    highlights_dir = Path(args.highlights_dir).resolve()
    db_path = Path(args.db).resolve()

    now = time.time()
    retention_before_ts: Optional[float] = None
    if float(args.retention_days) > 0:
        retention_before_ts = now - float(args.retention_days) * 86400.0

    files = list(_iter_video_files(highlights_dir))
    files.sort(key=lambda r: r.mtime)

    deleted_urls: set[str] = set()
    deleted_files = 0
    deleted_bytes = 0

    def _url_for(p: Path) -> Optional[str]:
        try:
            rel = p.resolve().relative_to(highlights_dir)
        except Exception:
            return None
        return "/highlights/" + str(rel).replace("\\", "/")

    # Age-based prune.
    if retention_before_ts is not None:
        for rec in list(files):
            if rec.mtime >= retention_before_ts:
                continue
            url = _url_for(rec.path)
            if url:
                deleted_urls.add(url)
            if yes and _delete_file(rec.path, yes=yes):
                deleted_files += 1
                deleted_bytes += int(rec.size)

        if yes:
            files = list(_iter_video_files(highlights_dir))
            files.sort(key=lambda r: r.mtime)

    # Size-budget prune (oldest first).
    max_bytes = int(float(args.max_gb) * (1024**3))
    if max_bytes > 0:
        total = sum(r.size for r in files)
        for rec in list(files):
            if total <= max_bytes:
                break
            url = _url_for(rec.path)
            if url:
                deleted_urls.add(url)
            if yes and _delete_file(rec.path, yes=yes):
                deleted_files += 1
                deleted_bytes += int(rec.size)
                total -= int(rec.size)
        if yes:
            files = list(_iter_video_files(highlights_dir))

    db_stats = _prune_clips_db(
        db_path,
        highlights_dir=highlights_dir,
        delete_clip_urls=deleted_urls,
        retention_before_ts=retention_before_ts,
        yes=yes,
    )

    if yes:
        print(f"[prune] deleted_files={deleted_files} deleted_gb={deleted_bytes / (1024**3):.3f}")
        print(f"[prune] db_deleted={db_stats['deleted']} db_missing={db_stats['missing']} db_retention={db_stats['retention']}")
    else:
        print("[prune] dry-run (pass --yes to delete)")
        print(f"[prune] would_delete_files={len(deleted_urls)}")
        print(f"[prune] would_delete_db_rows≈{max(len(deleted_urls), db_stats['missing'] + db_stats['retention'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

