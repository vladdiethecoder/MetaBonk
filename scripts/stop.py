#!/usr/bin/env python3
"""Stop the last MetaBonk job started by scripts/start.py.

This is a safety valve for cases where a prior run was killed and left GPU-heavy
processes (gamescope / ffmpeg / workers) running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Iterable


def _job_state_path(repo_root: Path) -> Path:
    return repo_root / "temp" / "metabonk_last_job.json"


def _wine_z_drive_variants(posix_path: Path) -> tuple[str, ...]:
    """Return likely Wine/Proton argv path spellings for a local POSIX path.

    Wine commonly maps the host filesystem to the Z: drive, so a path like
    `/mnt/.../MetaBonk/temp/megabonk_instances` can appear in `ps` output as:
      - `Z:/mnt/.../MetaBonk/temp/megabonk_instances`
      - `Z:\\mnt\\...\\MetaBonk\\temp\\megabonk_instances`
    """

    s = str(posix_path)
    win_slash = "Z:" + s
    win_backslash = "Z:" + s.replace("/", "\\")
    return (
        s,
        win_slash,
        win_backslash,
        win_slash.lower(),
        win_backslash.lower(),
    )


def _kill_pgid(pgid: int) -> None:
    if os.name != "posix":
        raise SystemExit("stop.py only supports posix killpg()")
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except Exception:
        return
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            os.killpg(int(pgid), 0)
        except Exception:
            return
        time.sleep(0.1)
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except Exception:
        pass


def _ps_lines() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,pgid,user,args"], timeout=2.0)
        return out.decode("utf-8", "replace").splitlines()
    except Exception:
        return []


def _find_metabonk_pgids(lines: Iterable[str], user: str, *, repo_root: Path) -> set[int]:
    # Keep this conservative: never match on "Megabonk" alone because the user may
    # be running a normal Steam session. We only auto-detect PGIDs via:
    #   - known MetaBonk python entrypoints, OR
    #   - repo-local temp paths we control (compatdata/megabonk_instances)
    strong = (
        "src.worker.main",
        "src.orchestrator.main",
        "src.learner.service",
        "src.vision.service",
        "scripts/start_omega.py",
        "scripts/start.py",
        "scripts/watch_visual.py",
    )
    temp_root = (repo_root / "temp").resolve()
    repo_sentinels = tuple(
        s
        for p in (temp_root / "compatdata", temp_root / "megabonk_instances")
        for s in _wine_z_drive_variants(p)
    )
    pgids: set[int] = set()
    for ln in lines:
        try:
            parts = ln.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid_s, pgid_s, usr, cmd = parts
            if usr != user:
                continue
            if not (any(n in cmd for n in strong) or any(s in cmd for s in repo_sentinels)):
                continue
            pgid = int(pgid_s)
            if pgid > 1:
                pgids.add(pgid)
        except Exception:
            continue
    return pgids


def _kill_pids(pids: Iterable[int], *, sig: int) -> None:
    for pid in sorted({int(p) for p in pids if int(p) > 1}):
        try:
            os.kill(pid, sig)
        except Exception:
            pass


def _collect_metabonk_pids(repo_root: Path, *, user: str) -> set[int]:
    """Best-effort PID sweep for stragglers that escaped process groups (Proton/Wine, etc)."""
    try:
        import psutil  # type: ignore
    except Exception:
        return set()

    # Important: keep this *conservative*. This script runs on a real desktop and
    # must not kill unrelated `gamescope`/`ffmpeg`/`wine` processes.
    #
    # Strategy:
    #   - Only match processes that very likely belong to this repo/job ("strong" needles)
    #   - Then kill their children recursively (catches Proton/Wine helpers).
    temp_root = (repo_root / "temp").resolve()
    repo_sentinels = tuple(
        s
        for p in (temp_root / "compatdata", temp_root / "megabonk_instances")
        for s in _wine_z_drive_variants(p)
    )
    strong_needles = (
        "src.worker.main",
        "src.orchestrator.main",
        "src.learner.service",
        "src.vision.service",
        "scripts/start_omega.py",
        "scripts/start.py",
        "scripts/watch_visual.py",
        # repo-local roots we control (strong signals, incl Wine Z: mappings)
        *repo_sentinels,
    )

    pids: set[int] = set()
    for p in psutil.process_iter(["pid", "username", "cmdline"]):
        try:
            if user and (p.info.get("username") or "") != user:
                continue
            cmdline = p.info.get("cmdline") or []
            cmd = " ".join(cmdline) if isinstance(cmdline, list) else str(cmdline)
            if not cmd:
                continue
            if not any(n in cmd for n in strong_needles):
                continue
            pids.add(int(p.info["pid"]))
            # Pull children too (Proton tends to spawn helper processes).
            try:
                for c in p.children(recursive=True):
                    pids.add(int(c.pid))
            except Exception:
                pass
        except Exception:
            continue

    # Avoid killing our own PID.
    try:
        pids.discard(os.getpid())
    except Exception:
        pass
    return pids


def _remaining_metabonk_pids(repo_root: Path, *, user: str) -> set[int]:
    """Return a conservative set of MetaBonk-related PIDs still running."""
    try:
        import psutil  # type: ignore
    except Exception:
        return set()

    temp_root = (repo_root / "temp").resolve()
    repo_sentinels = tuple(
        s
        for p in (temp_root / "compatdata", temp_root / "megabonk_instances")
        for s in _wine_z_drive_variants(p)
    )
    strong_needles = (
        "src.worker.main",
        "src.orchestrator.main",
        "src.learner.service",
        "src.vision.service",
        "scripts/start_omega.py",
        "scripts/start.py",
        "scripts/watch_visual.py",
        *repo_sentinels,
    )

    pids: set[int] = set()
    for p in psutil.process_iter(["pid", "username", "cmdline"]):
        try:
            if user and (p.info.get("username") or "") != user:
                continue
            cmdline = p.info.get("cmdline") or []
            cmd = " ".join(cmdline) if isinstance(cmdline, list) else str(cmdline)
            if not cmd:
                continue
            if not any(n in cmd for n in strong_needles):
                continue
            pids.add(int(p.info["pid"]))
        except Exception:
            continue
    try:
        pids.discard(os.getpid())
    except Exception:
        pass
    return pids


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop the last MetaBonk job (and any detected stragglers).")
    parser.add_argument("--all", action="store_true", help="Also kill any detected MetaBonk-related process groups.")
    parser.add_argument(
        "--go2rtc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also stop go2rtc docker-compose stack (docker/docker-compose.go2rtc.yml).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    p = _job_state_path(repo_root)
    killed_any = False
    try:
        st = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        st = {}
        try:
            p.unlink()
        except Exception:
            pass
    try:
        pgid = int(st.get("pgid"))
    except Exception:
        pgid = None

    if pgid:
        print(f"[stop] killing last job process group {pgid}")
        _kill_pgid(pgid)
        killed_any = True

    if args.all or not killed_any:
        user = os.environ.get("USER") or ""
        lines = _ps_lines()
        pgids = _find_metabonk_pgids(lines, user=user, repo_root=repo_root) if user else set()
        # Avoid killing our own pgid.
        try:
            pgids.discard(os.getpgrp())
        except Exception:
            pass
        if pgids:
            print(f"[stop] killing detected MetaBonk process groups: {sorted(pgids)}")
            for g in sorted(pgids):
                _kill_pgid(int(g))
            killed_any = True
        else:
            print("[stop] no detected MetaBonk process groups")

        # Second pass: kill individual stragglers that escaped process groups
        # (common with Proton/Wine helper processes).
        if user:
            pids = _collect_metabonk_pids(repo_root, user=user)
        else:
            pids = set()
        if pids:
            print(f"[stop] killing detected MetaBonk PIDs: {sorted(pids)[:12]}{'...' if len(pids) > 12 else ''}")
            _kill_pids(pids, sig=signal.SIGTERM)
            # Wait briefly then SIGKILL remaining.
            t0 = time.time()
            while time.time() - t0 < 6.0:
                alive: set[int] = set()
                for pid in list(pids):
                    try:
                        os.kill(pid, 0)
                        alive.add(pid)
                    except Exception:
                        pass
                if not alive:
                    break
                time.sleep(0.2)
                pids = alive
            if pids:
                _kill_pids(pids, sig=signal.SIGKILL)
            killed_any = True

    # go2rtc runs in a Docker container and won't be killed by killpg().
    if args.go2rtc or args.all:
        compose = os.environ.get("METABONK_DOCKER_COMPOSE") or "docker"
        cmd = [compose]
        if compose == "docker":
            cmd += ["compose"]
        cmd += ["-f", str(repo_root / "docker" / "docker-compose.go2rtc.yml"), "down", "--remove-orphans"]
        env0 = dict(os.environ)
        try:
            subprocess.call(cmd, cwd=str(repo_root), env=env0)
        except Exception:
            pass

        # Some environments export DOCKER_HOST to a rootless podman socket while the
        # MetaBonk go2rtc container is managed by rootful docker. Best-effort: also
        # try stopping via the default docker socket (DOCKER_HOST unset).
        dh = str(env0.get("DOCKER_HOST") or "")
        if dh and "podman" in dh:
            env1 = dict(env0)
            env1.pop("DOCKER_HOST", None)
            try:
                subprocess.call(cmd, cwd=str(repo_root), env=env1)
            except Exception:
                pass

    # Verification pass: ensure no known MetaBonk processes remain.
    user = os.environ.get("USER") or ""
    rem = _remaining_metabonk_pids(repo_root, user=user) if user else set()
    if rem:
        print(f"[stop] WARNING: {len(rem)} MetaBonk PIDs still running; force killing")
        _kill_pids(rem, sig=signal.SIGKILL)
        # Short wait then re-check once.
        time.sleep(0.5)
        rem2 = _remaining_metabonk_pids(repo_root, user=user) if user else set()
        if rem2:
            print(f"[stop] WARNING: still running after SIGKILL: {sorted(rem2)[:12]}{'...' if len(rem2) > 12 else ''}")
        else:
            print("[stop] ok: no remaining MetaBonk PIDs detected")
    else:
        print("[stop] ok: no remaining MetaBonk PIDs detected")

    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
