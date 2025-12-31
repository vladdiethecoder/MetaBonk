#!/usr/bin/env python3
"""Run an A/B study as isolated MetaBonk runs.

This script turns a YAML study spec into a repeatable set of runs:
  - spawns `./start ...` with per-variant env/args
  - waits for orchestrator + workers to register
  - collects basic metrics snapshots from the orchestrator
  - stops the stack
  - runs cert checks against the run directory

Example:
  python scripts/run_study.py --template studies/gold_smoke.yaml
  python scripts/run_study.py --study studies/gold_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is on sys.path when executed as a script (so `import src.*` works).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.study import StudySpec, load_study, slugify, write_study_template  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def _now_id() -> str:
    return str(int(time.time()))


def _http_json(url: str, timeout_s: float = 2.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "metabonk-study-runner/1.0"})
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read().decode("utf-8", "replace")
    return json.loads(raw)


def _normalize_workers_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("workers_by_id"), dict):
        return payload.get("workers_by_id") or {}
    if isinstance(payload.get("workers"), list):
        out: dict[str, Any] = {}
        for row in payload.get("workers") or []:
            if not isinstance(row, dict):
                continue
            iid = str(row.get("instance_id") or "")
            if iid:
                out[iid] = row
        return out
    return payload


def _wait_orchestrator_ready(orch_url: str, timeout_s: float) -> None:
    deadline = time.time() + max(1.0, float(timeout_s))
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            _ = _http_json(f"{orch_url.rstrip('/')}/status", timeout_s=1.5)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"orchestrator not ready within {timeout_s:.0f}s: {last_err}")


def _wait_workers(orch_url: str, expected: int, timeout_s: float) -> dict:
    deadline = time.time() + max(1.0, float(timeout_s))
    last: Optional[dict] = None
    while time.time() < deadline:
        try:
            data = _http_json(f"{orch_url.rstrip('/')}/workers", timeout_s=2.0)
            workers = _normalize_workers_payload(data)
            if isinstance(workers, dict):
                last = workers
                if len(workers) >= int(expected):
                    return workers
        except Exception:
            pass
        time.sleep(1.0)
    count = len(last or {}) if isinstance(last, dict) else 0
    raise RuntimeError(f"only {count} workers registered (expected {expected})")


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    if pgid:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass


def _stop_stack(repo_root: Path) -> None:
    try:
        subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "stop.py"), "--all", "--go2rtc"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30.0,
        )
    except Exception:
        pass


def _run_cert(repo_root: Path, run_dir: Path, *, require_fps: bool, require_vision: bool, require_input: bool) -> dict:
    cmd = [sys.executable, str(repo_root / "scripts" / "certify_orchestration.py"), "--run-dir", str(run_dir)]
    if require_fps:
        cmd.append("--require-fps")
    if require_vision:
        cmd.append("--require-vision")
    if require_input:
        cmd.append("--require-input-audit")
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(p.returncode),
        "stdout": p.stdout,
        "stderr": p.stderr,
        "ok": bool(p.returncode == 0),
    }


def _metrics_snapshot(orch_url: str, run_id: str) -> dict:
    out: dict[str, Any] = {"run_id": run_id}
    try:
        out["runs"] = _http_json(f"{orch_url.rstrip('/')}/runs", timeout_s=2.0)
    except Exception:
        out["runs"] = None
    try:
        out["workers"] = _http_json(f"{orch_url.rstrip('/')}/workers", timeout_s=2.0)
    except Exception:
        out["workers"] = None
    try:
        qs = f"run_ids={urllib.parse.quote(run_id)}&metrics=reward,score,obs_fps,act_hz,action_entropy,stream_fps&window_s=3600&stride=1"
        out["metrics"] = _http_json(f"{orch_url.rstrip('/')}/runs/metrics?{qs}", timeout_s=2.0)
    except Exception:
        out["metrics"] = None
    return out


def _run_variant(repo_root: Path, study: StudySpec, variant_idx: int) -> dict:
    v = study.variants[variant_idx]
    run_id = f"run-{slugify(study.study_id)}-{v.slug}-{_now_id()}"
    run_dir = repo_root / "runs" / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({k: str(vv) for k, vv in (study.base_env or {}).items()})
    env.update({k: str(vv) for k, vv in (v.env or {}).items()})
    env["METABONK_EXPERIMENT_ID"] = str(study.experiment_id)
    env["METABONK_EXPERIMENT_TITLE"] = str(study.title)
    env["METABONK_RUN_ID"] = str(run_id)
    env["METABONK_RUN_DIR"] = str(run_dir)

    # If cert requires these, ensure the audit emitters are enabled.
    if v.require_vision:
        env.setdefault("METABONK_VISION_AUDIT", "1")
    if v.require_input_audit:
        env.setdefault("METABONK_INPUT_AUDIT", "1")

    cmd = [str(repo_root / "start"), *[str(x) for x in (study.base_args or [])], *[str(x) for x in (v.extra_args or [])]]
    start_log = logs_dir / "study_start.log"
    with start_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(repo_root),
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )

    orch_port = None
    try:
        # Best-effort parse: `--orch-port` appears in args.
        args = cmd[1:]
        for i, tok in enumerate(args):
            if tok == "--orch-port" and i + 1 < len(args):
                orch_port = int(args[i + 1])
                break
    except Exception:
        orch_port = None
    if orch_port is None:
        try:
            orch_port = int(env.get("METABONK_ORCH_PORT") or env.get("ORCH_PORT") or 8040)
        except Exception:
            orch_port = 8040
    orch_url = env.get("ORCHESTRATOR_URL") or f"http://127.0.0.1:{orch_port}"

    ok = True
    err: Optional[str] = None
    workers_snapshot: Optional[dict] = None
    try:
        _wait_orchestrator_ready(str(orch_url), timeout_s=float(v.ready_timeout_s))
        # Determine expected workers by looking at base args if present, else default to 1.
        expected_workers = 1
        try:
            args = cmd[1:]
            for i, tok in enumerate(args):
                if tok == "--workers" and i + 1 < len(args):
                    expected_workers = max(1, int(args[i + 1]))
                    break
        except Exception:
            expected_workers = 1
        workers_snapshot = _wait_workers(str(orch_url), expected=expected_workers, timeout_s=float(v.ready_timeout_s))

        t_end = time.time() + max(0.0, float(v.duration_s))
        while time.time() < t_end:
            if proc.poll() is not None:
                raise RuntimeError(f"stack exited early (code={proc.returncode})")
            time.sleep(1.0)
    except Exception as e:
        ok = False
        err = str(e)

    # Snapshot metrics while the stack is still alive (best-effort).
    metrics = _metrics_snapshot(str(orch_url), run_id=str(run_id))

    # Stop stack.
    _stop_stack(repo_root)
    _kill_process_group(proc)
    try:
        proc.wait(timeout=15.0)
    except Exception:
        pass

    cert = _run_cert(
        repo_root,
        run_dir,
        require_fps=bool(v.require_fps),
        require_vision=bool(v.require_vision),
        require_input=bool(v.require_input_audit),
    )
    if not cert.get("ok"):
        ok = False
        if err is None:
            err = "certification failed"

    result = {
        "study_id": study.study_id,
        "experiment_id": study.experiment_id,
        "variant": {
            "name": v.name,
            "slug": v.slug,
            "description": v.description,
            "env": v.env,
            "extra_args": v.extra_args,
        },
        "run_id": run_id,
        "run_dir": str(run_dir),
        "orchestrator_url": str(orch_url),
        "ok": ok,
        "error": err,
        "workers_snapshot": workers_snapshot,
        "metrics_snapshot": metrics,
        "cert": cert,
    }
    (run_dir / "study_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=str, default="", help="Path to study YAML")
    ap.add_argument("--template", type=str, default="", help="Write a study template YAML to this path and exit")
    ap.add_argument("--max-variants", type=int, default=0, help="Optional cap on variants to run (0 = all)")
    args = ap.parse_args()

    repo_root = _repo_root()
    if args.template:
        write_study_template(Path(args.template))
        print(f"[study] wrote template: {args.template}")
        return 0
    if not args.study:
        raise SystemExit("pass --study <path> or --template <path>")

    study = load_study(Path(args.study))
    max_variants = int(args.max_variants or 0)
    variants = study.variants if max_variants <= 0 else study.variants[:max_variants]
    if not variants:
        raise SystemExit("[study] ERROR: no variants defined")

    # Always start from a clean slate.
    _stop_stack(repo_root)

    results = []
    for i in range(len(variants)):
        print(f"[study] running variant {i+1}/{len(variants)}: {variants[i].name}")
        res = _run_variant(repo_root, study, i)
        results.append(res)
        status = "OK" if res.get("ok") else "FAIL"
        print(f"[study] {status} run_id={res.get('run_id')}")

    out_dir = repo_root / "runs" / "studies" / slugify(study.study_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{_now_id()}.json"
    summary = {
        "study_id": study.study_id,
        "title": study.title,
        "experiment_id": study.experiment_id,
        "source": str(Path(args.study).expanduser().resolve()),
        "results": results,
        "created_ts": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[study] wrote summary: {summary_path}")

    if all(bool(r.get("ok")) for r in results):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
