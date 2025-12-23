#!/usr/bin/env python3
"""End-to-end Learning Proof Harness for MetaBonk.

Runs the full stack, validates runtime health, executes pre/post learning evals
(with negative controls), captures proof videos, and emits reproducible artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests  # type: ignore

from src.proof_harness.audit import scan_action_selection
from src.proof_harness.manifest import hash_artifacts, write_manifest
from src.proof_harness.stats import cohens_d, mean_ci, paired_permutation_test
from src.proof_harness.video import build_ffmpeg_record_cmd, build_hstack_cmd


@dataclass
class ProofConfig:
    workers: int
    seeds: List[int]
    eval_timeout_s: float
    train_seconds: float
    train_steps: int
    orch_port: int
    worker_base_port: int
    instance_prefix: str
    go2rtc: bool
    go2rtc_mode: str
    go2rtc_url: str
    stream_backend: str
    output_dir: Path
    game_dir: str
    dry_run: bool
    min_return_delta: float
    min_length_delta: float
    p_value: float
    capture_duration_s: float


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
         stdout: Optional[Path] = None, stderr: Optional[Path] = None) -> subprocess.Popen:
    out_f = open(stdout, "w") if stdout else None
    err_f = open(stderr, "w") if stderr else None
    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=out_f,
        stderr=err_f or subprocess.STDOUT,
        start_new_session=True,
    )


def _terminate(p: Optional[subprocess.Popen]) -> None:
    if not p:
        return
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < 8.0 and p.poll() is None:
        time.sleep(0.1)
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def _git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": "", "dirty": False, "status": []}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
            .decode("utf-8", "replace")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO_ROOT))
            .decode("utf-8", "replace")
            .splitlines()
        )
        info["status"] = status
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return info
    try:
        out = subprocess.check_output([
            nvsmi,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        info["nvidia"] = out.decode("utf-8", "replace").strip().splitlines()
    except Exception:
        pass
    return info


def _version_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split(" ")[0]}
    try:
        import torch  # type: ignore

        info["torch"] = str(getattr(torch, "__version__", ""))
        info["torch_cuda"] = str(getattr(torch.version, "cuda", ""))
    except Exception:
        info["torch"] = "unavailable"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            out = subprocess.check_output([ffmpeg, "-version"], timeout=2.0)
            info["ffmpeg"] = out.decode("utf-8", "replace").splitlines()[0]
        except Exception:
            info["ffmpeg"] = "unavailable"
    return info


def _orch_url(cfg: ProofConfig) -> str:
    return f"http://127.0.0.1:{cfg.orch_port}"


def _env_truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _wait_for_workers(cfg: ProofConfig, timeout_s: float = 120.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    url = f"{_orch_url(cfg)}/workers"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1.0)
            if r.ok:
                data = r.json()
                if isinstance(data, dict) and len(data) >= cfg.workers:
                    return data
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("timed out waiting for workers")


def _set_config(cfg: ProofConfig, instance_id: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["instance_id"] = instance_id
    url = f"{_orch_url(cfg)}/config/{instance_id}"
    r = requests.post(url, json=payload, timeout=2.0)
    if not r.ok:
        raise RuntimeError(f"failed to set config for {instance_id}: {r.status_code} {r.text}")


def _read_eval_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _wait_for_eval_metric(path: Path, policy: str, seed: int, *, after_ts: float, timeout_s: float) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = _read_eval_history(path)
        for row in hist[::-1]:
            if row.get("policy_name") != policy:
                continue
            if row.get("eval_seed") != seed:
                continue
            if float(row.get("ts") or 0.0) < after_ts:
                continue
            return row
        time.sleep(1.0)
    raise RuntimeError(f"eval metric timeout for policy={policy} seed={seed}")


def _overlay_text(
    *,
    policy_label: str,
    seed: int,
    step: int,
    reward: Optional[float],
    entropy: Optional[float],
) -> str:
    rtxt = f"{reward:.2f}" if reward is not None else "--"
    etxt = f"{entropy:.2f}" if entropy is not None else "--"
    return (
        f"policy={policy_label}\n"
        f"seed={seed}\n"
        f"step={step}\n"
        f"reward={rtxt}\n"
        f"entropy={etxt}\n"
    )


def _start_stream_recording(
    *,
    cfg: ProofConfig,
    instance_id: str,
    stream_url: str,
    out_path: Path,
    policy_label: str,
    seed: int,
) -> Dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found for video capture")
    overlay_path = out_path.with_suffix(".overlay.txt")
    overlay_path.write_text("initializing...")
    stop = threading.Event()

    def _overlay_loop():
        while not stop.is_set():
            try:
                r = requests.get(f"{_orch_url(cfg)}/workers", timeout=1.0)
                data = r.json() if r.ok else {}
                hb = data.get(instance_id, {}) if isinstance(data, dict) else {}
                step = int(hb.get("step") or 0)
                reward = hb.get("reward")
                entropy = hb.get("action_entropy")
                overlay_path.write_text(
                    _overlay_text(
                        policy_label=policy_label,
                        seed=seed,
                        step=step,
                        reward=reward if isinstance(reward, (int, float)) else None,
                        entropy=entropy if isinstance(entropy, (int, float)) else None,
                    )
                )
            except Exception:
                pass
            time.sleep(0.5)

    th = threading.Thread(target=_overlay_loop, daemon=True)
    th.start()
    cmd = build_ffmpeg_record_cmd(
        src_url=stream_url,
        out_path=out_path,
        duration_s=cfg.capture_duration_s,
        overlay_textfile=overlay_path,
    )
    proc = subprocess.Popen(cmd)
    return {"proc": proc, "stop": stop, "thread": th, "overlay": overlay_path}


def _stop_stream_recording(handle: Dict[str, Any]) -> None:
    proc = handle.get("proc")
    stop = handle.get("stop")
    thread = handle.get("thread")
    overlay_path = handle.get("overlay")
    if stop is not None:
        stop.set()
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < 5.0 and proc.poll() is None:
            time.sleep(0.1)
    if thread is not None:
        try:
            thread.join(timeout=2.0)
        except Exception:
            pass
    if overlay_path is not None:
        try:
            Path(overlay_path).unlink()
        except Exception:
            pass


def _validate_streams(cfg: ProofConfig, workers: Dict[str, Any]) -> None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not available (required to validate streams)")
    deadline = time.time() + 90.0
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            workers = _wait_for_workers(cfg, timeout_s=5.0)
        except Exception:
            workers = workers
        ok = True
        for iid, hb in workers.items():
            stream_url = hb.get("stream_url")
            if not stream_url:
                ok = False
                last_err = f"worker {iid} missing stream_url"
                break
            if hb.get("stream_ok") is False:
                ok = False
                last_err = f"worker {iid} stream_ok=false"
                break
            cmd = [
                ffprobe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_streams",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate",
                "-of",
                "json",
                str(stream_url),
            ]
            try:
                out = subprocess.check_output(cmd, timeout=6.0)
                data = json.loads(out.decode("utf-8", "replace"))
                streams = data.get("streams") or []
                if not streams:
                    raise RuntimeError("no video streams")
                s0 = streams[0]
                if s0.get("codec_name") not in ("h264", "hevc", "av1"):
                    raise RuntimeError(f"unexpected codec {s0.get('codec_name')}")
            except Exception as e:
                ok = False
                last_err = f"ffprobe failed for {iid}: {e}"
                break
        if ok:
            return
        time.sleep(2.0)
    raise RuntimeError(last_err or "stream validation failed")


def _run_stream_diagnostics(cfg: ProofConfig, log_path: Path, env: Dict[str, str]) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "stream_diagnostics.py"), "--backend", cfg.stream_backend],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )
    ret = proc.wait(timeout=30.0)
    if ret != 0:
        raise RuntimeError(f"stream diagnostics failed (see {log_path})")


def _static_audit(report_lines: List[str]) -> bool:
    findings = scan_action_selection(REPO_ROOT / "src" / "worker" / "main.py")
    if not findings:
        report_lines.append("- Static audit: OK (no disallowed menu automation outside allowed blocks).")
        return True
    report_lines.append("- Static audit: FAIL (disallowed patterns outside allowed blocks).")
    for f in findings:
        report_lines.append(f"  - line {f['line']}: {f['pattern']} :: {f['text']}")
    return False


def _ensure_input_backend(env: Dict[str, str]) -> None:
    backend = str(env.get("METABONK_INPUT_BACKEND") or "").strip()
    if not backend:
        raise RuntimeError("METABONK_INPUT_BACKEND must be set (uinput or xdotool) for proof harness")
    if backend not in ("uinput", "xdotool", "xdo"):
        raise RuntimeError("METABONK_INPUT_BACKEND must be uinput or xdotool for proof harness")
    buttons = env.get("METABONK_INPUT_BUTTONS") or ""
    if not buttons.strip():
        buttons = env.get("METABONK_BUTTON_KEYS") or ""
        if buttons.strip():
            env["METABONK_INPUT_BUTTONS"] = buttons
    if not buttons.strip():
        raise RuntimeError("METABONK_INPUT_BUTTONS or METABONK_BUTTON_KEYS must be set for keyboard/mouse input")


def _build_env(cfg: ProofConfig) -> Dict[str, str]:
    env = os.environ.copy()
    env["METABONK_REQUIRE_CUDA"] = "1"
    env.setdefault("METABONK_STREAM", "1")
    env.setdefault("METABONK_STREAM_CODEC", "h264")
    env.setdefault("METABONK_STREAM_CONTAINER", "mp4")
    env.setdefault("METABONK_STREAM_BACKEND", cfg.stream_backend)
    env.setdefault("METABONK_CAPTURE_CPU", "0")
    env.setdefault("METABONK_CAPTURE_DISABLED", "0")
    env.setdefault("METABONK_CAPTURE_ALL", "1")
    env.setdefault("METABONK_GST_ENCODER", "nvh264enc")
    env.setdefault("METABONK_FFMPEG_ENCODER", "h264_nvenc")
    env.setdefault("METABONK_REQUIRE_PIPEWIRE_STREAM", "1")
    env.setdefault("METABONK_USE_VLM_MENU", "0")
    env.setdefault("METABONK_MENU_EPS", "0")
    env.setdefault("METABONK_ACTION_GUARD", "1")
    env.setdefault("METABONK_ACTION_SOURCE", "policy")
    env.setdefault("METABONK_EVAL_HISTORY_PATH", str(cfg.output_dir / "eval_history.json"))
    env.setdefault("METABONK_EVAL_BEST_PATH", str(cfg.output_dir / "eval_best.json"))
    env.setdefault("METABONK_FREEZE_POLICIES", "ProofFrozen")
    env.setdefault("METABONK_REWARD_SHUFFLE_POLICIES", "ProofShuffle")
    env.setdefault("METABONK_SUPERVISE_WORKERS", "0")
    env.setdefault("METABONK_HIGHLIGHTS", "0")
    _ensure_input_backend(env)
    return env


def _collect_metrics_snapshot(phase: str, workers: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    ts = time.time()
    for iid, hb in workers.items():
        rows.append(
            {
                "ts": ts,
                "phase": phase,
                "instance_id": iid,
                "policy_name": hb.get("policy_name"),
                "step": hb.get("step"),
                "reward": hb.get("reward"),
                "steam_score": hb.get("steam_score"),
                "action_entropy": hb.get("action_entropy"),
                "survival_prob": hb.get("survival_prob"),
            }
        )


def _append_eval_rows(phase: str, eval_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    for r in eval_rows:
        rows.append(
            {
                "ts": r.get("ts"),
                "phase": phase,
                "instance_id": r.get("instance_id"),
                "policy_name": r.get("policy_name"),
                "eval_seed": r.get("eval_seed"),
                "mean_return": r.get("mean_return"),
                "mean_length": r.get("mean_length"),
                "steps": r.get("steps"),
            }
        )

def _write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_svg_line(path: Path, series: List[Tuple[float, float]], title: str) -> None:
    if not series:
        return
    width, height = 800, 320
    min_x = min(x for x, _ in series)
    max_x = max(x for x, _ in series)
    min_y = min(y for _, y in series)
    max_y = max(y for _, y in series)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1
    def sx(x):
        return 40 + (x - min_x) / (max_x - min_x) * (width - 80)
    def sy(y):
        return height - 40 - (y - min_y) / (max_y - min_y) * (height - 80)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in series)
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='#050709'/>
<text x='40' y='24' fill='#e7f6ff' font-size='14' font-family='sans-serif'>{title}</text>
<polyline fill='none' stroke='#7bffe6' stroke-width='2' points='{pts}' />
</svg>"""
    path.write_text(svg)


def _write_svg_bar(path: Path, labels: List[str], values: List[float], title: str) -> None:
    if not labels:
        return
    width, height = 800, 320
    max_v = max(values) if values else 1
    if max_v == 0:
        max_v = 1
    bar_w = (width - 80) / max(1, len(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    bars = []
    for i, v in enumerate(values):
        x = 40 + i * bar_w
        h = (v / max_v) * (height - 80)
        y = height - 40 - h
        bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w * 0.7:.1f}' height='{h:.1f}' fill='#ff9b6a' />")
        bars.append(f"<text x='{x + 4:.1f}' y='{height - 18:.1f}' fill='#e7f6ff' font-size='10'>{labels[i]}</text>")
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='#050709'/>
<text x='40' y='24' fill='#e7f6ff' font-size='14' font-family='sans-serif'>{title}</text>
{''.join(bars)}
</svg>"""
    path.write_text(svg)


def _run_eval_phase(
    *,
    cfg: ProofConfig,
    instance_id: str,
    workers: Dict[str, Any],
    policy_name: str,
    action_source: str,
    seeds: List[int],
    eval_history_path: Path,
    phase_label: str,
    video_out: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    metrics: List[Dict[str, Any]] = []
    stream_url = workers[instance_id].get("stream_url")
    # Pause training on all workers during eval to keep weights stable.
    for iid, hb in workers.items():
        _set_config(
            cfg,
            iid,
            {
                "policy_name": hb.get("policy_name") or "ProofHold",
                "eval_mode": True,
                "action_source": "policy",
                "config_poll_s": 1.0,
            },
        )
    for idx, seed in enumerate(seeds):
        _set_config(
            cfg,
            instance_id,
            {
                "policy_name": policy_name,
                "eval_mode": True,
                "eval_seed": int(seed),
                "action_source": action_source,
                "config_poll_s": 1.0,
                "capture_enabled": True,
            },
        )
        t0 = time.time()
        record_handle = None
        try:
            if video_out is not None and idx == 0:
                record_handle = _start_stream_recording(
                    cfg=cfg,
                    instance_id=instance_id,
                    stream_url=stream_url,
                    out_path=video_out,
                    policy_label=policy_name,
                    seed=int(seed),
                )
            row = _wait_for_eval_metric(
                eval_history_path,
                policy_name,
                int(seed),
                after_ts=t0,
                timeout_s=cfg.eval_timeout_s,
            )
        finally:
            if record_handle is not None:
                _stop_stream_recording(record_handle)
                if not video_out.exists():
                    raise RuntimeError(f"video capture failed for {video_out}")
        row["phase"] = phase_label
        metrics.append(row)
    return metrics, video_out


def _train_phase(
    *,
    cfg: ProofConfig,
    workers: Dict[str, Any],
    policy_name: str,
    action_source: str,
    phase_label: str,
    metrics_rows: List[Dict[str, Any]],
) -> None:
    for iid in workers.keys():
        _set_config(
            cfg,
            iid,
            {
                "policy_name": policy_name,
                "eval_mode": False,
                "action_source": action_source,
                "config_poll_s": 1.0,
            },
        )
    start_ts = time.time()
    start_steps = {iid: int(w.get("step") or 0) for iid, w in workers.items()}
    while True:
        now = time.time()
        if cfg.train_seconds > 0 and (now - start_ts) >= cfg.train_seconds:
            break
        if cfg.train_steps > 0:
            cur = _wait_for_workers(cfg, timeout_s=5.0)
            done = True
            for iid, hb in cur.items():
                delta = int(hb.get("step") or 0) - int(start_steps.get(iid, 0))
                if delta < cfg.train_steps:
                    done = False
                    break
            if done:
                break
        workers_snapshot = _wait_for_workers(cfg, timeout_s=5.0)
        _collect_metrics_snapshot(phase_label, workers_snapshot, metrics_rows)
        time.sleep(2.0)


def _dry_run(cfg: ProofConfig) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "video").mkdir(exist_ok=True)
    (cfg.output_dir / "plots").mkdir(exist_ok=True)
    (cfg.output_dir / "logs").mkdir(exist_ok=True)
    metrics_path = cfg.output_dir / "metrics.csv"
    rows = [
        {"ts": time.time(), "phase": "dry", "instance_id": "omega-0", "policy_name": "ProofTrain", "step": 0, "reward": 0.0},
    ]
    _write_metrics_csv(metrics_path, rows)
    (cfg.output_dir / "video" / "baseline_random_seeded.mp4").write_text("dry-run")
    (cfg.output_dir / "video" / "baseline_untrained.mp4").write_text("dry-run")
    (cfg.output_dir / "video" / "trained_after.mp4").write_text("dry-run")
    (cfg.output_dir / "video" / "side_by_side_before_after.mp4").write_text("dry-run")
    write_manifest(cfg.output_dir / "manifest.json", {"dry_run": True, "artifacts": hash_artifacts(cfg.output_dir.rglob("*"))})
    (cfg.output_dir / "REPORT.md").write_text("Dry-run proof report.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MetaBonk end-to-end learning proof harness")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--eval-timeout-s", type=float, default=600.0)
    ap.add_argument("--train-seconds", type=float, default=600.0)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--orch-port", type=int, default=8040)
    ap.add_argument("--worker-base-port", type=int, default=5000)
    ap.add_argument("--instance-prefix", default="omega")
    ap.add_argument("--go2rtc", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--go2rtc-mode", default="fifo")
    ap.add_argument("--go2rtc-url", default="http://127.0.0.1:1984")
    ap.add_argument("--stream-backend", default=os.environ.get("METABONK_STREAM_BACKEND", "auto"))
    ap.add_argument("--game-dir", default=os.environ.get("MEGABONK_GAME_DIR", ""))
    ap.add_argument("--output", default="")
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min-return-delta", type=float, default=5.0)
    ap.add_argument("--min-length-delta", type=float, default=50.0)
    ap.add_argument("--p-value", type=float, default=0.1)
    ap.add_argument("--capture-duration-s", type=float, default=180.0)
    args = ap.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise RuntimeError("--seeds must include at least one seed")
    out_dir = Path(args.output) if args.output else (REPO_ROOT / "runs" / f"{_timestamp()}__e2e_proof")
    cfg = ProofConfig(
        workers=int(args.workers),
        seeds=seeds,
        eval_timeout_s=float(args.eval_timeout_s),
        train_seconds=float(args.train_seconds),
        train_steps=int(args.train_steps),
        orch_port=int(args.orch_port),
        worker_base_port=int(args.worker_base_port),
        instance_prefix=str(args.instance_prefix),
        go2rtc=bool(args.go2rtc),
        go2rtc_mode=str(args.go2rtc_mode),
        go2rtc_url=str(args.go2rtc_url),
        stream_backend=str(args.stream_backend),
        output_dir=out_dir,
        game_dir=str(args.game_dir),
        dry_run=bool(args.dry_run),
        min_return_delta=float(args.min_return_delta),
        min_length_delta=float(args.min_length_delta),
        p_value=float(args.p_value),
        capture_duration_s=float(args.capture_duration_s),
    )

    if cfg.train_seconds <= 0 and cfg.train_steps <= 0:
        raise RuntimeError("train-seconds or train-steps must be > 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "video").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    report_lines: List[str] = ["# MetaBonk Learning Proof Report", ""]

    if cfg.dry_run:
        return _dry_run(cfg)

    if not cfg.game_dir:
        raise RuntimeError("--game-dir or MEGABONK_GAME_DIR must be set for real proof run")

    env = _build_env(cfg)
    env["METABONK_RUN_ID"] = f"e2e-proof-{_timestamp()}"
    env["METABONK_EXPERIMENT_ID"] = "e2e_learning_proof"

    # Static audit.
    audit_ok = _static_audit(report_lines)

    # Start go2rtc if requested (FIFO/exec).
    go2rtc_started = False
    go2rtc_external = False
    if cfg.go2rtc:
        fifo_dir = str(env.get("METABONK_STREAM_FIFO_DIR") or (REPO_ROOT / "temp" / "streams"))
        env["METABONK_STREAM_FIFO_DIR"] = fifo_dir
        env["METABONK_FIFO_STREAM"] = "1"
        env["METABONK_GO2RTC_URL"] = cfg.go2rtc_url
        cfg_path = REPO_ROOT / "temp" / "go2rtc.yaml"
        subprocess.check_call(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "go2rtc_generate_config.py"),
                "--workers",
                str(cfg.workers),
                "--instance-prefix",
                cfg.instance_prefix,
                "--mode",
                cfg.go2rtc_mode,
                "--fifo-dir",
                fifo_dir,
                "--out",
                str(cfg_path),
            ],
            cwd=str(REPO_ROOT),
            env=env,
        )
        env["METABONK_GO2RTC_CONFIG"] = str(cfg_path)
        try:
            url = cfg.go2rtc_url.rstrip("/") + "/api"
            go2rtc_external = bool(requests.get(url, timeout=1.0).ok)
        except Exception:
            go2rtc_external = False
        if go2rtc_external:
            go2rtc_started = True
        else:
            compose = ["docker", "compose", "-f", str(REPO_ROOT / "docker" / ("docker-compose.go2rtc.exec.yml" if cfg.go2rtc_mode == "exec" else "docker-compose.go2rtc.yml"))]
            subprocess.check_call(compose + ["up", "-d", "--remove-orphans"], cwd=str(REPO_ROOT), env=env)
            go2rtc_started = True

    omega_log = out_dir / "logs" / "start_omega.log"
    env["METABONK_RUN_DIR"] = str(out_dir)
    game_log_dir = out_dir / "logs" / "game"
    game_log_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MEGABONK_LOG_DIR", str(game_log_dir))
    proton_log_dir = out_dir / "logs" / "proton"
    proton_crash_dir = out_dir / "logs" / "proton_crash"
    proton_log_dir.mkdir(parents=True, exist_ok=True)
    proton_crash_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("METABONK_PROTON_LOG", "1")
    env.setdefault("METABONK_PROTON_LOG_DIR", str(proton_log_dir))
    env.setdefault("METABONK_PROTON_CRASH_DIR", str(proton_crash_dir))
    env.setdefault("METABONK_WINEDEBUG", "-all,+seh,+tid,+timestamp,+loaddll")
    gamescope_enabled = _env_truthy(os.environ.get("METABONK_E2E_GAMESCOPE", "1"))
    omega_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "start_omega.py"),
        "--mode",
        "train",
        "--workers",
        str(cfg.workers),
        "--orch-port",
        str(cfg.orch_port),
        "--worker-base-port",
        str(cfg.worker_base_port),
        "--instance-prefix",
        cfg.instance_prefix,
        "--stream-backend",
        cfg.stream_backend,
        "--game-dir",
        cfg.game_dir,
    ]
    if gamescope_enabled:
        omega_cmd.append("--gamescope")
    else:
        omega_cmd.append("--no-gamescope")
    omega = _run(
        omega_cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=omega_log,
    )

    eval_history_path = Path(env["METABONK_EVAL_HISTORY_PATH"])
    metrics_rows: List[Dict[str, Any]] = []
    exit_code = 1
    passed = False

    try:
        workers = _wait_for_workers(cfg, timeout_s=240.0)
        _run_stream_diagnostics(cfg, out_dir / "logs" / "stream_diagnostics.log", env)
        _validate_streams(cfg, workers)
        report_lines.append(f"- Workers online: {len(workers)}")
        report_lines.append(f"- Stream backend: {env.get('METABONK_STREAM_BACKEND')}")
        report_lines.append(f"- Input backend: {env.get('METABONK_INPUT_BACKEND')}")
        if go2rtc_started:
            report_lines.append(f"- go2rtc: on{' (external)' if go2rtc_external else ''}")
        else:
            report_lines.append("- go2rtc: off")
        eval_instance_id = sorted(workers.keys())[0]

        # Baseline: random policy
        random_video = out_dir / "video" / "baseline_random_seeded.mp4"
        random_metrics, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofRandom",
            action_source="random",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="baseline_random",
            video_out=random_video,
        )
        _append_eval_rows("baseline_random", random_metrics, metrics_rows)

        # Baseline: untrained policy (pre)
        untrained_video = out_dir / "video" / "baseline_untrained.mp4"
        pre_metrics, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofTrain",
            action_source="policy",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="pre_train",
            video_out=untrained_video,
        )
        _append_eval_rows("pre_train", pre_metrics, metrics_rows)

        # Train main policy
        _train_phase(
            cfg=cfg,
            workers=workers,
            policy_name="ProofTrain",
            action_source="policy",
            phase_label="train",
            metrics_rows=metrics_rows,
        )

        # Post-training eval
        trained_video = out_dir / "video" / "trained_after.mp4"
        post_metrics, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofTrain",
            action_source="policy",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="post_train",
            video_out=trained_video,
        )
        _append_eval_rows("post_train", post_metrics, metrics_rows)

        # Reward shuffle negative control
        shuffle_video = out_dir / "video" / "negative_control_reward_shuffle.mp4"
        shuffle_pre, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofShuffle",
            action_source="policy",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="shuffle_pre",
            video_out=shuffle_video,
        )
        _append_eval_rows("shuffle_pre", shuffle_pre, metrics_rows)
        _train_phase(
            cfg=cfg,
            workers=workers,
            policy_name="ProofShuffle",
            action_source="policy",
            phase_label="shuffle_train",
            metrics_rows=metrics_rows,
        )
        shuffle_post, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofShuffle",
            action_source="policy",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="shuffle_post",
            video_out=None,
        )
        _append_eval_rows("shuffle_post", shuffle_post, metrics_rows)

        # Frozen weights negative control
        frozen_video = out_dir / "video" / "negative_control_frozen_weights.mp4"
        frozen_pre, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofFrozen",
            action_source="policy",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="frozen_pre",
            video_out=frozen_video,
        )
        _append_eval_rows("frozen_pre", frozen_pre, metrics_rows)
        _train_phase(
            cfg=cfg,
            workers=workers,
            policy_name="ProofFrozen",
            action_source="policy",
            phase_label="frozen_train",
            metrics_rows=metrics_rows,
        )
        frozen_post, _ = _run_eval_phase(
            cfg=cfg,
            instance_id=eval_instance_id,
            workers=workers,
            policy_name="ProofFrozen",
            action_source="policy",
            seeds=cfg.seeds,
            eval_history_path=eval_history_path,
            phase_label="frozen_post",
            video_out=None,
        )
        _append_eval_rows("frozen_post", frozen_post, metrics_rows)

        # Side-by-side montage
        montage = out_dir / "video" / "side_by_side_before_after.mp4"
        subprocess.check_call(build_hstack_cmd(untrained_video, trained_video, montage))

        # Stats
        pre_returns = [float(r.get("mean_return") or 0.0) for r in pre_metrics]
        post_returns = [float(r.get("mean_return") or 0.0) for r in post_metrics]
        pre_lengths = [float(r.get("mean_length") or 0.0) for r in pre_metrics]
        post_lengths = [float(r.get("mean_length") or 0.0) for r in post_metrics]

        mean_pre, ci_pre_lo, ci_pre_hi = mean_ci(pre_returns)
        mean_post, ci_post_lo, ci_post_hi = mean_ci(post_returns)
        delta = mean_post - mean_pre
        pval = paired_permutation_test(pre_returns, post_returns)
        d = cohens_d(pre_returns, post_returns)

        mean_pre_len, _, _ = mean_ci(pre_lengths)
        mean_post_len, _, _ = mean_ci(post_lengths)
        delta_len = mean_post_len - mean_pre_len

        report_lines += [
            "",
            "## Results",
            f"- ProofTrain mean return: pre {mean_pre:.2f} (CI {ci_pre_lo:.2f}-{ci_pre_hi:.2f})",
            f"- ProofTrain mean return: post {mean_post:.2f} (CI {ci_post_lo:.2f}-{ci_post_hi:.2f})",
            f"- Return delta: {delta:.2f}; p-value {pval:.4f}; Cohen d {d:.2f}",
            f"- Mean episode length delta: {delta_len:.2f}",
        ]

        # Negative controls summary
        def _mean(vals: List[Dict[str, Any]]) -> float:
            return float(sum(float(v.get("mean_return") or 0.0) for v in vals) / max(1, len(vals)))

        report_lines += [
            "",
            "## Negative controls",
            f"- Reward shuffle pre/post mean returns: {_mean(shuffle_pre):.2f} -> {_mean(shuffle_post):.2f}",
            f"- Frozen weights pre/post mean returns: {_mean(frozen_pre):.2f} -> {_mean(frozen_post):.2f}",
        ]

        # Plots
        eval_plot = out_dir / "plots" / "eval_returns.svg"
        _write_svg_bar(
            eval_plot,
            ["pre", "post"],
            [mean_pre, mean_post],
            "ProofTrain mean return",
        )
        reward_curve = out_dir / "plots" / "reward_curve.svg"
        curve = [(row["ts"], float(row.get("reward") or 0.0)) for row in metrics_rows if row.get("phase") == "train"]
        _write_svg_line(reward_curve, curve[-200:], "Training reward (latest)")

        # Metrics CSV
        _write_metrics_csv(out_dir / "metrics.csv", metrics_rows)

        # Report videos
        report_lines += [
            "",
            "## Videos",
            f"- baseline_random_seeded.mp4: {random_video}",
            f"- baseline_untrained.mp4: {untrained_video}",
            f"- trained_after.mp4: {trained_video}",
            f"- side_by_side_before_after.mp4: {montage}",
            f"- negative_control_reward_shuffle.mp4: {shuffle_video}",
            f"- negative_control_frozen_weights.mp4: {frozen_video}",
        ]

        # Guard violations
        guard_dir = Path(env.get("METABONK_ACTION_GUARD_PATH", "") or "temp/action_guard_violations")
        guard_hits = list(guard_dir.glob("*.txt")) if guard_dir.exists() else []
        if guard_hits:
            report_lines.append("\n## Action guard violations")
            for p in guard_hits:
                report_lines.append(f"- {p}: {p.read_text(errors='replace')}")

        # Pass criteria
        passed = True
        if delta < cfg.min_return_delta:
            passed = False
            report_lines.append(f"- FAIL: return delta {delta:.2f} < {cfg.min_return_delta:.2f}")
        if delta_len < cfg.min_length_delta:
            passed = False
            report_lines.append(f"- FAIL: length delta {delta_len:.2f} < {cfg.min_length_delta:.2f}")
        if pval > cfg.p_value:
            passed = False
            report_lines.append(f"- FAIL: p-value {pval:.4f} > {cfg.p_value:.4f}")
        if guard_hits:
            passed = False
            report_lines.append("- FAIL: action-source guard violations detected")
        if not audit_ok:
            passed = False
            report_lines.append("- FAIL: static audit violations detected")

        report_lines.append("\n## Status")
        report_lines.append("PASS" if passed else "FAIL")

        (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n")

        # Manifest
        manifest = {
            "git": _git_info(),
            "config": {
                "workers": cfg.workers,
                "seeds": cfg.seeds,
                "train_seconds": cfg.train_seconds,
                "train_steps": cfg.train_steps,
                "eval_timeout_s": cfg.eval_timeout_s,
                "capture_duration_s": cfg.capture_duration_s,
                "min_return_delta": cfg.min_return_delta,
                "min_length_delta": cfg.min_length_delta,
                "p_value": cfg.p_value,
                "stream_backend": cfg.stream_backend,
                "instance_prefix": cfg.instance_prefix,
                "go2rtc": cfg.go2rtc,
                "go2rtc_mode": cfg.go2rtc_mode,
            },
            "env": {
                "METABONK_INPUT_BACKEND": env.get("METABONK_INPUT_BACKEND"),
                "METABONK_INPUT_BUTTONS": env.get("METABONK_INPUT_BUTTONS") or env.get("METABONK_BUTTON_KEYS"),
                "METABONK_STREAM_BACKEND": env.get("METABONK_STREAM_BACKEND"),
                "METABONK_REQUIRE_CUDA": env.get("METABONK_REQUIRE_CUDA"),
            },
            "seeds": cfg.seeds,
            "versions": _version_info(),
            "gpu": _gpu_info(),
            "artifacts": hash_artifacts(out_dir.rglob("*")),
            "passed": passed,
        }
        write_manifest(out_dir / "manifest.json", manifest)
        exit_code = 0 if passed else 1
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        report_lines.append("")
        report_lines.append("## Root cause + next action")
        report_lines.append(f"- {err}")
        report_lines.append("- Review logs in logs/ and ensure GPU streaming + input backend are configured.")
        (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n")
        manifest = {
            "git": _git_info(),
            "config": {
                "workers": cfg.workers,
                "seeds": cfg.seeds,
                "train_seconds": cfg.train_seconds,
                "train_steps": cfg.train_steps,
                "eval_timeout_s": cfg.eval_timeout_s,
                "capture_duration_s": cfg.capture_duration_s,
                "min_return_delta": cfg.min_return_delta,
                "min_length_delta": cfg.min_length_delta,
                "p_value": cfg.p_value,
                "stream_backend": cfg.stream_backend,
                "instance_prefix": cfg.instance_prefix,
                "go2rtc": cfg.go2rtc,
                "go2rtc_mode": cfg.go2rtc_mode,
            },
            "env": {
                "METABONK_INPUT_BACKEND": env.get("METABONK_INPUT_BACKEND"),
                "METABONK_INPUT_BUTTONS": env.get("METABONK_INPUT_BUTTONS") or env.get("METABONK_BUTTON_KEYS"),
                "METABONK_STREAM_BACKEND": env.get("METABONK_STREAM_BACKEND"),
                "METABONK_REQUIRE_CUDA": env.get("METABONK_REQUIRE_CUDA"),
            },
            "seeds": cfg.seeds,
            "versions": _version_info(),
            "gpu": _gpu_info(),
            "artifacts": hash_artifacts(out_dir.rglob("*")),
            "passed": False,
            "error": err,
        }
        write_manifest(out_dir / "manifest.json", manifest)
        exit_code = 1
    finally:
        _terminate(omega)
        if go2rtc_started:
            try:
                compose = ["docker", "compose", "-f", str(REPO_ROOT / "docker" / ("docker-compose.go2rtc.exec.yml" if cfg.go2rtc_mode == "exec" else "docker-compose.go2rtc.yml"))]
                subprocess.check_call(compose + ["down", "--remove-orphans"], cwd=str(REPO_ROOT), env=env)
            except Exception:
                pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
