#!/usr/bin/env python3
"""MetaBonk unified launcher.

Goal: one command (`./launch`) to bring up:
  - Cognitive server (docker compose)
  - Omega stack + Vite UI (scripts/start.py)
  - Optional live terminal dashboard

This intentionally reuses existing MetaBonk scripts (start_cognitive_server.sh,
scripts/start.py, scripts/stop.py) so the launcher stays thin and robust.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover
    zmq = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


BANNER = f"""{Colors.CYAN}
╔════════════════════════════════════════════════════════════╗
║                         METABONK                           ║
║                 Unified Training Launcher                  ║
╚════════════════════════════════════════════════════════════╝
{Colors.END}"""


def _truthy(v: object) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object at root in {path}")
    return raw


def _docker_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return env for docker commands; prefer system docker when DOCKER_HOST points to podman."""
    env = dict(base or os.environ)
    dh = str(env.get("DOCKER_HOST") or "")
    if dh and "podman" in dh and Path("/var/run/docker.sock").exists():
        env.pop("DOCKER_HOST", None)
    return env


def _compose_cmd() -> list[str]:
    """Resolve compose command based on METABONK_DOCKER_COMPOSE (docker|docker-compose)."""
    compose = str(os.environ.get("METABONK_DOCKER_COMPOSE") or "docker").strip()
    if compose == "docker":
        return ["docker", "compose"]
    return [compose]


def _run(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
    )


def _docker_rm_if_exists(name: str) -> None:
    """Force-remove a container by name if it exists (best-effort)."""
    env = _docker_env()
    try:
        r = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        exists = r.returncode == 0 and any(ln.strip() == name for ln in (r.stdout or "").splitlines())
    except Exception:
        exists = False
    if not exists:
        return
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=15, env=env, check=False)
    except Exception:
        pass


def _nvidia_smi_ok() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _docker_ok() -> bool:
    try:
        r = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=5, env=_docker_env())
        return r.returncode == 0
    except Exception:
        return False


def _docker_gpu_ok() -> bool:
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:13.1.0-base-ubuntu24.04", "nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            env=_docker_env(),
        )
        return r.returncode == 0
    except Exception:
        return False


def _probe_http(url: str, *, timeout_s: float = 1.5) -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(url, timeout=max(0.2, float(timeout_s)))
        return bool(r.ok)
    except Exception:
        return False


def _zmq_metrics(server_url: str, *, timeout_s: float = 1.0) -> Optional[Dict[str, Any]]:
    if zmq is None:
        return None
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.linger = 0
    sock.setsockopt_string(zmq.IDENTITY, "metabonk-launch")
    sock.connect(server_url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        try:
            sock.send_json({"type": "metrics", "timestamp": time.time()}, flags=zmq.NOBLOCK)
        except Exception:
            return None
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            socks = dict(poller.poll(timeout=50))
            if sock not in socks:
                continue
            try:
                data = sock.recv(flags=zmq.NOBLOCK)
                obj = json.loads(data.decode("utf-8"))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None
    finally:
        try:
            sock.close(0)
        except Exception:
            pass


@dataclass
class LaunchResult:
    run_id: str
    run_dir: Path
    start_log: Path


def _default_config() -> Dict[str, Any]:
    return {
        "mode": "train",
        "workers": 5,
        "gamescope": {"width": 1280, "height": 720},
        "ui": {"enabled": True, "host": "127.0.0.1", "port": 5173, "install": False},
        "streaming": {"profile": "rtx5090_webrtc_8", "go2rtc": True},
        "training": {
            "pure_vision_mode": True,
            "exploration_rewards": True,
            "rl_logging": True,
            "strategy_frequency": 2.0,
        },
        "system2": {
            "enabled": True,
            "server_url": "tcp://127.0.0.1:5555",
        },
        "cognitive_server": {
            "enabled": True,
            # These map directly to docker/docker-compose.cognitive.yml environment vars.
            "env": {
                "METABONK_COGNITIVE_BACKEND": "sglang",
                "METABONK_COGNITIVE_MODEL_PATH": "/models/Phi_3_Vision_128k_Instruct_AWQ_4bit",
                "METABONK_COGNITIVE_QUANTIZATION": "awq_marlin",
                "METABONK_COGNITIVE_CONTEXT_LEN": "2048",
                "METABONK_COGNITIVE_MAX_REQS": "1",
                "METABONK_COGNITIVE_MAX_CONCURRENCY": "1",
                "METABONK_COGNITIVE_MAX_NEW_TOKENS": "12",
                "METABONK_COGNITIVE_ALWAYS_SINGLE_FRAME": "1",
                "METABONK_COGNITIVE_TILE_EDGE": "64",
                "METABONK_COGNITIVE_STUCK_SINGLE_FRAME": "1",
                "METABONK_COGNITIVE_TILE_EDGE_STUCK": "64",
                "METABONK_COGNITIVE_SGLANG_MEM_FRACTION": "0.5",
                "METABONK_COGNITIVE_SGLANG_DISABLE_CUDA_GRAPH": "1",
            },
            "compose_project": "metabonk-cognitive",
        },
        "monitoring": {"enabled": True, "interval_s": 1.0},
        "validation": {"enabled": True, "timeout_s": 2.0},
        "optimize_5090": True,
    }


def _find_config_path(name: str) -> Optional[Path]:
    name = str(name or "").strip()
    if not name:
        return None
    candidates = [
        REPO_ROOT / "configs" / f"launch_{name}.json",
        REPO_ROOT / "config" / f"launch_{name}.json",
        REPO_ROOT / f"launch_{name}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(*, profile: str, config_file: Optional[str]) -> Dict[str, Any]:
    cfg = _default_config()
    path = Path(config_file).expanduser().resolve() if config_file else _find_config_path(profile)
    if path and path.exists():
        cfg = _deep_merge(cfg, _load_json(path))
    return cfg


class MetaBonkLauncher:
    def __init__(self, cfg: Dict[str, Any], *, no_dashboard: bool, skip_stop: bool):
        self.cfg = cfg
        self.no_dashboard = bool(no_dashboard)
        self.skip_stop = bool(skip_stop)
        self._start_proc: Optional[subprocess.Popen] = None
        self._started_cognitive = False
        self._launch_result: Optional[LaunchResult] = None

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        print(f"\n{Colors.YELLOW}[launch] received signal {signum}; shutting down...{Colors.END}")
        self.stop()
        raise SystemExit(0)

    def _ensure_game_dir_env(self, env: Dict[str, str]) -> None:
        default_dir = "/run/media/vdubrov/Active Storage/SteamLibrary/steamapps/common/Megabonk"
        if not str(env.get("MEGABONK_GAME_DIR") or "").strip() and Path(default_dir).is_dir():
            env["MEGABONK_GAME_DIR"] = default_dir

    def check_prereqs(self) -> None:
        print(f"{Colors.BOLD}Prerequisites{Colors.END}")

        if not _nvidia_smi_ok():
            raise SystemExit("[launch] ERROR: nvidia-smi not working (MetaBonk is GPU-only).")
        print(f"  {Colors.GREEN}✓{Colors.END} GPU detected (nvidia-smi)")

        if self.cfg.get("cognitive_server", {}).get("enabled", True):
            if not _docker_ok():
                raise SystemExit("[launch] ERROR: docker not working (required for cognitive server).")
            print(f"  {Colors.GREEN}✓{Colors.END} docker available")

            if not _docker_gpu_ok():
                raise SystemExit("[launch] ERROR: docker --gpus all failed (nvidia-container-toolkit not configured).")
            print(f"  {Colors.GREEN}✓{Colors.END} docker GPU runtime ok")

        if self.cfg.get("ui", {}).get("enabled", True):
            if not shutil.which("npm"):
                raise SystemExit("[launch] ERROR: npm not found (required for the Vite UI).")
            print(f"  {Colors.GREEN}✓{Colors.END} npm available")

        env = dict(os.environ)
        self._ensure_game_dir_env(env)
        if not str(env.get("MEGABONK_GAME_DIR") or "").strip():
            raise SystemExit(
                "[launch] ERROR: MEGABONK_GAME_DIR not set and default Steam path not found.\n"
                "[launch]        Set MEGABONK_GAME_DIR to your Megabonk install directory."
            )
        if not Path(env["MEGABONK_GAME_DIR"]).exists():
            raise SystemExit(f"[launch] ERROR: game dir not found: {env['MEGABONK_GAME_DIR']}")
        print(f"  {Colors.GREEN}✓{Colors.END} game dir: {env['MEGABONK_GAME_DIR']}")

        # Optional python deps for monitoring/validation.
        if requests is None:
            print(f"  {Colors.YELLOW}⚠{Colors.END} python requests not installed (monitoring/validation limited)")
        if zmq is None:
            print(f"  {Colors.YELLOW}⚠{Colors.END} python zmq not installed (cognitive metrics limited)")

    def stop(self) -> None:
        # Best-effort: stop omega stack (and go2rtc) using the repo's conservative stop script.
        try:
            _run([sys.executable, str(REPO_ROOT / "scripts" / "stop.py"), "--all"], cwd=REPO_ROOT, check=False)
        except Exception:
            pass

        # Stop cognitive server (best-effort; safe even if it wasn't started by this launcher).
        self._stop_cognitive_server()
        self._started_cognitive = False

        # Ensure launcher-managed process is terminated.
        if self._start_proc and self._start_proc.poll() is None:
            try:
                self._start_proc.terminate()
                self._start_proc.wait(timeout=5)
            except Exception:
                try:
                    self._start_proc.kill()
                except Exception:
                    pass

    def _start_cognitive_server(self, env: Dict[str, str]) -> None:
        if not self.cfg.get("cognitive_server", {}).get("enabled", True):
            return

        cs = self.cfg.get("cognitive_server", {}) or {}
        cs_env = cs.get("env", {}) or {}
        for k, v in cs_env.items():
            env[str(k)] = str(v)
        if cs.get("compose_project"):
            env["METABONK_COGNITIVE_COMPOSE_PROJECT"] = str(cs.get("compose_project"))

        script = REPO_ROOT / "scripts" / "start_cognitive_server.sh"
        if not script.exists():
            raise SystemExit("[launch] ERROR: scripts/start_cognitive_server.sh missing.")

        print(f"{Colors.BOLD}Cognitive Server{Colors.END}")
        # Defensive: a previous run may have left a stopped container with the fixed
        # container_name (metabonk-cognitive-server), which causes docker compose to fail
        # with a name conflict. Remove it proactively.
        cname = str(env.get("METABONK_COGNITIVE_CONTAINER") or os.environ.get("METABONK_COGNITIVE_CONTAINER") or "metabonk-cognitive-server")
        _docker_rm_if_exists(cname)
        subprocess.check_call([str(script)], cwd=str(REPO_ROOT), env=env)
        self._started_cognitive = True

    def _stop_cognitive_server(self) -> None:
        project = str(self.cfg.get("cognitive_server", {}).get("compose_project") or os.environ.get("METABONK_COGNITIVE_COMPOSE_PROJECT") or "metabonk-cognitive")
        cmd = _compose_cmd() + [
            "-p",
            project,
            "-f",
            str(REPO_ROOT / "docker" / "docker-compose.cognitive.yml"),
            "down",
            "--remove-orphans",
        ]
        env = _docker_env()
        try:
            _run(cmd, cwd=REPO_ROOT, env=env, check=False)
        except Exception:
            pass

        # Also try stopping by container name (defensive).
        cname = str(os.environ.get("METABONK_COGNITIVE_CONTAINER") or "metabonk-cognitive-server")
        try:
            subprocess.run(["docker", "stop", cname], cwd=str(REPO_ROOT), env=env, check=False, capture_output=True, text=True)
        except Exception:
            pass
        _docker_rm_if_exists(cname)

    def _spawn_start(self, env: Dict[str, str]) -> LaunchResult:
        run_id = f"run-launch-{int(time.time())}"
        env.setdefault("METABONK_RUN_ID", run_id)
        run_dir = REPO_ROOT / "runs" / run_id
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        start_log = logs_dir / "launch.log"

        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "start.py"),
            "--mode",
            str(self.cfg.get("mode") or "train"),
            "--workers",
            str(int(self.cfg.get("workers") or 5)),
            "--gamescope-width",
            str(int(self.cfg.get("gamescope", {}).get("width") or 1280)),
            "--gamescope-height",
            str(int(self.cfg.get("gamescope", {}).get("height") or 720)),
        ]

        stream_profile = str(self.cfg.get("streaming", {}).get("profile") or "").strip()
        if stream_profile:
            cmd += ["--stream-profile", stream_profile]

        # UI
        ui_cfg = self.cfg.get("ui", {}) or {}
        if not bool(ui_cfg.get("enabled", True)):
            cmd += ["--no-ui"]
        else:
            cmd += ["--ui-host", str(ui_cfg.get("host") or "127.0.0.1"), "--ui-port", str(int(ui_cfg.get("port") or 5173))]
            if bool(ui_cfg.get("install", False)):
                cmd += ["--ui-install"]

        # go2rtc
        go2rtc_on = bool(self.cfg.get("streaming", {}).get("go2rtc", True))
        cmd += ["--go2rtc" if go2rtc_on else "--no-go2rtc"]

        # Performance preset
        if bool(self.cfg.get("optimize_5090", True)):
            cmd += ["--optimize-5090"]

        print(f"{Colors.BOLD}Omega + UI{Colors.END}")
        print(f"[launch] starting: {' '.join(cmd)}")
        print(f"[launch] logs: {start_log}")

        with open(start_log, "ab", buffering=0) as f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._start_proc = proc
        return LaunchResult(run_id=run_id, run_dir=run_dir, start_log=start_log)

    def _apply_training_env(self, env: Dict[str, str]) -> None:
        tr = self.cfg.get("training", {}) or {}
        env["METABONK_PURE_VISION_MODE"] = "1" if bool(tr.get("pure_vision_mode", True)) else "0"
        env["METABONK_EXPLORATION_REWARDS"] = "1" if bool(tr.get("exploration_rewards", True)) else "0"
        env["METABONK_RL_LOGGING"] = "1" if bool(tr.get("rl_logging", True)) else "0"
        env["METABONK_STRATEGY_FREQUENCY"] = str(tr.get("strategy_frequency", 2.0))

        sys2 = self.cfg.get("system2", {}) or {}
        env["METABONK_SYSTEM2_ENABLED"] = "1" if bool(sys2.get("enabled", True)) else "0"
        env["METABONK_COGNITIVE_SERVER_URL"] = str(sys2.get("server_url") or "tcp://127.0.0.1:5555")

    def _wait_for_ready(self, *, timeout_s: float) -> None:
        deadline = time.time() + max(1.0, float(timeout_s))
        ui = self.cfg.get("ui", {}) or {}
        ui_url = f"http://{ui.get('host','127.0.0.1')}:{int(ui.get('port',5173))}/"
        orch_url = "http://127.0.0.1:8040/workers"
        cog_url = str(self.cfg.get("system2", {}).get("server_url") or "tcp://127.0.0.1:5555")

        while time.time() < deadline:
            if self._start_proc and self._start_proc.poll() is not None:
                raise SystemExit(f"[launch] ERROR: scripts/start.py exited early with code {self._start_proc.returncode}")

            ok_ui = (not bool(ui.get("enabled", True))) or _probe_http(ui_url, timeout_s=0.75)
            ok_orch = _probe_http(orch_url, timeout_s=0.75)
            ok_cog = True
            if bool(self.cfg.get("cognitive_server", {}).get("enabled", True)):
                ok_cog = _zmq_metrics(cog_url, timeout_s=0.75) is not None

            if ok_ui and ok_orch and ok_cog:
                return
            time.sleep(0.5)

    def _validate(self) -> None:
        if not bool(self.cfg.get("validation", {}).get("enabled", True)):
            return
        timeout_s = float(self.cfg.get("validation", {}).get("timeout_s", 2.0) or 2.0)
        workers = int(self.cfg.get("workers") or 0)
        cog_url = str(self.cfg.get("system2", {}).get("server_url") or "tcp://127.0.0.1:5555")

        print(f"{Colors.BOLD}Validation{Colors.END}")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_deployment.py"),
            "--cognitive-url",
            cog_url,
            "--workers",
            str(min(workers, 5)),
            "--timeout-s",
            str(timeout_s),
        ]
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
        if r.returncode != 0:
            print(f"{Colors.YELLOW}[launch] validation returned {r.returncode} (see above).{Colors.END}")

        # Quick UI checks (best-effort).
        ui_cfg = self.cfg.get("ui", {}) or {}
        if bool(ui_cfg.get("enabled", True)) and requests is not None:
            base = f"http://{ui_cfg.get('host','127.0.0.1')}:{int(ui_cfg.get('port',5173))}"
            for path in ("/stream", "/neural/broadcast"):
                ok = _probe_http(base + path, timeout_s=1.0)
                print(f"{'✅' if ok else '⚠️ '} UI {path} -> {base + path}")

    def _dashboard(self) -> None:
        if self.no_dashboard or not bool(self.cfg.get("monitoring", {}).get("enabled", True)):
            # Just wait for start.py to exit.
            if self._start_proc:
                self._start_proc.wait()
            return

        interval_s = float(self.cfg.get("monitoring", {}).get("interval_s", 1.0) or 1.0)
        ui_cfg = self.cfg.get("ui", {}) or {}
        base_ui = f"http://{ui_cfg.get('host','127.0.0.1')}:{int(ui_cfg.get('port',5173))}"
        cog_url = str(self.cfg.get("system2", {}).get("server_url") or "tcp://127.0.0.1:5555")

        while True:
            if self._start_proc and self._start_proc.poll() is not None:
                print(f"\n[launch] scripts/start.py exited with code {self._start_proc.returncode}")
                return

            print("\033[2J\033[H", end="")
            print(BANNER)
            print(f"{Colors.BOLD}Status{Colors.END}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if self._launch_result:
                print(f"run_id={self._launch_result.run_id}  logs={self._launch_result.start_log}")
            print("-" * 60)

            # GPU
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                util_s, used_s, total_s = [p.strip() for p in out.split(",")[:3]]
                util = int(util_s)
                used = float(used_s) / 1024.0
                total = float(total_s) / 1024.0
                pct = (used / max(0.001, total)) * 100.0
                print(f"GPU: {util:3d}% util  {used:5.1f}/{total:5.1f} GB ({pct:3.0f}%)")
            except Exception:
                print("GPU: (unavailable)")

            # Cognitive metrics
            m = _zmq_metrics(cog_url, timeout_s=0.25) if bool(self.cfg.get("cognitive_server", {}).get("enabled", True)) else None
            if m:
                try:
                    req = int(m.get("request_count") or 0)
                    avg = float(m.get("avg_latency_ms") or 0.0)
                    agents = int(m.get("active_agents") or 0)
                    print(f"System2: {avg:4.0f}ms avg  requests={req}  agents={agents}")
                except Exception:
                    print("System2: (metrics parse error)")
            else:
                print("System2: (no metrics)")

            # Workers
            orch_ok = False
            workers = []
            if requests is not None:
                try:
                    r = requests.get("http://127.0.0.1:8040/workers", timeout=0.5)
                    if r.ok:
                        orch_ok = True
                        workers = (r.json() or {}).get("workers") or []
                except Exception:
                    pass
            if orch_ok:
                print(f"Workers: {len(workers)}/{int(self.cfg.get('workers') or 0)}")
                for w in workers[: min(8, len(workers))]:
                    try:
                        wid = str(w.get("instance_id") or "?")
                        port = int(w.get("port") or 0)
                        fps = None
                        scene = None
                        if port and requests is not None:
                            st = requests.get(f"http://127.0.0.1:{port}/status", timeout=0.35).json()
                            fps = float(st.get("frames_fps") or 0.0)
                            scene = ((st.get("system2_reasoning") or {}) if isinstance(st.get("system2_reasoning"), dict) else {}).get("scene_type")
                        fps_s = f"{fps:.0f}" if fps is not None else "?"
                        scene_s = str(scene or "unknown")
                        print(f"  {wid:<10} fps={fps_s:>3}  scene={scene_s}")
                    except Exception:
                        continue
            else:
                print("Workers: (orchestrator unavailable)")

            print("-" * 60)
            print(f"UI: {base_ui}/stream  {base_ui}/neural/broadcast")
            print("Ctrl+C to stop")
            time.sleep(max(0.2, interval_s))

    def start(self) -> int:
        print(BANNER)

        self.check_prereqs()

        env = dict(os.environ)
        self._ensure_game_dir_env(env)
        self._apply_training_env(env)

        if not self.skip_stop:
            print(f"{Colors.BOLD}Cleanup{Colors.END}")
            _run([sys.executable, str(REPO_ROOT / "scripts" / "stop.py"), "--all"], cwd=REPO_ROOT, check=False)

        # Cognitive server first (so System2 can start serving quickly).
        self._start_cognitive_server(env)

        # Start omega/ui.
        self._launch_result = self._spawn_start(env)

        # Wait for basic readiness before entering dashboard.
        try:
            self._wait_for_ready(timeout_s=120)
        except SystemExit:
            self.stop()
            raise

        self._validate()

        # Main loop (dashboard or passive wait).
        self._dashboard()

        # If we get here, start.py exited; clean up.
        self.stop()
        return 0


def cmd_stop(cfg: Dict[str, Any]) -> int:
    launcher = MetaBonkLauncher(cfg, no_dashboard=True, skip_stop=True)
    launcher.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MetaBonk unified launcher (cognitive + omega + UI).")
    ap.add_argument("command", nargs="?", default="start", choices=["start", "stop"], help="start (default) or stop")
    ap.add_argument("--config", default="default", help="Config profile name (loads configs/launch_<name>.json)")
    ap.add_argument("--config-file", default=None, help="Path to a JSON config file")
    ap.add_argument("--workers", type=int, default=None, help="Override worker count")
    ap.add_argument("--mode", choices=["train", "play", "dream", "watch"], default=None, help="Override mode")
    ap.add_argument("--no-dashboard", action="store_true", help="Disable terminal dashboard")
    ap.add_argument("--skip-stop", action="store_true", help="Skip initial cleanup (scripts/stop.py --all)")
    args = ap.parse_args()

    cfg = load_config(profile=str(args.config), config_file=args.config_file)

    if args.workers is not None:
        cfg["workers"] = int(args.workers)
    if args.mode is not None:
        cfg["mode"] = str(args.mode)

    if args.command == "stop":
        return cmd_stop(cfg)

    launcher = MetaBonkLauncher(cfg, no_dashboard=bool(args.no_dashboard), skip_stop=bool(args.skip_stop))
    return launcher.start()


if __name__ == "__main__":
    raise SystemExit(main())
