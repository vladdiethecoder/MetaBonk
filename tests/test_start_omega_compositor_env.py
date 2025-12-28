from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType


def _load_start_omega_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = repo_root / "scripts" / "start_omega.py"
    spec = importlib.util.spec_from_file_location("_metabonk_start_omega", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wait_for_compositor_env_respects_mtime_guard(tmp_path: Path) -> None:
    mod = _load_start_omega_module()
    env_path = tmp_path / "compositor.env"
    env_path.write_text("XDG_RUNTIME_DIR=/tmp/a\nWAYLAND_DISPLAY=wl-a\nDISPLAY=:4\n", encoding="utf-8")
    baseline_ns = env_path.stat().st_mtime_ns

    def _write_fresh() -> None:
        time.sleep(0.05)
        env_path.write_text("XDG_RUNTIME_DIR=/tmp/b\nWAYLAND_DISPLAY=wl-b\nDISPLAY=:3\n", encoding="utf-8")
        # Ensure mtime advances even on coarse timestamp filesystems.
        try:
            os.utime(env_path, ns=(baseline_ns + 2_000_000_000, baseline_ns + 2_000_000_000))
        except Exception:
            pass

    t = threading.Thread(target=_write_fresh, daemon=True)
    t.start()
    kv = mod._wait_for_compositor_env(env_path, 2.0, min_mtime_ns=baseline_ns)
    assert kv.get("DISPLAY") == ":3"
    assert kv.get("WAYLAND_DISPLAY") == "wl-b"
    assert kv.get("XDG_RUNTIME_DIR") == "/tmp/b"
