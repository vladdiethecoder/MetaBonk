import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_launch_py_compiles():
    result = subprocess.run(["python3", "-m", "py_compile", str(REPO_ROOT / "launch.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launch_help_works():
    result = subprocess.run(["python3", str(REPO_ROOT / "launch.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "MetaBonk unified launcher" in result.stdout


def test_launch_wrapper_exists():
    launch = REPO_ROOT / "launch"
    assert launch.exists()
    assert os.access(launch, os.X_OK), "launch wrapper should be executable"


def test_launcher_profiles_are_valid_json():
    for name in ("default", "production"):
        path = REPO_ROOT / "configs" / f"launch_{name}.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "workers" in data
        assert "cognitive_server" in data


def test_stream_route_present_in_frontend():
    app_tsx = REPO_ROOT / "src" / "frontend" / "src" / "App.tsx"
    stream_page = REPO_ROOT / "src" / "frontend" / "src" / "pages" / "Stream.tsx"
    assert app_tsx.exists()
    assert stream_page.exists()
    content = app_tsx.read_text(encoding="utf-8")
    assert 'to: "/stream"' in content or '"/stream"' in content
