#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ORCH_URL="${METABONK_SMOKE_ORCH_URL:-http://127.0.0.1:8040}"
WORKERS="${METABONK_SMOKE_WORKERS:-2}"
MODE="${METABONK_SMOKE_MODE:-train}"
WAIT_S="${METABONK_SMOKE_WAIT_S:-120}"
STREAM_REQUIRED="${METABONK_SMOKE_STREAM:-1}"
GO2RTC_ENABLED="${METABONK_SMOKE_GO2RTC:-1}"
GO2RTC_MODE="${METABONK_SMOKE_GO2RTC_MODE:-fifo}"
UI_ENABLED="${METABONK_SMOKE_UI:-0}"
INPUT_CHECK="${METABONK_SMOKE_INPUT:-0}"
FAILOVER_CHECK="${METABONK_SMOKE_FAILOVER:-0}"
GAME_FAILOVER_CHECK="${METABONK_SMOKE_GAME_FAILOVER:-0}"

LOG_DIR="${REPO_ROOT}/temp/smoke"
mkdir -p "${LOG_DIR}"
START_LOG="${LOG_DIR}/start.log"

cleanup() {
  python "${REPO_ROOT}/scripts/stop.py" --all --go2rtc || true
}
trap cleanup EXIT

CMD=("${REPO_ROOT}/start" "--mode" "${MODE}" "--workers" "${WORKERS}")
if [[ "${GO2RTC_ENABLED}" == "1" ]]; then
  CMD+=("--go2rtc" "--go2rtc-mode" "${GO2RTC_MODE}")
fi
if [[ "${UI_ENABLED}" != "1" ]]; then
  CMD+=("--no-ui")
fi

echo "[smoke] launching stack: ${CMD[*]}"
("${CMD[@]}" >"${START_LOG}" 2>&1) &
START_PID=$!

export METABONK_SMOKE_ORCH_URL="${ORCH_URL}"
export METABONK_SMOKE_WORKERS="${WORKERS}"
export METABONK_SMOKE_STREAM="${STREAM_REQUIRED}"
export METABONK_SMOKE_WAIT_S="${WAIT_S}"

python - <<'PY'
import json
import os
import sys
import time
import urllib.request

orch = os.environ.get("METABONK_SMOKE_ORCH_URL", "http://127.0.0.1:8040").rstrip("/")
workers_expected = int(os.environ.get("METABONK_SMOKE_WORKERS", "1"))
wait_s = int(os.environ.get("METABONK_SMOKE_WAIT_S", "120"))

def _get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

deadline = time.time() + wait_s
last_err = None
while time.time() < deadline:
    try:
        _get_json(f"{orch}/status", timeout=1.0)
        break
    except Exception as e:
        last_err = e
        time.sleep(1.0)
else:
    raise SystemExit(f"[smoke] orchestrator not ready: {last_err}")

deadline = time.time() + wait_s
last = None
while time.time() < deadline:
    try:
        last = _get_json(f"{orch}/workers", timeout=2.0)
        if isinstance(last, dict) and len(last) >= workers_expected:
            break
    except Exception:
        pass
    time.sleep(1.0)
else:
    count = len(last or {}) if isinstance(last, dict) else 0
    raise SystemExit(f"[smoke] only {count} workers registered (expected {workers_expected})")

print(f"[smoke] orchestrator ready; workers registered: {len(last or {})}")
PY

if [[ "${STREAM_REQUIRED}" == "1" ]]; then
  if ! command -v ffprobe >/dev/null 2>&1; then
    echo "[smoke] ffprobe not found (install ffmpeg)"
    exit 1
  fi
  python - <<'PY'
import importlib.util
missing = []
for name in ("PIL", "numpy"):
    if importlib.util.find_spec(name) is None:
        missing.append(name)
if missing:
    raise SystemExit(f"[smoke] missing python deps: {', '.join(missing)}")
PY
  python - <<'PY'
import json
import os
import time
import urllib.request
from io import BytesIO
import subprocess

import numpy as np
from PIL import Image

orch = os.environ.get("METABONK_SMOKE_ORCH_URL", "http://127.0.0.1:8040").rstrip("/")

def _get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _worker_base_url(hb: dict) -> str | None:
    cu = (hb.get("control_url") or "").strip()
    if cu:
        return cu.rstrip("/")
    su = (hb.get("stream_url") or "").strip()
    if not su:
        return None
    if "/stream" in su:
        return su.split("/stream", 1)[0].rstrip("/")
    if "/" in su:
        return su.rsplit("/", 1)[0].rstrip("/")
    return su.rstrip("/")

workers = _get_json(f"{orch}/workers", timeout=2.0)
target = None
for hb in workers.values():
    base = _worker_base_url(hb or {})
    if base:
        target = base
        break
if not target:
    raise SystemExit("[smoke] no worker base URL found for stream probe")

frame_url = f"{target}/frame.jpg"
with urllib.request.urlopen(frame_url, timeout=3.0) as resp:
    data = resp.read()
img = Image.open(BytesIO(data)).convert("L")
arr = np.array(img, dtype=np.float32)
var = float(arr.var())
if var < 5.0:
    raise SystemExit(f"[smoke] frame variance too low ({var:.2f}) from {frame_url}")
print(f"[smoke] frame variance OK: {var:.2f}")

stream_url = f"{target}/stream.mp4"
cmd = [
    "ffprobe",
    "-v", "error",
    "-rw_timeout", "5000000",
    "-read_intervals", "%+1",
    "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,width,height,r_frame_rate",
    "-of", "json",
    stream_url,
]
out = subprocess.check_output(cmd, timeout=10.0)
info = json.loads(out.decode("utf-8", "replace"))
streams = info.get("streams") or []
if not streams:
    raise SystemExit(f"[smoke] ffprobe found no video streams at {stream_url}")
st = streams[0]
codec = st.get("codec_name")
if codec != "h264":
    raise SystemExit(f"[smoke] expected h264 codec, got {codec}")
if int(st.get("width") or 0) <= 0 or int(st.get("height") or 0) <= 0:
    raise SystemExit("[smoke] invalid stream dimensions")
print(f"[smoke] stream OK: {codec} {st.get('width')}x{st.get('height')} at {stream_url}")
PY
fi

if [[ "${GO2RTC_ENABLED}" == "1" ]]; then
  python - <<'PY'
import os
from pathlib import Path
import urllib.request
import urllib.error

cfg = os.environ.get("METABONK_GO2RTC_CONFIG") or str(Path(__file__).resolve().parent.parent / "temp" / "go2rtc.yaml")
mode = str(os.environ.get("METABONK_GO2RTC_MODE", "fifo") or "fifo").strip().lower()
if not os.path.exists(cfg):
    raise SystemExit(f"[smoke] go2rtc config missing: {cfg}")
text = Path(cfg).read_text(errors="replace")
if mode == "fifo" and "#raw" not in text:
    raise SystemExit("[smoke] go2rtc fifo config missing #raw passthrough tag")
url = os.environ.get("METABONK_GO2RTC_URL", "http://127.0.0.1:1984").rstrip("/") + "/"
try:
    with urllib.request.urlopen(url, timeout=2.0) as resp:
        code = getattr(resp, "status", 200)
        if int(code) >= 400:
            raise SystemExit(f"[smoke] go2rtc HTTP {code} at {url}")
except urllib.error.HTTPError as e:
    raise SystemExit(f"[smoke] go2rtc HTTP {e.code} at {url}") from e
except Exception as e:
    raise SystemExit(f"[smoke] go2rtc not reachable at {url}: {e}") from e
print("[smoke] go2rtc config OK + api reachable")
PY
fi

if [[ "${INPUT_CHECK}" == "1" ]]; then
  if [[ "${METABONK_INPUT_BACKEND:-}" == "uinput" ]]; then
    python "${REPO_ROOT}/scripts/virtual_input.py" --tap-key ENTER
  elif [[ "${METABONK_INPUT_BACKEND:-}" == "xdotool" ]]; then
    xdotool -v >/dev/null
  else
    echo "[smoke] input check skipped (set METABONK_INPUT_BACKEND=uinput|xdotool)"
  fi
fi

if [[ "${FAILOVER_CHECK}" == "1" ]]; then
  export METABONK_SUPERVISE_WORKERS=1
  python - <<'PY'
import json
import os
import signal
import time
import urllib.request

orch = os.environ.get("METABONK_SMOKE_ORCH_URL", "http://127.0.0.1:8040").rstrip("/")
expected = int(os.environ.get("METABONK_SMOKE_WORKERS", "1"))

def _get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

workers = _get_json(f"{orch}/workers", timeout=2.0)
target = None
for hb in workers.values():
    pid = hb.get("worker_pid")
    iid = hb.get("instance_id")
    if pid and iid:
        target = (int(pid), str(iid))
        break
if not target:
    raise SystemExit("[smoke] failover: no worker_pid found in heartbeats")

pid, iid = target
print(f"[smoke] failover: killing worker {iid} pid={pid}")
os.kill(pid, signal.SIGTERM)

deadline = time.time() + 60
new_pid = None
while time.time() < deadline:
    workers = _get_json(f"{orch}/workers", timeout=2.0)
    hb = workers.get(iid)
    if hb:
        cur = hb.get("worker_pid")
        if cur and int(cur) != pid:
            new_pid = int(cur)
            break
    time.sleep(1.0)

if not new_pid:
    raise SystemExit("[smoke] failover: worker did not restart")

print(f"[smoke] failover: worker restarted pid={new_pid}")
PY
fi

if [[ "${GAME_FAILOVER_CHECK}" == "1" ]]; then
  python - <<'PY'
import json
import os
import signal
import time
import urllib.request

orch = os.environ.get("METABONK_SMOKE_ORCH_URL", "http://127.0.0.1:8040").rstrip("/")

def _get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

workers = _get_json(f"{orch}/workers", timeout=2.0)
target = None
for hb in workers.values():
    cu = (hb.get("control_url") or "").rstrip("/")
    if not cu:
        continue
    try:
        status = _get_json(f"{cu}/status", timeout=2.0)
    except Exception:
        continue
    pid = status.get("launcher_pid")
    iid = status.get("instance_id")
    if pid and iid:
        target = (cu, int(pid), str(iid))
        break
if not target:
    print("[smoke] game failover: no launcher_pid found; skipping")
    raise SystemExit(0)

base, pid, iid = target
print(f"[smoke] game failover: killing launcher for {iid} pid={pid}")
os.kill(pid, signal.SIGTERM)

deadline = time.time() + 90
new_pid = None
while time.time() < deadline:
    try:
        status = _get_json(f"{base}/status", timeout=2.0)
    except Exception:
        time.sleep(1.0)
        continue
    cur = status.get("launcher_pid")
    alive = status.get("launcher_alive")
    if cur and int(cur) != pid and alive:
        new_pid = int(cur)
        break
    time.sleep(1.0)

if not new_pid:
    raise SystemExit("[smoke] game failover: launcher did not restart")
print(f"[smoke] game failover: launcher restarted pid={new_pid}")
PY
fi

echo "[smoke] OK"
