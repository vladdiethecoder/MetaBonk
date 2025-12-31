#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535 >/dev/null 2>&1 || true

export MEGABONK_LOG_DIR="${MEGABONK_LOG_DIR:-/tmp/megabonk_logs}"
export METABONK_STREAM_MAX_CLIENTS="${METABONK_STREAM_MAX_CLIENTS:-4}"
export METABONK_STREAM_STARTUP_TIMEOUT_S="${METABONK_STREAM_STARTUP_TIMEOUT_S:-30}"
export METABONK_STREAM_STALL_TIMEOUT_S="${METABONK_STREAM_STALL_TIMEOUT_S:-30}"
export METABONK_CAPTURE_ALL="${METABONK_CAPTURE_ALL:-0}"
export CHOKIDAR_USEPOLLING="${CHOKIDAR_USEPOLLING:-1}"
export CHOKIDAR_INTERVAL="${CHOKIDAR_INTERVAL:-500}"

LOG_DIR="${METABONK_LOG_DIR:-${MEGABONK_LOG_DIR:-}}"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="$(pwd)/temp"
fi
mkdir -p "${LOG_DIR}"

omega_log_default="${LOG_DIR}/metabonk_omega.log"
ui_log_default="${LOG_DIR}/metabonk_ui.log"
OMEGA_LOG="${METABONK_OMEGA_LOG_PATH:-${omega_log_default}}"
UI_LOG="${METABONK_UI_LOG_PATH:-${ui_log_default}}"

if [[ -e "${OMEGA_LOG}" && ! -w "${OMEGA_LOG}" ]]; then
  OMEGA_LOG="${LOG_DIR}/metabonk_omega.$$.log"
fi
if [[ -e "${UI_LOG}" && ! -w "${UI_LOG}" ]]; then
  UI_LOG="${LOG_DIR}/metabonk_ui.$$.log"
fi

MODE="${METABONK_MODE:-train}"
WORKERS="${METABONK_WORKERS:-6}"
GAME_DIR="${MEGABONK_GAME_DIR:-}"

mkdir -p temp

omega_cmd=(bash ./start --mode "${MODE}" --workers "${WORKERS}" --no-ui)
if [[ -n "${GAME_DIR}" ]]; then
  omega_cmd+=(--game-dir "${GAME_DIR}")
fi

"${omega_cmd[@]}" > "${OMEGA_LOG}" 2>&1 &

for _ in {1..120}; do
  python - <<'PY' && break || true
import sys
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8040/status", timeout=2).read(1)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
  sleep 0.5
done

(cd src/frontend && npm run dev -- --host 127.0.0.1 --port 5173 > "${UI_LOG}" 2>&1) &

stream_url=""
for _ in {1..120}; do
  stream_url="$(python - <<'PY' || true
import json
import sys
import urllib.request
import os
try:
    # Prefer orchestrator /workers (single heartbeat payloads with featured_slot annotations).
    payload = json.load(urllib.request.urlopen("http://127.0.0.1:8040/workers", timeout=2))
except Exception:
    sys.exit(1)

data = payload
if isinstance(payload, dict) and isinstance(payload.get("workers_by_id"), dict):
    data = payload.get("workers_by_id") or {}
elif isinstance(payload, dict) and isinstance(payload.get("workers"), list):
    mapped = {}
    for row in payload.get("workers") or []:
        if not isinstance(row, dict):
            continue
        iid = str(row.get("instance_id") or "")
        if iid:
            mapped[iid] = row
    data = mapped

def okish(hb: dict) -> bool:
    # stream_ok may be False until the first client connects; pipewire_node_ok indicates capture ready.
    return hb.get("stream_ok") is True or hb.get("pipewire_node_ok") is not False

featured = []
for hb in data.values():
    if hb.get("featured_slot") and hb.get("stream_url") and okish(hb):
        featured.append(hb)
if featured:
    # Sort by slot name for determinism (slot strings like "A", "B", ...).
    featured.sort(key=lambda x: str(x.get("featured_slot") or ""))
    print(featured[0]["stream_url"])
    sys.exit(0)

allow_any = str(os.environ.get("METABONK_RUNALL_ALLOW_ANY_STREAM", "0") or "").strip().lower() in ("1", "true", "yes", "on")
if allow_any:
    any_ok = []
    for hb in data.values():
        if hb.get("stream_url") and okish(hb):
            any_ok.append(hb)
    if any_ok:
        any_ok.sort(key=lambda x: str(x.get("instance_id") or ""))
        print(any_ok[0]["stream_url"])
        sys.exit(0)

sys.exit(1)
PY
)"
  [[ -n "${stream_url}" ]] && break
  sleep 1
done

if [[ -z "${stream_url}" ]]; then
  echo "no featured stream url (check ${OMEGA_LOG})"
  exit 1
fi

echo "stream url: ${stream_url}"

if [[ "${METABONK_RUNALL_CAPTURE:-0}" != "1" ]]; then
  echo "skipping ffmpeg capture (set METABONK_RUNALL_CAPTURE=1 to enable)"
  exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; install it and re-run"
  exit 1
fi

for _ in {1..180}; do
  STREAM_URL="${stream_url}" python - <<'PY' || { sleep 2; continue; }
import sys
import urllib.request
import os
url = os.environ.get("STREAM_URL", "")
if not url:
    sys.exit(1)
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        data = r.read(4096)
    if b"moov" in data and b"avcC" in data:
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
  if ffmpeg -y -loglevel error -probesize 2M -analyzeduration 2M -i "${stream_url}" -frames:v 1 temp/gameplay_capture.png; then
    python - <<'PY' && { echo "Captured: temp/gameplay_capture.png"; exit 0; } || true
import sys
try:
    from PIL import Image
    im = Image.open("temp/gameplay_capture.png").convert("L")
    mean = sum(im.getdata()) / (im.width * im.height)
    sys.exit(0 if mean > 10 else 1)
except Exception:
    sys.exit(0)
PY
  fi
  sleep 2
done

echo "Captured (check manually): temp/gameplay_capture.png"
