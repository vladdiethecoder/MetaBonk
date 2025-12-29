#!/usr/bin/env bash
# MetaBonk Stream Interruption Diagnostics
# Focus: keyframe cadence + dropped frames

set -euo pipefail

echo "🔍 MetaBonk Stream Interruption Diagnostics"
echo "==========================================="
echo ""

ORCH_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8040}"
WORKER_ID="${METABONK_DIAG_WORKER_ID:-omega-0}"

if ! command -v curl &> /dev/null; then
  echo "❌ curl not found"
  exit 1
fi

if ! curl -sf "${ORCH_URL}/workers" > /dev/null; then
  echo "❌ MetaBonk orchestrator not reachable at ${ORCH_URL}"
  exit 1
fi

export ORCHESTRATOR_URL="${ORCH_URL}"
export METABONK_DIAG_WORKER_ID="${WORKER_ID}"

mapfile -t URLS < <(python - <<'PY'
import json, os, sys, urllib.request

orch = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8040").rstrip("/")
wid = os.environ.get("METABONK_DIAG_WORKER_ID", "omega-0")

try:
    with urllib.request.urlopen(f"{orch}/workers", timeout=2.5) as r:
        data = json.load(r)
except Exception:
    print("")
    sys.exit(0)

w = data.get(wid) or {}
stream_url = (w.get("stream_url") or "").strip()
control_url = (w.get("control_url") or "").strip()
if not stream_url and control_url:
    stream_url = control_url.rstrip("/") + "/stream.mp4"
print(stream_url)
PY
)

STREAM_URL="${URLS[0]:-}"
if [ -z "${STREAM_URL}" ]; then
  echo "❌ Could not resolve worker stream URL from ${ORCH_URL}/workers"
  exit 1
fi

echo "ORCH_URL: ${ORCH_URL}"
echo "WORKER_ID: ${WORKER_ID}"
echo "STREAM_URL: ${STREAM_URL}"
echo ""

if ! command -v ffprobe &> /dev/null; then
  echo "⚠️  ffprobe not found; install ffmpeg to run keyframe diagnostics"
  exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Keyframe cadence (first ~100 frames)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ffprobe -v error -show_frames -select_streams v:0 \
  -show_entries frame=key_frame,pkt_pts_time \
  -read_intervals '%+#100' \
  "${STREAM_URL}" 2>/dev/null | grep -A1 'key_frame=1' | head -n 40 || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Dropped frames (60s decode probe)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v ffmpeg &> /dev/null; then
  echo "⚠️  ffmpeg not found; install ffmpeg to run drop probe"
  exit 0
fi

echo "Monitoring stream for 60 seconds..."
timeout 60 ffmpeg -hide_banner -loglevel info -i "${STREAM_URL}" -f null - 2>&1 | \
  grep -E 'frame=|drop=' | tail -n 20 || true

