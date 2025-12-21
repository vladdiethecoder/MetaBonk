#!/usr/bin/env bash
set -euo pipefail

# Quick sanity check for the exec-mode headless producer:
# - runs a tiny stream for a moment
# - verifies that H.264 Annex-B start codes appear in output

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-temp/headless_agent_smoke.h264}"
mkdir -p "$(dirname "$OUT")"

echo "[smoke] running headless_agent (cpu renderer, low-res) ..."
ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null || true)"
FORCE_SW=""
if echo "$ENCODERS" | grep -q "libx264" || echo "$ENCODERS" | grep -q "libopenh264"; then
  FORCE_SW="--force-sw"
fi

timeout 6s python3 -m src.streaming.headless_agent \
  --instance-id smoke-0 \
  --renderer cpu \
  --width 320 \
  --height 180 \
  --fps 10 \
  --gop 10 \
  --bitrate 800k \
  $FORCE_SW \
  >"$OUT" || true

if [[ ! -s "$OUT" ]]; then
  if [[ -z "$FORCE_SW" ]]; then
    echo "[smoke] no output produced (this environment likely requires working NVENC). Output: $OUT" >&2
    exit 3
  fi
  echo "[smoke] no output produced: $OUT" >&2
  exit 1
fi

# Look for Annex-B start code 00 00 00 01 in the first 512KB.
if head -c 524288 "$OUT" | grep -a -q $'\\x00\\x00\\x00\\x01'; then
  echo "[smoke] OK: found Annex-B start code in $OUT"
  exit 0
fi

echo "[smoke] WARN: output does not look like Annex-B H.264 (start codes missing) in $OUT" >&2
exit 2
