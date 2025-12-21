#!/usr/bin/env bash
set -euo pipefail

# go2rtc exec helper: run a headless producer that outputs a raw H.264 elementary stream
# to stdout (Annex-B). go2rtc reads stdout and publishes via RTSP/WebRTC/MSE.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <instance-id> [extra args...]" >&2
  exit 2
fi

INSTANCE_ID="$1"
shift

exec python3 -m src.streaming.headless_agent --instance-id "$INSTANCE_ID" "$@"

