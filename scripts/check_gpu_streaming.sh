#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

say() {
  printf "%s\n" "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_gst_encoder() {
  if ! have_cmd gst-inspect-1.0; then
    say "[gst] gst-inspect-1.0 not found (install gstreamer1.0-tools)"
    return 1
  fi
  if gst-inspect-1.0 nvh264enc >/dev/null 2>&1; then
    say "[gst] nvh264enc available"
    return 0
  fi
  if gst-inspect-1.0 vaapih264enc >/dev/null 2>&1; then
    say "[gst] nvh264enc missing; vaapih264enc available"
    return 0
  fi
  if gst-inspect-1.0 amfh264enc >/dev/null 2>&1; then
    say "[gst] nvh264enc missing; amfh264enc available"
    return 0
  fi
  say "[gst] no GPU H.264 encoder elements found (nvh264enc/vaapih264enc/amfh264enc)"
  return 1
}

check_ffmpeg_encoder() {
  if ! have_cmd ffmpeg; then
    say "[ffmpeg] ffmpeg not found"
    return 1
  fi
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -qE "[[:space:]]h264_nvenc[[:space:]]"; then
    say "[ffmpeg] h264_nvenc available"
    return 0
  fi
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -qE "[[:space:]]h264_vaapi[[:space:]]"; then
    say "[ffmpeg] h264_nvenc missing; h264_vaapi available"
    return 0
  fi
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -qE "[[:space:]]h264_amf[[:space:]]"; then
    say "[ffmpeg] h264_nvenc missing; h264_amf available"
    return 0
  fi
  say "[ffmpeg] no GPU H.264 encoders found (h264_nvenc/h264_vaapi/h264_amf)"
  return 1
}

say "MetaBonk GPU streaming check"
say "repo: ${ROOT}"
say ""
say "Environment"
say "  METABONK_STREAM_BACKEND=${METABONK_STREAM_BACKEND:-auto}"
say "  METABONK_STREAM_CODEC=${METABONK_STREAM_CODEC:-h264}"
say "  METABONK_STREAM_CONTAINER=${METABONK_STREAM_CONTAINER:-mp4}"
say "  PIPEWIRE_NODE=${PIPEWIRE_NODE:-unset}"
say "  METABONK_STREAM_FIFO_DIR=${METABONK_STREAM_FIFO_DIR:-${ROOT}/temp/streams}"
say "  METABONK_GO2RTC_URL=${METABONK_GO2RTC_URL:-unset}"
say "  METABONK_FIFO_STREAM=${METABONK_FIFO_STREAM:-${METABONK_GO2RTC:-0}}"
say ""

check_gst_encoder || true
check_ffmpeg_encoder || true

if have_cmd nvidia-smi; then
  say "[nvidia] nvidia-smi detected"
else
  say "[nvidia] nvidia-smi not found (skip GPU device check)"
fi

FIFO_DIR="${METABONK_STREAM_FIFO_DIR:-${ROOT}/temp/streams}"
if [ -d "${FIFO_DIR}" ]; then
  COUNT="$(ls -1 "${FIFO_DIR}" 2>/dev/null | wc -l | tr -d ' ')"
  say "[fifo] ${FIFO_DIR} exists (${COUNT} entries)"
else
  say "[fifo] ${FIFO_DIR} not found"
fi

say ""
say "Running stream diagnostics"
set +e
python scripts/stream_diagnostics.py --backend "${METABONK_STREAM_BACKEND:-auto}"
DIAG_RC=$?
set -e
say ""

if [ "${DIAG_RC}" -ne 0 ]; then
  say "Diagnostics reported failure. Recommended next checks:"
  say "  1) Force backend: METABONK_STREAM_BACKEND=gst or METABONK_STREAM_BACKEND=ffmpeg"
  say "  2) Verify PIPEWIRE_NODE is valid (gamescope capture node/serial)"
  say "  3) Run smoke test: python scripts/egl_vpf_demo.py --fifo temp/streams/test.h264 --frames 120"
  exit "${DIAG_RC}"
fi

say "Diagnostics OK."
say "If video is still choppy, ensure the UI is using /stream.mp4 or go2rtc, not /frame.jpg."
