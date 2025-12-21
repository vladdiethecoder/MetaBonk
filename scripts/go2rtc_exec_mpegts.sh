#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 -- <cmd...>" >&2
  exit 2
fi

if [[ "$1" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "missing command after --" >&2
  exit 2
fi

"$@" | ffmpeg -hide_banner -loglevel error \
  -fflags nobuffer -flags low_delay \
  -f h264 -i pipe:0 -c copy -f mpegts pipe:1
