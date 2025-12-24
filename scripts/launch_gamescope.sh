#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "[gamescope] usage: launch_gamescope.sh <instance_id> -- <game command...>" >&2
  exit 2
fi
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if ! command -v gamescope >/dev/null 2>&1; then
  echo "[gamescope] ERROR: gamescope not found in PATH." >&2
  exit 1
fi

if [[ ! -d /sys/class/drm ]] || [[ -z "$(ls /sys/class/drm 2>/dev/null | grep -E '^card[0-9]+$' || true)" ]]; then
  echo "[gamescope] ERROR: /sys/class/drm is empty; DRM backend unavailable." >&2
  exit 1
fi

if [[ -n "${METABONK_EDID_FILE:-}" ]]; then
  if [[ ! -f "${METABONK_EDID_FILE}" ]]; then
    echo "[gamescope] ERROR: METABONK_EDID_FILE not found: ${METABONK_EDID_FILE}" >&2
    exit 1
  fi
  export GAMESCOPE_PATCHED_EDID_FILE="${METABONK_EDID_FILE}"
fi

export STEAM_MULTIPLE_XWAYLANDS=1
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

WIDTH="${MEGABONK_WIDTH:-1920}"
HEIGHT="${MEGABONK_HEIGHT:-1080}"
FPS="${MEGABONK_FPS:-60}"

EXTRA_ARGS=()
if [[ -n "${METABONK_GAMESCOPE_VK_DEVICE:-}" ]]; then
  EXTRA_ARGS+=("--prefer-vk-device" "${METABONK_GAMESCOPE_VK_DEVICE}")
fi
if [[ -n "${METABONK_GAMESCOPE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(${METABONK_GAMESCOPE_ARGS})
fi

if [[ $# -eq 0 ]]; then
  echo "[gamescope] ERROR: no game command provided." >&2
  exit 2
fi

if [[ -n "${NOTIFY_SOCKET:-}" ]] && command -v systemd-notify >/dev/null 2>&1; then
  systemd-notify --ready
  (
    while true; do
      systemd-notify WATCHDOG=1
      sleep 2
    done
  ) &
  WATCHDOG_PID=$!
  trap 'kill ${WATCHDOG_PID} 2>/dev/null || true' EXIT
fi

GAMESCOPE_CMD=(
  gamescope
  --backend drm
  --immediate-flips
  --headless
  -W "${WIDTH}"
  -H "${HEIGHT}"
  -r "${FPS}"
  "${EXTRA_ARGS[@]}"
  --
  "$@"
)

if command -v setsid >/dev/null 2>&1; then
  setsid "${GAMESCOPE_CMD[@]}" &
  GAMESCOPE_PID=$!
else
  "${GAMESCOPE_CMD[@]}" &
  GAMESCOPE_PID=$!
fi

PGID="${GAMESCOPE_PID}"
trap 'kill -TERM -'"${PGID}"' 2>/dev/null || true; sleep 1; kill -KILL -'"${PGID}"' 2>/dev/null || true' EXIT
wait "${GAMESCOPE_PID}"
