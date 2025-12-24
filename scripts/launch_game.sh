#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "[game] usage: launch_game.sh <instance_id>" >&2
  exit 2
fi

if [[ -z "${METABONK_GAME_CMD:-}" ]]; then
  echo "[game] ERROR: METABONK_GAME_CMD is not set." >&2
  exit 1
fi

CMD="${METABONK_GAME_CMD//\{instance_id\}/${INSTANCE_ID}}"
exec bash -lc "$CMD"
