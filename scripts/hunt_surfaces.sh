#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DISCOVERY_TIMEOUT_S="${DISCOVERY_TIMEOUT_S:-45}"
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-35}"
AFTER_FRAME="${AFTER_FRAME:-600}"
BRIGHTNESS_THRESHOLD="${BRIGHTNESS_THRESHOLD:-10.0}"
MIN_UNIQUE_COLORS="${MIN_UNIQUE_COLORS:-500}"
MAX_BLACK_FRAC="${MAX_BLACK_FRAC:-0.95}"
MAX_CANDIDATES="${MAX_CANDIDATES:-25}"

OUT_ROOT="runs/hunt_surfaces/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"

echo "🔍 Phase 1: Discovery run (${DISCOVERY_TIMEOUT_S}s) to gather surface IDs..."

rm -f /tmp/debug_staging_*.ppm /tmp/debug_source_*.ppm /tmp/debug_staging_*.png /tmp/debug_source_*.png

python3 scripts/stop.py --all > /dev/null 2>&1 || true

METABONK_STEAM_AUTOSTART=1 \
METABONK_FORCE_LINEAR_EXPORT=1 \
METABONK_SYNTHETIC_EYE=1 \
METABONK_VISION_AUDIT=1 \
timeout "${DISCOVERY_TIMEOUT_S}s" ./start --workers 1 --mode train --no-ui --no-go2rtc > /dev/null 2>&1 || true

python3 scripts/stop.py --all > /dev/null 2>&1 || true

LATEST_RUN="$(ls -td runs/run-omega-* | head -1)"
LOG_FILE="$LATEST_RUN/logs/synthetic_eye_0.log"
echo "   Reading from: $LOG_FILE"

echo "🔍 Phase 2: Extracting candidate wl_surface IDs..."

mapfile -t CANDIDATES < <(
  python3 - "$LOG_FILE" "$MAX_CANDIDATES" <<'PY'
import re, sys
from collections import Counter

log_path = sys.argv[1]
max_candidates = int(sys.argv[2])

want = ("dmabuf commit", "xwayland window associated with wl_surface", "switching primary surface capture")
wl_re = re.compile(r"\bwl_surface=(\d+)\b")

counts = Counter()
with open(log_path, "r", errors="replace") as f:
    for line in f:
        if not any(w in line for w in want):
            continue
        m = wl_re.search(line)
        if not m:
            continue
        counts[int(m.group(1))] += 1

items = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
for wl, n in items[:max_candidates]:
    print(wl)
PY
)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "❌ No candidates found in $LOG_FILE"
  echo "   Tip: ensure compositor logs include wl_surface in 'dmabuf commit' lines."
  exit 1
fi

echo "   Found candidates (top ${#CANDIDATES[@]}): ${CANDIDATES[*]}"
echo ""

SUMMARY="$OUT_ROOT/summary.tsv"
printf "wl_surface\tmean_luma\tblack_frac\tunique_colors_sampled\trun_dir\tstaging_ppm\tsource_ppm\n" > "$SUMMARY"

for ID in "${CANDIDATES[@]}"; do
  echo "🎯 Testing wl_surface=$ID (timeout=${TEST_TIMEOUT_S}s, dump_frame=${AFTER_FRAME})"

  rm -f /tmp/debug_staging_*.ppm /tmp/debug_source_*.ppm /tmp/debug_staging_*.png /tmp/debug_source_*.png

  METABONK_EYE_FORCE_WL_SURFACE="$ID" \
  METABONK_DEBUG_DUMP_STAGING=1 \
  METABONK_DEBUG_DUMP_SOURCE=1 \
  METABONK_DEBUG_DUMP_STAGING_AFTER_FRAME="$AFTER_FRAME" \
  METABONK_DEBUG_DUMP_STAGING_MAX_DUMPS=1 \
  METABONK_STEAM_AUTOSTART=1 \
  METABONK_FORCE_LINEAR_EXPORT=1 \
  METABONK_SYNTHETIC_EYE=1 \
  METABONK_VISION_AUDIT=1 \
  METABONK_SYNTHETIC_EYE_STALL_RESTART_S=0 \
  timeout "${TEST_TIMEOUT_S}s" ./start --workers 1 --mode train --no-ui --no-go2rtc > /dev/null 2>&1 || true

  python3 scripts/stop.py --all > /dev/null 2>&1 || true

  NEW_RUN="$(ls -td runs/run-omega-* | head -1)"
  NEW_LOG="$NEW_RUN/logs/synthetic_eye_0.log"

  STAGING_LINE="$(rg -n "staging dump stats" "$NEW_LOG" | tail -n 1 || true)"
  SOURCE_LINE="$(rg -n "source dump stats" "$NEW_LOG" | tail -n 1 || true)"

  if [ -z "$STAGING_LINE" ]; then
    echo "   ⚠️  No staging dump stats found (surface may not have produced frames)"
    printf "%s\t\t\t\t%s\t\t\n" "$ID" "$NEW_RUN" >> "$SUMMARY"
    echo "   --------------------------------"
    continue
  fi

  # Extract stats + file paths from the log lines.
  MEAN_LUMA="$(printf "%s" "$STAGING_LINE" | rg -o "mean_luma=[0-9.]+" | head -n 1 | cut -d= -f2 || true)"
  BLACK_FRAC="$(printf "%s" "$STAGING_LINE" | rg -o "black_frac=[0-9.]+" | head -n 1 | cut -d= -f2 || true)"
  UNIQUE="$(printf "%s" "$STAGING_LINE" | rg -o "unique_colors_sampled=[0-9]+" | head -n 1 | cut -d= -f2 || true)"
  STAGING_PPM="$(printf "%s" "$STAGING_LINE" | rg -o "/tmp/debug_staging_[0-9]+\\.ppm" | head -n 1 || true)"
  SOURCE_PPM="$(printf "%s" "$SOURCE_LINE" | rg -o "/tmp/debug_source_[0-9]+\\.ppm" | head -n 1 || true)"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$ID" "${MEAN_LUMA:-}" "${BLACK_FRAC:-}" "${UNIQUE:-}" "$NEW_RUN" "${STAGING_PPM:-}" "${SOURCE_PPM:-}" >> "$SUMMARY"

  echo "   📊 mean_luma=${MEAN_LUMA:-?} black_frac=${BLACK_FRAC:-?} unique_colors_sampled=${UNIQUE:-?}"

  # Archive dumps (if present).
  OUT_DIR="$OUT_ROOT/wl_surface_${ID}"
  mkdir -p "$OUT_DIR"
  cp -f "$NEW_LOG" "$OUT_DIR/synthetic_eye_0.log" 2>/dev/null || true
  if [ -n "$STAGING_PPM" ] && [ -f "$STAGING_PPM" ]; then
    cp -f "$STAGING_PPM" "$OUT_DIR/$(basename "$STAGING_PPM")"
  fi
  if [ -n "$SOURCE_PPM" ] && [ -f "$SOURCE_PPM" ]; then
    cp -f "$SOURCE_PPM" "$OUT_DIR/$(basename "$SOURCE_PPM")"
  fi

  # Threshold decision (aims to reject banded/empty surfaces).
  if python3 - "$MEAN_LUMA" "$BLACK_FRAC" "$UNIQUE" "$BRIGHTNESS_THRESHOLD" "$MAX_BLACK_FRAC" "$MIN_UNIQUE_COLORS" <<'PY'
import sys
try:
    mean = float(sys.argv[1])
    black = float(sys.argv[2])
    uniq = int(sys.argv[3])
    thr_mean = float(sys.argv[4])
    thr_black = float(sys.argv[5])
    thr_uniq = int(sys.argv[6])
except Exception:
    sys.exit(1)
ok = (mean > thr_mean) and (black < thr_black) and (uniq >= thr_uniq)
sys.exit(0 if ok else 2)
PY
  then
    echo "   ✅ FOUND likely content surface: wl_surface=$ID (mean_luma=${MEAN_LUMA} black_frac=${BLACK_FRAC} unique=${UNIQUE})"
    echo "   Artifacts saved in: $OUT_DIR"
    echo "   Summary: $SUMMARY"
    exit 0
  fi

  echo "   --------------------------------"
done

echo "❌ Search complete. No surface exceeded mean_luma > ${BRIGHTNESS_THRESHOLD}."
echo "Summary: $SUMMARY"
exit 2
