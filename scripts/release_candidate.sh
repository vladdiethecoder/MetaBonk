#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/release_candidate.sh [options]

Builds MetaBonk "Release Candidate" desktop UI bundles (Tauri) and packages the
Python worker stack + Synthetic Eye binary as application resources.

Options:
  --out DIR            Output directory (default: dist/release-candidate/<tag>)
  --tag TAG            Tag used for output dir naming (default: metabonk-<version>-<gitsha>)
  --bundles B...       Bundles to build (default: deb appimage). Values: deb rpm appimage
  --npm-install MODE   Dependency install mode: ci|install|skip (default: ci)
  --skip-eye           Skip building rust/metabonk_smithay_eye
  --skip-ui            Skip building the Tauri app (artifact collection still runs)
  --clean              Delete the output directory before copying artifacts
  -h, --help           Show this help

Examples:
  scripts/release_candidate.sh
  scripts/release_candidate.sh --bundles deb appimage --npm-install install
  scripts/release_candidate.sh --tag rc1 --out dist/release-candidate/rc1
EOF
}

OUT_DIR=""
TAG=""
NPM_MODE="ci"
SKIP_EYE=0
SKIP_UI=0
CLEAN=0
BUNDLES=("deb" "appimage")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --bundles)
      shift
      BUNDLES=()
      while [[ $# -gt 0 && "${1:-}" != --* ]]; do
        BUNDLES+=("$1")
        shift
      done
      ;;
    --npm-install)
      NPM_MODE="${2:-}"
      shift 2
      ;;
    --skip-eye)
      SKIP_EYE=1
      shift
      ;;
    --skip-ui)
      SKIP_UI=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

TAURI_VERSION="$(
  python3 - <<'PY'
import json
from pathlib import Path

cfg = Path("src/frontend/src-tauri/tauri.conf.json")
data = json.loads(cfg.read_text(encoding="utf-8"))
print(str(data.get("version") or "0.0.0").strip())
PY
)"

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -z "${TAG}" ]]; then
  TAG="metabonk-${TAURI_VERSION}-${GIT_SHA}"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/dist/release-candidate/${TAG}"
fi

mkdir -p "$OUT_DIR"
if [[ "$CLEAN" == "1" ]]; then
  case "$OUT_DIR" in
    "${REPO_ROOT}/dist/"*|"${REPO_ROOT}/dist")
      rm -rf "$OUT_DIR"
      mkdir -p "$OUT_DIR"
      ;;
    *)
      die "--clean refused: OUT_DIR must be under ${REPO_ROOT}/dist (got: $OUT_DIR)"
      ;;
  esac
fi

echo "[release_candidate] repo: $REPO_ROOT"
echo "[release_candidate] tag:  $TAG"
echo "[release_candidate] out:  $OUT_DIR"
echo "[release_candidate] bundles: ${BUNDLES[*]}"

if [[ "$SKIP_EYE" != "1" ]]; then
  echo "[release_candidate] building Synthetic Eye (rust/metabonk_smithay_eye)..."
  (cd rust && cargo build -p metabonk_smithay_eye --release)
  [[ -f rust/target/release/metabonk_smithay_eye ]] || die "missing rust/target/release/metabonk_smithay_eye after build"
fi

if [[ "$SKIP_UI" != "1" ]]; then
  [[ -f rust/target/release/metabonk_smithay_eye ]] || die \
    "missing rust/target/release/metabonk_smithay_eye (required for Tauri bundling; run without --skip-eye)"
  echo "[release_candidate] building Tauri bundles..."
  pushd src/frontend >/dev/null
  case "$NPM_MODE" in
    ci)
      npm ci
      ;;
    install)
      npm install
      ;;
    skip)
      ;;
    *)
      die "--npm-install must be one of: ci|install|skip (got: $NPM_MODE)"
      ;;
  esac
  npx tauri build --ci --bundles "${BUNDLES[@]}"
  popd >/dev/null
fi

echo "[release_candidate] collecting artifacts..."
ARTIFACTS=()
shopt -s nullglob
for b in "${BUNDLES[@]}"; do
  case "$b" in
    deb)
      ARTIFACTS+=(src/frontend/src-tauri/target/release/bundle/deb/*.deb)
      ;;
    appimage)
      ARTIFACTS+=(src/frontend/src-tauri/target/release/bundle/appimage/*.AppImage)
      ;;
    rpm)
      ARTIFACTS+=(src/frontend/src-tauri/target/release/bundle/rpm/*.rpm)
      ;;
    *)
      die "unsupported bundle: $b (expected: deb|rpm|appimage)"
      ;;
  esac
done
shopt -u nullglob

if [[ "${#ARTIFACTS[@]}" -lt 1 ]]; then
  die "no artifacts found under src/frontend/src-tauri/target/release/bundle/*"
fi

for f in "${ARTIFACTS[@]}"; do
  cp -f "$f" "$OUT_DIR/"
done

python3 - <<PY
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(${REPO_ROOT@Q})
out_dir = Path(${OUT_DIR@Q})

def _run(cmd: list[str]) -> str:
  try:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
  except Exception:
    return "unknown"

def _sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()

artifacts = []
for p in sorted(out_dir.iterdir()):
  if p.is_file() and p.suffix in {".deb", ".rpm", ".AppImage"}:
    artifacts.append(
      {
        "name": p.name,
        "bytes": p.stat().st_size,
        "sha256": _sha256(p),
      }
    )

manifest = {
  "tag": ${TAG@Q},
  "tauri_version": ${TAURI_VERSION@Q},
  "git_sha": ${GIT_SHA@Q},
  "built_at": datetime.now(timezone.utc).isoformat(),
  "tooling": {
    "python": _run(["python3", "--version"]),
    "node": _run(["node", "--version"]),
    "npm": _run(["npm", "--version"]),
    "rustc": _run(["rustc", "--version"]),
    "cargo": _run(["cargo", "--version"]),
  },
  "artifacts": artifacts,
}

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
(out_dir / "SHA256SUMS").write_text(
  "".join([f"{a['sha256']}  {a['name']}\\n" for a in artifacts]),
  encoding="utf-8",
)

print(f"[release_candidate] wrote {out_dir/'manifest.json'}")
print(f"[release_candidate] wrote {out_dir/'SHA256SUMS'}")
PY

echo "[release_candidate] done"
echo "[release_candidate] output: $OUT_DIR"
