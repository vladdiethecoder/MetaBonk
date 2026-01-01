# Pure Vision Mode (Strict)

Pure vision mode enforces a simple rule: **no privileged game knowledge**.
The agent learns from pixels (or GPU-preprocessed pixels) and uses only visual
signals for progress and intrinsic rewards.

## What “Pure Vision” Means Here

- No menu/scene classifiers in the worker fast-path
- No hardcoded scene labels (MainMenu/CharacterSelect/etc.)
- No special-cased “menu navigation” code paths
- Progress detection is based on **visual fingerprints** and **novelty**

## Enable

- `METABONK_PURE_VISION_MODE=1`

## Exploration Signals

The worker maintains vision-only intrinsic signals via:

- `src/agent/visual_exploration_reward.py` (`VisualExplorationReward`)
  - `visual_novelty`: mean luma change (0..1) at a small downsampled resolution
  - `scene_hash`: perceptual hash (dHash) of the downsampled frame
  - `scenes_discovered`: count of unique hashes seen
  - `stuck_score`: increases when novelty stays low for a long time

These metrics are exposed in worker `/status`.

## Validation

- Static enforcement:
  - `./scripts/validate_pure_vision.sh`
- Unit tests:
  - `pytest -q tests/test_pure_vision_enforcement.py`

## Common Failure Modes

- **CPU fallback**: production paths should hard-fail when GPU prerequisites are missing.
  Use `METABONK_REQUIRE_CUDA=1` to enforce CUDA 13.1+ / CC 9.0+.
- **Stream artifacts**: use `python3 scripts/validate_streams.py --use-orch` and the worker
  stream watchdog (black/frozen detection + bounded self-heal).

