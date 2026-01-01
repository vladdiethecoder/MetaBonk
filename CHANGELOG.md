# Changelog

## 2025-12-31

### Changed
- Enforced strict pure-vision runs by removing menu bootstrap shortcuts and their configuration knobs.
- Updated stack verifier to avoid relying on System 2 scene labels for "menu stuck" heuristics.

### Added
- `tests/test_pure_vision_enforcement.py` to prevent regression of removed bootstrap paths.
- `scripts/validate_pure_vision.sh` for a one-command local validation.
- `docs/pure_vision_learning.md` documenting the strict pure-vision policy.

