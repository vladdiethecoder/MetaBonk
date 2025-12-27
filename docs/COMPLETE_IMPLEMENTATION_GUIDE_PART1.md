# MetaBonk: Complete Production Implementation Guide
## Part 1: Autonomous Discovery (Phases 0–1)

This repo now contains the working, test-covered implementations for Phase 0–1.

### Phase 0: Input Enumeration
- Code: `src/discovery/input_enumerator.py`
- Seed selection: `src/discovery/input_seed.py`
- Outputs (via runner):
  - `cache/discovery/<env>/input_space.json`
  - `cache/discovery/<env>/seed_buttons.json`
  - `cache/discovery/<env>/suggested_env.json`

### Phase 1: Effect Detection + Input Exploration
- Effect detection (pixel diff + optical flow + confidence): `src/discovery/effect_detector.py`
- Exploration + artifact writing (`effect_map.json` with `{metadata, results}`): `src/discovery/input_explorer.py`
- Runner: `scripts/run_autonomous.py`

Run Phase 1 locally (no real game required):
```bash
python3 scripts/run_autonomous.py --env megabonk --phase 1 --env-adapter mock --exploration-budget 200 --hold-frames 10
```

Outputs:
```
cache/discovery/megabonk/effect_map.json
```

