# To Do / Next Steps / To Evolve

This file replaces the completed plan_#.md checklist. It captures forward-looking
work that is intentionally not yet implemented or needs validation.

## Immediate validation
- Run a 6-worker training session for >2h and confirm stability (no stream stalls, no PPO divergence).
- Verify Stream HUD at 1280x720 and 1920x1080 with real feeds, check legibility and motion.
- Validate eval ladder updates: run eval-only workers and confirm `/eval/ladder` ranking stability.

## PBT + policy ops
- Add a small “policy ops” panel to the Dev dashboard (policy versions, eval scores, last mutation time).
- Add a safe PBT mute switch (per-policy or global) exposed via orchestrator API.
- Add scheduled policy snapshot export (per-policy best eval + most recent live).

## Vision/menu pipeline
- Build the menu classifier dataset from multiple game versions/themes.
- Measure menu classifier accuracy and false positives by UI subtype (reward vs selection vs pause).
- Train a clickability detector with explicit hard-negative mining on non-interactive UI.

## GPU-first throughput
- Add GPU-side action masking (optional) to avoid CPU round-trips for large UI grids.
- Profile per-worker GPU utilization and log end-to-end latency histograms.
- Experiment with lower-frame-rate policy inference during menu screens to reduce GPU load.

## Robustness + recovery
- Add worker self-heal for stream codec errors (auto-restart NVENC pipeline only).
- Add episode boundary check when vision metrics supply `menu_mode` and `is_dead`.
- Add watchdog for stale `policy_version` (stuck weights vs learner update).

## Replay/Highlight system
- Add retention policy (cap storage by GB or clips per run).
- Generate “top N” highlight playlist for stream rotation.
- Add metadata overlay to clip exports (policy version, seed, episode length).
- Auto-markable feats + Hall of Fame/Shame (see `docs/feats_auto_markable.md`).

## S2 planner integration 
- Wire a small-budget latent planner hook behind a feature flag.
- Track planner cost vs. return uplift on evaluation seeds.
- Distill planner actions into PPO policy (KD pass).

## Documentation
- Add a README for training + eval modes and their env knobs.
- Add a runbook section for PBT + eval ladder operations.
