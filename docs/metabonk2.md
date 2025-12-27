# MetaBonk2: Tripartite Controller (System 1 / System 2 / Metacognition)

MetaBonk2 is an **opt-in** controller layer intended to make experimentation with:

- **System 1**: fast reflex policy (per-step)
- **System 2**: slower deliberation / planning (multi-step “intent”)
- **Metacognition**: safety + time-budgeting + interruption heuristics

easier to integrate into the existing Omega worker loop without destabilizing the default training
stack.

## Enable

Set:

- `METABONK_USE_METABONK2=1`

Optional:

- `METABONK2_TIME_BUDGET_MS=150` (passed to metacognition / System 2)
- `METABONK2_LOG=1` (emit `[MetaBonk2]` trace lines)
- `METABONK2_OVERRIDE_DISCRETE=1` (override discrete key/button presses when using OS input backends)
- `METABONK2_OVERRIDE_CONT=1` (override continuous mouse deltas; **opt-in** to avoid clobbering aim policies)

## Status / Debug

When enabled, the worker `/status` endpoint includes a `metabonk2` object.

The Dev UI provides a live view at:

- `/reasoning`

This page polls the selected worker’s `/status` and renders:

- current `mode`
- current high-level `intent`
- active `skill` + remaining steps
- confidence / novelty / uncertainty (best-effort heuristics)
- a short System 2 “plan” list (if available)

## Notes

- MetaBonk2 is designed to be **safe to disable**; if it fails to initialize the worker falls back
  to the default control path.
- The long-term goal is to bridge MetaBonk2’s System 2 planning into existing “Neuro-Genie” / MoR
  modules where appropriate, while keeping the hot path deterministic and GPU-first.

