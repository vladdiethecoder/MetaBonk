# Auto-markable Feats + Auto-clip Pipeline

This repo supports auto-markable feats that can trigger highlight clips without manual tracking.
Feats are evaluated in the orchestrator based on structured events pushed by workers (or
telemetry plugins). When a feat triggers, the orchestrator requests a highlight clip from
that worker’s rolling buffer and stores the result in the Hall of Fame / Shame.

## Quick start
1) Define feats in `configs/feats.megabonk.json` (see examples).
2) Enable worker highlight buffers:
   - `METABONK_HIGHLIGHTS=1`
3) Ensure your telemetry/events are sent to `POST /events` on the orchestrator.

The orchestrator loads the feats file from:
- `METABONK_FEATS_PATH` (default: `configs/feats.megabonk.json`).

## Feat schema (JSON)
Each feat is a small declarative predicate with a clip window:

```json
{
  "id": "minor.pots.20_in_run",
  "name": "Pot Popper",
  "tier": "minor",
  "scope": "run",
  "predicate": { "counter_gte": ["pots_broken", 20] },
  "clip": { "pre_s": 12, "post_s": 6, "speed": 3.0 },
  "hall": "fame",
  "dedupe": "once_per_run"
}
```

### Fields
- `id`: stable identifier
- `name`: display name
- `tier`: `minor` | `major`
- `scope`: `run` | `stage` | `lifetime`
- `predicate`: boolean expression (see below)
- `clip`: `pre_s`, `post_s`, optional `speed`
- `hall`: `fame` or `shame`
- `dedupe`: `once_per_run`, `once_per_stage`, `best_only`

## Supported predicates
- `counter_gte: [name, value]`
- `payload_gte: [name, value]`
- `event_happened: [event_type, {payload filters}]`
- `event_then_within: [eventA, eventB, seconds]`
- `and`, `or`, `not`

Example composite:
```json
{
  "and": [
    { "counter_gte": ["boss_curse_activations_stage", 2] },
    { "event_happened": ["stage_end", {"result":"clear"}] }
  ]
}
```

Time-window example:
```json
{
  "event_then_within": [
    ["boss_spawned", {"boss_id":"bark_vader"}],
    ["boss_defeated", {"boss_id":"bark_vader"}],
    20
  ]
}
```

## Telemetry/event contract (minimal)
Send structured events to the orchestrator with the fields you need for predicates.
Examples:
- `event_type="counter_delta"`, payload: `{ "name": "pots_broken", "delta": 1 }`
- `event_type="boss_defeated"`, payload: `{ "boss_id": "bark_vader" }`
- `event_type="stage_end"`, payload: `{ "result": "clear", "stage": 2, "biome": "forest" }`
- `event_type="goal_completed"`, payload: `{ "goal_id": "m01" }`

Optional snapshot payload:
```json
{ "counters": { "pots_broken": 22, "gold_total": 1200 } }
```

## Clip capture
On feat trigger, the orchestrator calls the worker’s highlight endpoint:
- `POST /highlight/encode` with `{ tag: "feat_<id>", score, speed }`.

Highlight clips are stored in `METABONK_HIGHLIGHTS_DIR` (default: `highlights/`).
The orchestrator serves them at `/highlights/...`.

## API endpoints
- `GET /feats` – list feat definitions
- `GET /feats/unlocks` – recent feat unlocks
- `GET /feats/hall` – hall of fame + shame records
- `GET /feats/validate` – report feats missing telemetry fields

## Telemetry contract
Generated from the current feat predicates:
- `docs/telemetry_contract.feats.json`

Diff live telemetry vs contract:
```bash
python scripts/telemetry_contract_diff.py
```

## Persistence
Feat unlocks are persisted to disk:
- `METABONK_FEATS_UNLOCKS_PATH` (default: `checkpoints/feats_unlocks.json`)

## Notes
- This is best-effort: if highlights are disabled or a worker is unavailable, feats still
  unlock but `clip_url` may be missing.
- To avoid training leakage, keep telemetry out of the agent observation space and use
  it only for evaluation/clip triggers.
