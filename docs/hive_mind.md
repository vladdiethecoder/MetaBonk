# Hive Mind / Federated Swarm (Recovery MVP)

This repo includes a “virtual swarm” single‑player parallelization path inspired by the Apex Protocol:

- Many low‑cost workers collect experience in parallel (“Scout”, “Speedrunner”, “Killer”, “Tank”, “Builder”).
- A learner‑side federated merge combines specialized policies into a single **God Agent** via **Task Arithmetic / TIES‑Merging**.

## What’s implemented now

### Role‑specialized swarm launcher

Script: `scripts/launch_hive_mind_cluster.py`

- Starts orchestrator, vision service, learner.
- Spawns a swarm of Python workers with role‑specific `policy_name` and hparams.
- Pre‑registers each instance’s config in the orchestrator so PBT doesn’t overwrite roles.

Example:

```bash
python scripts/launch_hive_mind_cluster.py \
  --scouts 4 --speedrunners 3 --killers 3 --tanks 2 --builders 2
```

Workers will come up as instance IDs like `hive-0`, `hive-1`, … with policies:
`Scout`, `Speedrunner`, `Killer`, `Tank`, `Builder`.

If you are using the IL2CPP `unity_plugin/MetabonkPlugin`, per‑role env vars such as
`METABONK_VELOCITY_WEIGHT` and `METABONK_SURVIVAL_TICK` will shape in‑game rewards.

### Federated merge endpoint

Learner endpoint: `POST /merge_policies`

Merges role policies into a target policy (default “God”) using TIES:

```bash
curl -X POST http://127.0.0.1:8061/merge_policies \
  -H 'Content-Type: application/json' \
  -d '{
    "source_policies": ["Scout","Speedrunner","Killer","Tank","Builder"],
    "target_policy": "God",
    "method": "ties",
    "topk": 0.2
  }'
```

Optional safeguards and dry-run:

```bash
curl -X POST http://127.0.0.1:8061/merge_policies \
  -H 'Content-Type: application/json' \
  -d '{
    "source_policies": ["Scout","Speedrunner"],
    "target_policy": "God",
    "method": "ties",
    "topk": 0.2,
    "dry_run": true,
    "min_cosine": 0.01,
    "min_sign_agreement": 0.02
  }'
```

Env overrides:
- `METABONK_MERGE_MIN_COSINE`
- `METABONK_MERGE_MIN_SIGN_AGREEMENT`
- `METABONK_MERGE_FORCE=1` (bypass gating)
- `METABONK_MERGE_SAVE_VECTOR=1` (persist merged vector to `METABONK_SKILL_DB_PATH`)

### Merge sidecar

Script: `scripts/federated_merge.py`

Runs the merge on a timer:

```bash
python scripts/federated_merge.py \
  --sources Scout Speedrunner Killer Tank Builder \
  --target God \
  --interval-s 30
```

## Current limitations

- The Python worker loop is still in recovery mode and uses a minimal survival reward unless a real game bridge is running.
- Real headless MegaBonk spawning/render‑disable is not automated here; you must start the game instances yourself.
- Causal build discovery, Liquid policies, and neural GameNGen world models exist as modules but are not wired into the live swarm loop yet.

## Next upgrades (when game bridge is available)

1. Move role rewards from env vars → true per‑role dense rewards using game memory (velocity/DPS/synergy).
2. Add automatic headless spawning/reset via Gamescope + BepInEx.
3. Promote merged “God” policy to a dedicated evaluation worker and feed its deltas back into the skill DB.
