# DreamBridge → Offline Replay Migration

This repo previously exposed `src/neuro_genie/dream_bridge.py` as a pixel-level Gym
environment for “dream training”. That environment is intentionally disabled
because synthetic initialization + placeholder reward logic makes debugging and
training signals unreliable.

## What to use now (honest + grounded)

### Offline replay environment (actions ignored)

Use `src/neuro_genie/offline_replay_env.py` to iterate over real `.pt` rollouts.

Rollout format (current repo default):

- `observations`: `Tensor[T, obs_dim]`
- `actions`: `Tensor[T, action_dim]`
- `rewards`: `Tensor[T]`
- `dones`: `Tensor[T]`

### Training entrypoint

`scripts/train_neuro_genie.py --mode dream_rl` now runs **behavior cloning**
from offline `.pt` rollouts (not interactive dreaming).

Example:

```bash
python scripts/train_neuro_genie.py --mode dream_rl --pt-dir rollouts/ --max_steps 5000
```

Outputs:

- `checkpoints/neuro_genie/offline_replay_policy.pt`

## Why this change

- Keeps the system offline-first (no live game dependency).
- Avoids “fake interactivity” claims: the replay env does not pretend actions
  change the next state.
- Produces a solid initialization for later online RL or learned-dynamics training.

