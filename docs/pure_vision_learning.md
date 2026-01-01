# Pure Vision Learning (Strict)

MetaBonk can run in a **strict pure-vision** mode: agents must learn from pixels without any scripted menu automation or privileged action shortcuts.

This document describes what "pure vision" means in this repo, what is forbidden, and how to validate enforcement.

## Definition

When `METABONK_PURE_VISION_MODE=1`:

- The training objective is driven by the agent's own learning stack (policy + intrinsic rewards).
- No hardcoded/menu-scripted actions are allowed to advance early prompts or menus.
- No bootstrap macros that click/confirm fixed UI positions are allowed.

Note: the stack may still run a separate System 2 server for **vision-based** reasoning (it consumes frames); however, **scripted** progression logic (fixed sequences / coordinates / "click confirm now") is not allowed as a bypass.

## What Is Forbidden

- Any "menu bootstrap" logic that:
  - injects pre-defined sequences (e.g., `ENTER`, `SPACE`, fixed click locations)
  - depends on game/menu names to choose a sequence
- Configuration or launcher knobs that enable such behavior
- Verification scripts that fail the run based on privileged scene labels (e.g., "stuck in character_select")

## What Is Allowed

- Intrinsic exploration rewards (e.g., RND / novelty) computed from agent observations
- Policy-driven exploration and action discovery
- Non-destructive observability tooling (streaming, dashboards, verifier scripts) as long as they do not drive control decisions via privileged labels

## Validation

Run the repo-provided validator:

```bash
./scripts/validate_pure_vision.sh
```

It performs:

- `py_compile` on key entrypoints
- a banned-token scan over runtime/config/launcher docs
- `pytest tests/test_pure_vision_enforcement.py`

You can also run the tests directly:

```bash
pytest -q tests/test_pure_vision_enforcement.py
```

## Operational Notes

- Removing scripted menu automation may increase the time-to-gameplay during early training. This is expected for strict pure-vision runs.
- The stack defaults to Synthetic Eye (`--synthetic-eye`) in the gold path; when enabled, worker env defaults `METABONK_OBS_BACKEND=pixels` if not set.

