# Pre‑Gameplay Fixes Summary (Menu Escape + Action Cadence)

**Date**: 2026‑01‑01  
**Goal**: workers reliably escape pre‑gameplay UI (warnings/menus) and reach gameplay, without game‑specific logic.

---

## Symptoms addressed

- Workers remain stuck in character selection / warning screens for minutes.
- `step` remains low despite high FPS.
- It’s hard to distinguish “frozen” vs “still navigating UI”.

---

## What changed (facts)

1) **System2 pre‑gameplay directive enforcement**  
When `Agent State.gameplay_started=false`, System2 is instructed (pure vision prompt) to output:
- `directive.action="interact"`
- `directive.target=[x,y]` in **normalized** coordinates (0..1)
- `confidence>=0.6` when selecting a target from `Agent State.ui_elements`  
Implementation: `docker/cognitive-server/cognitive_server.py`

2) **Pre‑gameplay post‑processing fallback (server‑side)**  
If `gameplay_started=false` (or `stuck=true`) and the model returns a low‑confidence or default target, the server:
- picks an “advance/proceed” click target from `ui_elements` (OCR/button candidates), and
- bumps `confidence` (>=0.6 pre‑gameplay; >=0.25 when stuck in gameplay)  
Implementation: `docker/cognitive-server/cognitive_server.py`

3) **Always provide System2 with UI candidates**  
Workers now always provide `Agent State.ui_elements`. If detectors/OCR return none, a coarse **grid fallback** is generated (spatial only).  
Implementation: `src/worker/main.py` (+ `src/worker/perception.py` grid helper)

4) **Pre‑gameplay epsilon UI exploration (worker‑side)**  
After `METABONK_UI_PRE_GAMEPLAY_GRACE_S` (default 2s), workers inject epsilon‑greedy UI clicks until `gameplay_started=true`.  
In pure‑vision mode, if `METABONK_UI_PRE_GAMEPLAY_EPS` is unset/<=0, an effective default `0.7` is used.  
Implementation: `src/worker/main.py`

5) **Telemetry to disambiguate “no progress”**  
`/status` now exposes:
- `act_hz`: action cadence (Hz)
- `actions_total`: total actions taken
- `gameplay_started`: pre‑gameplay vs gameplay  
Implementation: `src/worker/main.py`

6) **Guardrails before long runs**  
New verifier flags:
- `scripts/verify_running_stack.py --require-gameplay-started`
- `scripts/verify_running_stack.py --require-act-hz <min>`  
`scripts/launch_24hr_test.sh` blocks the 24‑hour run until these checks pass.  
Implementation: `scripts/verify_running_stack.py`, `scripts/launch_24hr_test.sh`

---

## How to validate (recommended)

From repo root:

```bash
./scripts/validate_pregameplay_fixes.sh
```

Or directly:
```bash
python3 scripts/verify_running_stack.py \
  --workers 5 \
  --skip-ui \
  --skip-go2rtc \
  --require-gameplay-started \
  --require-act-hz 5
```

---

## Interpreting metrics (important)

- `step` is not a raw action counter; it can remain low in menus.
- Use `act_hz` and `actions_total` to confirm the agent is actively trying UI interactions.
- If `stream_ok=false` or `stream_frozen=true`, fix streaming first.
