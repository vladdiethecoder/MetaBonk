# Quick Test: Pre‑Gameplay UI Advancement Fixes (5 minutes)

This checks that workers can escape pre‑gameplay UI (warning/menu/character select) and that action cadence is healthy.

From repo root:
```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk
```

---

## 1) Start the stack

Use your normal start method. Example (headless):
```bash
python3 launch.py stop
sleep 3
python3 launch.py --config-file configs/launch_24hr_test.json --no-dashboard start
sleep 60
```

---

## 2) Run the automated validator (recommended)

```bash
./scripts/validate_pregameplay_fixes.sh
```

This verifies:
- all workers respond to `/status`
- required fields exist (`gameplay_started`, `act_hz`, `actions_total`)
- within timeout, all workers reach `gameplay_started=true`
- `act_hz >= 5` (default threshold)

---

## 3) Manual spot‑check (optional)

```bash
for p in {5000..5004}; do
  echo "worker $((p-5000))"
  curl -fsS "http://127.0.0.1:$p/status" | jq '{gameplay_started,act_hz,actions_total,vlm_hints_used,stream_ok,stream_frozen}'
done
```

Interpretation:
- `step` can be low in menus; use `act_hz` + `actions_total` for “is it doing anything”.
- if `stream_ok=false` or `stream_frozen=true`, treat it as a streaming/capture blocker.
