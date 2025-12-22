# Unsolvable Issues (with attempts)

This document captures items from the research-grade UI plan that could not be fully implemented with the current repository state. Each issue includes at least three concrete attempts and the outcome.

---

## 1) End-to-end OpenTelemetry + Sentry correlation (frontend + orchestrator)

**Why blocked:** No OTel/Sentry dependencies or configuration exist in the repo; adding them would require new packages, deployment config, and DSNs that are not present.

**Attempts**
1. **Dependency search:** `rg -n "opentelemetry|sentry" src requirements*.txt docs` → no references or packages found.
2. **Tracing hooks search:** `rg -n "otel|trace|tracing" src` → only internal “reasoning trace” mentions; no actual OTel instrumentation or exporter wiring.
3. **Config/env search:** `rg -n "SENTRY|sentry" -S .` → no DSN/env vars or config blocks present.

**What is needed to unblock**
- Add dependencies (`opentelemetry-*`, `sentry-sdk`) to `requirements.txt` and frontend package.json.
- Provide DSN/service endpoints + sampling config in env.
- Decide on exporter targets (OTLP endpoint, Sentry org/project) and release versioning (git SHA).

---

## 2) Context Drawer “Last N frames/keys/actions” flight recorder

**Why blocked:** Worker does not store or expose action history or a frame buffer beyond the latest JPEG; no API exists to query per-step input history.

**Attempts**
1. **Action history search:** `rg -n "action_history|last_actions|action_seq|inputs" src/worker src/input` → no action buffer or key history found.
2. **Frame buffer search:** `rg -n "latest_frame|jpeg|frame" src/worker/nvenc_streamer.py` → only on-demand `capture_jpeg()` and latest JPEG cache; no ring buffer of frames.
3. **Entropy/telemetry integration search:** `rg -n "action_entropy|entropy" src/worker src/learner src/streaming` → entropy exists in learners, but not forwarded to worker heartbeat or API.

**What is needed to unblock**
- Implement a ring buffer in worker (recent frames + decoded inputs).
- Extend worker heartbeat payload or new `/telemetry` endpoint to expose action sequences.
- Add capture/serialization to avoid blowing up bandwidth.

---

## 3) Build Lab “example runs + clips per combo”

**Why blocked:** Highlights/clip system exists but does not capture inventory combos nor queryable clip metadata; Build Lab currently uses live heartbeats only.

**Attempts**
1. **Highlight system scan:** `rg -n "highlight|clip" src/orchestrator` → highlight endpoints exist, but no build combo linkage or combo query endpoints.
2. **Inventory metadata scan:** `rg -n "inventory_items" src/orchestrator src/worker` → inventory is present only in heartbeats; no persistence for historical runs/clips.
3. **BuildLab UI scan:** checked `src/frontend/src/pages/BuildLab.tsx` → “EXAMPLE RUNS” list only uses currently live workers and does not have clip URLs or historical store.

**What is needed to unblock**
- Persist build inventories per episode/run (DB or structured artifact store).
- Attach inventory snapshot metadata to highlight encode requests.
- Add new backend endpoints `/buildlab/examples?combo=...` returning clip URLs + run metadata.

---

If you want, I can draft the storage schema and endpoint contracts for these three items.
