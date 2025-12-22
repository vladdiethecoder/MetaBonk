# Unsolvable Issues (with attempts)

This document captures remaining blockers that could not be fully resolved inside the repo alone. Each issue includes at least three concrete attempts and the outcome.

---

## 1) External OTel/Sentry infra still required (collector + DSN)

**Why blocked:** The repo now contains optional Sentry/OTel wiring, but end-to-end correlation still depends on external infrastructure (Sentry project/DSN and an OTLP collector endpoint) which cannot be provisioned inside the codebase.

**Attempts**
1. **Dependency wiring:** Added Sentry + OpenTelemetry packages to `requirements.txt` and guarded initialization in `src/orchestrator/main.py` (best-effort, no crash if missing).
2. **Trace propagation:** Added W3C `traceparent` generation and injection in `src/frontend/src/api.ts` so browser → orchestrator requests carry trace context.
3. **FastAPI instrumentation:** Added optional `FastAPIInstrumentor` + OTLP exporter configuration in `src/orchestrator/main.py` (activates when `OTEL_EXPORTER_OTLP_ENDPOINT` is set).

**What is needed to unblock**
- Real Sentry DSN + project for `SENTRY_DSN`.
- OTLP collector endpoint reachable by the orchestrator.
- (Optional) a frontend OTel SDK if you want full browser spans instead of generated traceparent only.

---

## 2) Build Lab archive ingestion pipeline (automatic)

**Why blocked:** The Build Lab DB + endpoints are in place, but no automated producer is emitting `BuildRun` events with inventory + clip data. Without producers, the archive remains empty unless manually populated.

**Attempts**
1. **Storage + endpoints:** Implemented SQLite store + `/buildlab/runs` and `/buildlab/examples` in `src/orchestrator/main.py`.
2. **Best-effort event hook:** Added persistence when events include `build_hash`/`inventory_snapshot` in `emit_event()`.
3. **UI wiring:** `src/frontend/src/pages/BuildLab.tsx` now queries archived examples and displays clip links when available.

**What is needed to unblock**
- A worker or post-process job to POST build runs (items + clip_url) to `/buildlab/runs`.
- Or emit `BuildRun` events with `inventory_snapshot` + `clip_url` from the highlight/clip pipeline.

---

If you want, I can add the worker-side emitter that posts build runs when a clip is encoded.
