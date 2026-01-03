# VLM Hive (System2 Cognitive Server)

> **Deprecated (2026-01-03):** The SGLang-based cognitive server has been removed
> from the active stack. This document is retained for historical context only.

MetaBonk uses a centralized “System2” cognitive server to generate low-frequency,
vision-language directives that can guide exploration when the policy appears stuck.

## Overview

- Workers send a **temporal strip** (1–9 JPEG frames) plus a small **agent state** dict.
- The cognitive server responds with a directive: goal + suggested action + confidence.
- Workers gate execution with a confidence threshold and do not rely on any menu/scene labels.

## Protocol

- Transport: ZeroMQ (DEALER on worker, server speaks a simple request/response JSON protocol).
- Worker client: `src/agent/cognitive_client.py`
  - builds temporal strips from a short frame buffer
  - staggers request timing across workers to avoid synchronized bursts

## Key Environment Variables

Worker-side:
- `METABONK_COGNITIVE_SERVER_URL` (default `tcp://127.0.0.1:5555`)
- `METABONK_SYSTEM2_FRAMES` (1..9, default 9)
- `METABONK_SYSTEM2_CLICK_CONF_MIN` (minimum directive confidence to act)

Cognitive-server-side (docker):
- `METABONK_COGNITIVE_STUCK_SINGLE_FRAME` (use single-frame reasoning when stuck)
- `METABONK_COGNITIVE_TILE_EDGE_STUCK` (tile edge override when stuck)

## Running

- Start the full stack (recommended): `./launch`
- Or start only the cognitive server: `scripts/start_cognitive_server.sh`

## Verifying

- Orchestrator shows worker `vlm_hints_used` in `/status`.
- The stack verifier includes an optional System2 metrics probe:
  - `python3 scripts/verify_running_stack.py --workers 5 --skip-cognitive=false`
