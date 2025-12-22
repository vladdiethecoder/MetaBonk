# ADR 0004: Worker Supervisor (Crash Recovery)

## Status
Accepted

## Context
Previously, any worker exit caused `start_omega.py` to terminate the entire stack.
This prevents recovery from transient worker crashes.

## Decision
Add a lightweight supervisor in `start_omega.py` to restart worker and Xvfb
processes with a bounded retry limit and backoff. Core services still fail fast.

## Consequences
- Worker crashes no longer tear down the whole stack.
- Restart attempts are bounded to avoid flapping.
- Behavior is configurable via `METABONK_SUPERVISE_WORKERS`.
