# ADR 0005: Game Auto-Restart in Worker

## Status
Accepted

## Context
Game crashes should not tear down the whole stack. Previously, a crashed game
required manual intervention to relaunch, even if the worker remained healthy.

## Decision
Add a lightweight game restart watchdog in the worker heartbeat loop. If the
launcher process exits and a game command is configured, the worker relaunches
the game with bounded retries and backoff.

## Consequences
- Game crashes can recover without restarting the orchestrator/learner.
- Restarts are bounded to prevent flapping.
- Disabled when no MEGABONK command/template is configured.
