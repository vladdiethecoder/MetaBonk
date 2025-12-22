# ADR 0002: Stream Watchdog (Black Frame Detection)

## Status
Accepted

## Context
Streams can appear “alive” (encoder producing buffers) while delivering black
frames due to PipeWire/encoder mis-binding. This is not detected by simple
timestamp checks and breaks UI monitoring.

## Decision
Add a lightweight watchdog that computes variance on cached JPEG frames and
triggers a streamer restart if the variance stays below a threshold for a
configurable duration.

## Consequences
- No extra capture cost (uses cached JPEGs).
- Adds guardrails without impacting hot streaming paths.
- Tunable thresholds for very dark scenes.
