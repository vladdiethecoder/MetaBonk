# ADR 0001: Stream Probe Method (ffprobe + frame variance)

## Status
Accepted

## Context
We need a deterministic, automated way to verify the streaming pipeline without
manual inspection. The probe must:
- Work with live, infinite streams
- Validate GPU path output (codec/resolution)
- Detect black or stale frames

## Decision
Use a two-step probe:
1) `ffprobe` against `/stream.mp4` to verify codec and dimensions.
2) `/frame.jpg` snapshot to compute pixel variance and reject black frames.

## Consequences
- Requires `ffprobe`, `Pillow`, and `numpy` in the test environment.
- Works for live streams without needing to record files.
- Avoids CPU fallback or reduced quality checks.
