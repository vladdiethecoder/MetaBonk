# The Asymmetry of Artificial Evolution: A Forensic Analysis of the Neuro-sama 2025 Ecosystem

## 0. How to Read This Report

This document is structured by **Fact → Inference → Hypothesis** to make the evidence chain auditable.

- **Facts** are supported by MetaBonk screenshots/logs and repo artifacts.
- **Inferences** are plausible and testable, but not directly shown.
- **Hypotheses** are speculative and explicitly labeled.

## 1. Executive Summary

**Core thesis:** the MetaBonk stack shows **asymmetric evolution**: the backend capabilities (world modeling, orchestration, stream intelligence) have advanced, while the operator-facing UI has degraded into staleness and performance collapse.

This revision separates evidence types, corrects public claims, and concludes with a prioritized engineering backlog.

## 2. Evidence Ledger (Facts)

These are verified from the provided UI/screenshots/logs and the repo artifacts.

### 2.1 UI/Telemetry Facts (Screenshots)

- **Build Lab is empty**: “inventory feed missing,” “coverage 0%,” “Synergy Web: EMPTY.”
- **Instances show stale or missing streams**: “STREAM_MISSING_NO_PIPEWIRE” appears on instance rows.
- **Stream Health is degraded**: “Stream OK: 0” and near-zero traffic.
- **Dream Policy is missing**: “Dream Policy: missing” in Spy/Pretrain panels.

### 2.2 Repo Facts (Local Artifacts)

- The UI contains **hive mind / latent / neuro-synaptic** language and components (see `src/frontend/src/pages/NeuroSynaptic.tsx`).
- Multiple docs describe **Omega / Neuro-Genie / Hive Mind** models and training pipelines (e.g. `docs/omega_protocol_research.md`, `docs/neuro_genie_architecture.md`, `docs/neuro_synaptic_forge.md`).

## 3. Inferences (Likely, Testable)

- **Schema drift**: heartbeats appear to arrive but payload fields are incompatible with the UI schema, causing stale/missing values.
- **Broken stream pipeline**: PipeWire failure prevents stream capture, which cascades into UI stalls.
- **Run/Instances elongation**: unbounded lists and log/event streams inflate DOM size, causing layout instability and memory pressure.

These are all testable with:
- schema validation + versioning,
- stream preflight checks,
- DOM size metrics + heap snapshots.

## 4. Hypotheses (Speculative)

> These are framed as hypotheses unless they can be anchored to MetaBonk’s own docs.

- **Hypothesis:** “Latent Manifold” and “Mamba Flow” language in the UI reflects a move toward latent/state-space modeling. This is consistent with MetaBonk’s internal research docs (e.g. `docs/neuro_genie_architecture.md`), but should not be treated as confirmed **Neuro-sama internals**.
- **Hypothesis:** The UI’s “Mixture-of-Reasonings” and “Hive Mind Cartography” imply a shift to multi-expert routing with geometry-driven state views. This is a plausible mapping to internal architecture, not public confirmation of any third-party system.

## 5. Public Claims (Corrected)

Only publicly verifiable claims should be stated as fact.

- **Supported:** Vedal/Neuro-sama reached **Hype Train Level 111**, a record reported by Dexerto on **January 2, 2025**.[^dexerto]
- **Not stated as fact:** any specific viewer counts, “hours watched,” or “Twitch confirmed” metrics without primary sources.

If those numbers are needed, add citations from TwitchTracker/StreamsCharts before asserting them.

## 6. Mechanism: Why the UI “Elongates”

### 6.1 Mechanism (Evidence → Cause)

- Long-running runs/instances/events tables append rows indefinitely.
- Browser layout + scroll height precision degrades as the DOM grows.

This is a **volume × time** bug that only appears at Subathon-scale uptime.

### 6.2 Canonical Fix (Evidence-backed)

Use list virtualization so only visible rows render. This is the standard approach for large lists (react-window).[^react-window]

## 7. PipeWire/Stream Failures: Make Them Observable + Self-Healing

Fedora routes desktop audio to PipeWire by default.[^pipewire-default] PipeWire uses a session manager to configure and route devices; Fedora moved to WirePlumber for this role.[^wireplumber]

### 7.1 Streamer Preflight (S1)

- At worker boot: detect PipeWire socket, session manager availability, and capture node presence.
- Emit a **structured issue** (e.g., `PIPEWIRE_UNAVAILABLE`) with evidence (stderr + exit code).

### 7.2 Stream Health Contract (S2)

Expose these in UI + logs:
- PipeWire running (yes/no)
- Session manager running (yes/no)
- Capture node detected (yes/no)
- Last keyframe age
- FPS / frame time

### 7.3 Self-Healing (S3)

- Retry attach with exponential backoff
- If still failing: degrade gracefully (video-only or “no audio” mode)
- Emit issue with playbook steps

## 8. Engineering Backlog (P0 → P2)

### P0 — Stop the Bleeding

1. **Virtualize all long lists** (runs, instances, events, logs) using react-window.[^react-window]
2. **Ring buffers everywhere** (client + server): cap at N items (e.g., 5k).
3. **Stream health contract** (PipeWire/session manager/keyframe/fps) surfaced in UI and logs.[^pipewire-default][^wireplumber]
4. **Heartbeat schema versioning**: UI should warn on mismatch instead of silently going stale.

### P1 — Make It Research-Grade

5. **Unified Issues system**: fingerprinting, dedupe, evidence links, ack/mute/TTL.
6. **Run/Step time-axis**: deep links across tabs (`run_id`, `instance_id`, `step/ts`).
7. **Pagination for historical views**: large backfills should page by time, not render as one list.

### P2 — Make It Stream-Grade Fun (without losing rigor)

8. **Diagnostic fun-metrics** (still measurable): “Bonk Confidence,” “Menu Doom Spiral.”
9. **Moments system**: auto-clip “clutch/disaster/weird build” events with replay queue.

## 9. MetaBonk Architecture vs. Public Claims (Boundary Clarification)

- **MetaBonk architecture** claims should be anchored to repo docs.
- **Neuro-sama internals** must be labeled hypothesis unless a public source confirms.

This keeps the report defensible under external scrutiny.

## 10. Appendix: Concrete Implementation Sketches

### 10.1 Virtualized List (React)

Use react-window’s `FixedSizeList` for constant-height rows.[^react-window]

### 10.2 Stream Preflight Checklist

- PipeWire socket present
- Session manager running (WirePlumber)
- Capture node enumerated
- Keyframe age < threshold
- FPS above minimum

---

### References

[^dexerto]: https://www.dexerto.com/entertainment/ai-powered-twitch-streamer-smashes-twitch-hype-train-world-record-3019551/
[^react-window]: https://web.dev/articles/virtualize-long-lists-react-window
[^pipewire-default]: https://fedoraproject.org/wiki/Changes/DefaultPipeWire
[^wireplumber]: https://fedoraproject.org/wiki/Changes/WirePlumber
