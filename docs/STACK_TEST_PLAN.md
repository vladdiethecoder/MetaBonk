# MetaBonk Stack Test Plan

## Purpose
Provide deterministic, automatable verification for the end-to-end MetaBonk stack:
orchestrator, workers, game instances, streaming pipeline (GPU-only), input bridge, UI,
and failure recovery.

## Scope
In scope:
- Orchestrator + N workers (N >= 2)
- Game instances launch + window/handle presence
- PipeWire capture + GPU encode streaming
- go2rtc distribution (FIFO/exec)
- UI health + API correctness
- Input bridge (keyboard + mouse)
- Failure recovery (worker/game/stream lifecycle)

Out of scope (for initial smoke):
- Long-run training stability
- Model convergence metrics
- Multi-host cluster scheduling

## Preconditions
- NVIDIA driver + CUDA available (no CPU fallback)
- PipeWire running
- Game install directory with `Megabonk.exe`
- go2rtc available (docker compose or local binary)
- `ffmpeg` + `ffprobe` available

## Test Matrix (Initial)
| Scenario | Workers | Stream | go2rtc Mode | Expectation |
| --- | --- | --- | --- | --- |
| S1 | 1 | On | fifo | Single instance stream valid + API healthy |
| S2 | 2 | On | fifo | Two workers healthy + at least one featured stream valid |
| S3 | 2 | On | exec | WebRTC/MSE stream valid + no FIFO deadlocks |
| S4 | 2 | Off | n/a | Workers + UI healthy without stream |
| S5 | 4 | On | fifo | No stream stalls, no dropped workers |
| S6 | 2 | On | fifo | Failure recovery: worker crash, game crash, stream restart |

## Success Criteria
- `/health` returns OK
- `/workers` returns expected worker count within timeout
- Worker heartbeats include required fields
- Stream probe returns H.264 with expected resolution/fps
- Non-black frames detected (pixel variance > threshold)
- Keyboard + mouse input causes observable UI change in-game (menu navigation)
- Crash recovery: worker/game restart detected and recorded
- go2rtc config uses raw H.264 passthrough or MPEG-TS (no unintended CPU transcode)
- go2rtc API responds on configured URL
- stream pacing check passes for FIFO/go2rtc sample (`scripts/stream_pacing_check.py --gate`)
- Instance history includes step_age_s and stream_frame_var fields

## Commands (Baseline)
- Launch: `./start --mode train --workers 2 --go2rtc --go2rtc-mode fifo`
- Stop: `python scripts/stop.py --all --go2rtc`
- Diagnostics: `python scripts/stream_diagnostics.py --require-pipewire`
- Smoke: `scripts/smoke_stack.sh`
- GPU preflight (when required): `METABONK_REQUIRE_CUDA=1 pytest -k gpu_requirements`

## Expected Artifacts
- Logs in `temp/`
- Stream FIFOs in `temp/streams/`
- go2rtc config in `temp/go2rtc/`

## Troubleshooting Decision Tree (Outline)
1. API not healthy → check orchestrator logs
2. Workers missing → check worker spawn + game launch logs
3. Stream black → check PipeWire node + encoder logs + ffprobe
4. Input not reflected → check BonkLink bridge + input injector
5. Recovery failed → check process supervisor, zombie encoders, and FIFO cleanup

## Future Additions
- Full regression test list
- Performance budget checks (latency/copy count)
- Detailed failure injection steps (worker kill, game kill, stream client disconnect)
- Input end-to-end validation (menu navigation)
