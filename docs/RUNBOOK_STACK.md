# MetaBonk Stack Runbook

## Quick Start (E2E)
```bash
./start --mode train --workers 2 --go2rtc --go2rtc-mode fifo
```

## Smoke Test (Automated)
```bash
scripts/smoke_stack.sh
```
Options:
- `METABONK_SMOKE_WORKERS=2`
- `METABONK_SMOKE_STREAM=1`
- `METABONK_SMOKE_GO2RTC=1`
- `METABONK_SMOKE_GO2RTC_MODE=fifo`
- `METABONK_SMOKE_WAIT_S=120`
- `METABONK_SMOKE_UI=0`
- `METABONK_SMOKE_INPUT=1`

## Verify (Manual)
- API: `curl -s http://127.0.0.1:8040/status`
- Workers: `curl -s http://127.0.0.1:8040/workers | jq 'keys | length'`
- Featured: `curl -s http://127.0.0.1:8040/featured`
- Stream frame: `curl -o /tmp/frame.jpg http://127.0.0.1:<worker_port>/frame.jpg`

## Stream Probing
```bash
ffprobe -v error -rw_timeout 5000000 -read_intervals %+1 \
  -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate \
  -of json http://127.0.0.1:<worker_port>/stream.mp4
```

## Stream Watchdog (Black Frame)
Environment knobs:
- `METABONK_STREAM_BLACK_VAR` (default 1.0)
- `METABONK_STREAM_BLACK_S` (default 8.0)
- `METABONK_STREAM_BLACKCHECK_S` (default 5.0)

## Input Sanity
- uinput:
```bash
METABONK_INPUT_BACKEND=uinput python scripts/virtual_input.py --tap-key ENTER
```
- xdotool:
```bash
METABONK_INPUT_BACKEND=xdotool xdotool -v
```

## Logs / Artifacts
- Stack logs: `temp/`
- go2rtc config: `temp/go2rtc/`
- Game logs: `temp/game_logs/`
- Stream FIFOs: `temp/streams/`

## Troubleshooting
1) **API down**: check `temp/start_omega.log` and orchestrator stdout.
2) **Workers missing**: check worker logs and game launch output in `temp/game_logs/`.
3) **Stream black**: validate PipeWire node, `stream_ok` field in `/workers`, and `ffprobe`.
4) **Input not reflected**: ensure BonkLink `EnableInputSnapshot=true`, and `METABONK_INPUT_BACKEND` is configured.

## Stop
```bash
python scripts/stop.py --all --go2rtc
```
