# MetaBonk Ops Checklist

## Preflight
- [ ] Game install path verified (contains `Megabonk.exe`)
- [ ] BonkLink installed into BepInEx
- [ ] GPU driver and NVENC available
- [ ] PipeWire running (for gamescope capture)

## Start (6 instances)
```
python scripts/update_plugins.py --game-dir "/path/to/steamapps/common/Megabonk"
bash ./start --mode train --workers 6 --game-dir "/path/to/steamapps/common/Megabonk"
```

## Optional: go2rtc FIFO distribution (recommended for shared viewing)
```
bash ./start --mode train --workers 10 --go2rtc
```

Then open:
- `http://127.0.0.1:1984/stream.html?src=omega-0`

## Stream backend check
Use this to verify PipeWire + encoder availability:
```
python scripts/stream_diagnostics.py --backend auto
```
Or run the full GPU streaming checklist:
```
./scripts/check_gpu_streaming.sh
```

## Verify (services + workers)
```
python - <<'PY'
import json, urllib.request
workers = json.load(urllib.request.urlopen("http://127.0.0.1:8040/workers"))
print("workers", len(workers))
keep = ["status","stream_ok","stream_type","stream_url","stream_error","stream_backend","pipewire_node_ok"]
for k in sorted(workers.keys()):
    v = workers[k]
    print(k, {kk: v.get(kk) for kk in keep})
PY
```

Expected:
- `workers 6`
- `stream_type=mp4`
- `stream_ok=true` for featured

## UI Check
- Open `http://127.0.0.1:5173/stream`
- Ensure 6 tiles render and play
- If OBS browser source crops the UI, see `docs/obs_browser_source_overlay.md`

## Logs
- Per-instance logs: `temp/game_logs/`
- Per-instance dirs: `temp/megabonk_instances/omega-*`

## Quick Recovery
- If stream errors: restart with `METABONK_STREAM_CODEC=h264` and `METABONK_STREAM_CONTAINER=mp4`.
- If `nvh264enc` is missing: keep `METABONK_STREAM_BACKEND=auto` (default) or set `METABONK_STREAM_BACKEND=ffmpeg` (or `obs`) to use FFmpeg NVENC.
- If PipeWire missing: use `MEGABONK_USE_XVFB=1` and `METABONK_STREAM_BACKEND=x11grab`.
- If BepInEx logs missing: enable console in `BepInEx/config/BepInEx.cfg`.

## Shutdown
- Ctrl+C the `start` process (cleans up workers and instances).
