# MetaBonk Unified Launcher

One command to run the full stack (cognitive server + Omega + UI):

```bash
./launch
```

## Commands

```bash
./launch                 # start (default profile)
./launch start           # start (explicit)
./launch stop            # stop omega + go2rtc + cognitive server
```

## Profiles

Profiles are JSON files loaded from `configs/launch_<name>.json`.

```bash
./launch --config default
./launch --config production
```

Override a couple common fields without editing JSON:

```bash
./launch --workers 3
./launch --mode train
```

## What It Starts

- Cognitive server via `scripts/start_cognitive_server.sh` (docker compose)
- Omega + Vite UI via `scripts/start.py`
- Terminal dashboard (optional) with GPU/worker/System2 metrics

The launcher reuses existing scripts so it stays stable as MetaBonk evolves.

## Interfaces

- Stream page: `http://127.0.0.1:5173/stream`
- Neural broadcast: `http://127.0.0.1:5173/neural/broadcast`
- Orchestrator: `http://127.0.0.1:8040/workers`

## Worker Endpoints (for smoke tests)

- Status: `http://127.0.0.1:5000/status`
- Video: `http://127.0.0.1:5000/stream.mp4` (use `GET`; `HEAD` returns `405`)

## Verify a Running Stack

After `./launch` is up, run a non-destructive verifier:

```bash
python3 scripts/verify_running_stack.py --workers 5
```

## Pure Vision Validation

Validate strict pure-vision enforcement (no bootstrap shortcuts):

```bash
./scripts/validate_pure_vision.sh
```

## Full System Validation

Run preflight + (if running) live stack checks:

```bash
./scripts/validate_system.sh
```

## Notes

- Some environments export `DOCKER_HOST` pointing to a rootless Podman socket. The launcher (and `scripts/start_cognitive_server.sh`) will prefer the system Docker daemon when `/var/run/docker.sock` exists.
- MetaBonk is GPU-only; launcher hard-fails if `nvidia-smi` or `docker --gpus all` is not working.
