# MetaBonk Deployment Checklist (Pure Vision Stack)

## Prereqs

- [ ] NVIDIA GPU present and visible: `nvidia-smi`
- [ ] CUDA-enabled PyTorch available (Blackwell target): `python3 -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability())"`
- [ ] Docker working (for cognitive server): `docker version`
- [ ] go2rtc available (if using public WebRTC): `go2rtc --version`

## Repo Setup

- [ ] Python deps: `pip install -r requirements.txt`
- [ ] Headless streaming deps (cuTile/CUDA extras): `pip install -r requirements-headless-streaming.txt`

## Preflight (Fail-Fast)

- [ ] GPU contract: `METABONK_REQUIRE_CUDA=1 pytest -q -k gpu_requirements`
- [ ] Pure vision enforcement: `./scripts/validate_pure_vision.sh`

## Start

- [ ] Start stack: `./launch --config default`
- [ ] Confirm orchestrator: `curl -s http://127.0.0.1:8040/workers | jq '.count'`
- [ ] Confirm UI: open `http://127.0.0.1:5173/stream`

## Streaming Quality (5 workers)

- [ ] Automated probe: `python3 scripts/validate_streams.py --use-orch --workers 5`
- [ ] Non-destructive stack verify: `python3 scripts/verify_running_stack.py --workers 5`

## Smoke E2E (opt-in)

- [ ] `METABONK_ENABLE_INTEGRATION_TESTS=1 pytest -q -m e2e`

## Success Criteria (1h)

- [ ] 5 workers registered and healthy
- [ ] Streams show live gameplay (not black/frozen)
- [ ] Stable step rate / FPS (no sustained stalls)
- [ ] `stuck_score` not permanently pegged at max
- [ ] No repeated restart loops in `runs/*/logs`

