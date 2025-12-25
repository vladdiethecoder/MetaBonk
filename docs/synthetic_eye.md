# Synthetic Eye (Smithay/Vulkan → DMABuf+Fences → CUDA)

This repo historically used Gamescope → PipeWire for agent-loop visual ingest. The Synthetic Eye path
removes PipeWire from the **agent loop** by exporting **DMA-BUF + explicit sync fence FDs** directly
to the worker for CUDA-side import.

Status: the DMABuf+fence transport and CUDA-side import/sync are implemented. Full Smithay scene
composition (including XWayland plumbing for games) is still evolving.

## Build the compositor exporter

```bash
cd rust
cargo build -p metabonk_smithay_eye --release
```

Binary: `rust/target/release/metabonk_smithay_eye`

## Run (single instance)

```bash
./rust/target/release/metabonk_smithay_eye --id omega-0 --width 1280 --height 720 --fps 60
```

This creates (default):

- frame socket: `$XDG_RUNTIME_DIR/metabonk/omega-0/frame.sock`
- env file: `$XDG_RUNTIME_DIR/metabonk/omega-0/compositor.env`

## Run via launcher

Enable the path for workers:

```bash
./start --synthetic-eye --workers 1
```

As of now this is the default; disable with `--no-synthetic-eye`.

Or (lower-level):

```bash
python scripts/start_omega.py --mode train --workers 1 --synthetic-eye
```

Key env vars:

- `METABONK_FRAME_SOURCE=synthetic_eye`
- `METABONK_FRAME_SOCK=$XDG_RUNTIME_DIR/metabonk/<instance_id>/frame.sock`

## ABI

See `docs/synthetic_eye_abi.md`.
