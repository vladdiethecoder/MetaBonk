# Zero‑Copy Gamescope Capture (NVIDIA)

This repo supports a GPU‑resident capture path via PipeWire DMABuf.

## Host prerequisites
- NVIDIA driver with DRM KMS enabled: kernel param `nvidia-drm.modeset=1`.
- Gamescope running (nested or `--headless`) and exporting a PipeWire node.
- PipeWire + WirePlumber on the host.

## Worker capture
`src/worker/stream.py` builds a GStreamer pipeline:

```
pipewiresrc path=$PIPEWIRE_NODE !
video/x-raw(memory:DMABuf),format=NV12 !
appsink emit-signals=true
```

Each `CapturedFrame` contains a duplicated DMA‑BUF FD plus caps metadata. No CPU
readback occurs unless you explicitly map the buffer yourself.

## CUDA interop
`src/worker/cuda_interop.py` provides helpers to import the FD into CUDA using
`cudaImportExternalMemory` and map it as:
- a linear buffer (`import_dmabuf_as_buffer`) for linear layouts, or
- a mipmapped array (`import_dmabuf_as_mipmapped_array`) for NVIDIA block‑linear
  layouts/modifiers.

## Docker / headless
Run workers in a container with the host GPU + PipeWire socket:

```
docker run -it --gpus all \
  --device /dev/dri \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-modeset \
  -v /run/user/1000/pipewire-0:/tmp/pipewire-0 \
  -e PIPEWIRE_RUNTIME_DIR=/tmp \
  -e PIPEWIRE_NODE=<gamescope-node-id> \
  metabonk-worker:latest
```

`PIPEWIRE_NODE` is the PipeWire node id or path for the Gamescope stream
(discover via `pw-cli ls Node` or `wpctl status` on the host).

## Notes
- If negotiation fails on NVIDIA, ensure the client requests
  `video/x-raw(memory:DMABuf)` and supports DRM modifiers.
- Avoid `x11grab`/CPU readbacks; they will stall the GPU and saturate PCIe.

