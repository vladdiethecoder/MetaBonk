# MetaBonk Synthetic Eye ABI (v1)

This ABI defines the **frame handoff contract** between a compositor process (producer)
and a worker process (consumer) using:

- **DMA-BUF** file descriptors for zero-copy GPU buffers.
- **Explicit sync** via exported **external semaphore FDs** ("fences") for producer→consumer (acquire)
  and consumer→producer (release) synchronization.
- A **Unix domain socket** transporting a small binary header + payload, with FDs passed via `SCM_RIGHTS`.

The goal is to keep this ABI stable so compositor and worker can evolve independently.

## Transport

- Unix domain stream socket (e.g. `$XDG_RUNTIME_DIR/metabonk/<id>/frame.sock`)
- Messages are sent as:
  - `HeaderV1` (fixed-size, little-endian)
  - `payload` (variable-size, little-endian)
  - `SCM_RIGHTS` ancillary data containing `fd_count` FDs

## HeaderV1 (fixed, little-endian)

| Field | Type | Notes |
|---|---:|---|
| `magic` | `[u8; 8]` | ASCII `MBEYEABI` |
| `version` | `u16` | `1` |
| `msg_type` | `u16` | enum below |
| `payload_len` | `u32` | bytes following the header |
| `fd_count` | `u32` | number of FDs attached via `SCM_RIGHTS` |

### Message types

| Name | Value |
|---|---:|
| `HELLO` | 1 |
| `HELLO_ACK` | 2 |
| `FRAME` | 3 |
| `RESET` | 4 |
| `PING` | 5 |
| `PONG` | 6 |

## HELLO / HELLO_ACK (optional handshake)

Both payloads are currently empty. (Reserved for future capability negotiation.)

## FRAME (per-frame payload + FDs)

### Payload (FrameV1, little-endian)

| Field | Type | Notes |
|---|---:|---|
| `frame_id` | `u64` | Monotonic id assigned by compositor |
| `width` | `u32` | pixels |
| `height` | `u32` | pixels |
| `drm_fourcc` | `u32` | DRM FourCC (e.g. `DRM_FORMAT_ARGB8888`) |
| `modifier` | `u64` | DRM format modifier (may be `DRM_FORMAT_MOD_LINEAR`) |
| `dmabuf_fd_count` | `u8` | number of DMA-BUF fds in FD list |
| `plane_count` | `u8` | number of planes described below |
| `reserved0` | `u16` | must be 0 |
| `planes[]` | `PlaneV1[plane_count]` | plane layout |

### PlaneV1 (repeated, little-endian)

| Field | Type | Notes |
|---|---:|---|
| `fd_index` | `u8` | index into DMA-BUF FD list (0..dmabuf_fd_count-1) |
| `reserved0` | `u8` | must be 0 |
| `reserved1` | `u16` | must be 0 |
| `stride` | `u32` | bytes per row for this plane |
| `offset` | `u32` | byte offset into the referenced DMA-BUF |
| `size_bytes` | `u32` | plane byte size (best-effort; may be 0) |
| `reserved2` | `u32` | must be 0 |

### FD list order for FRAME

The message includes `fd_count = dmabuf_fd_count + 2` file descriptors in `SCM_RIGHTS`,
in this exact order:

1. `dmabuf[0..dmabuf_fd_count-1]` (DMA-BUF fds)
2. `acquire_fence_fd` (producer → consumer)
3. `release_fence_fd` (consumer → producer)

The fence FDs are expected to be **Vulkan external semaphore FDs** (opaque or timeline),
which the worker imports via CUDA external semaphore APIs:

- wait on `acquire_fence_fd` before reading the image
- signal `release_fence_fd` after inference completes

## RESET (control message)

Sent by the compositor when it detects an unrecoverable display stack issue (e.g. XWayland stall)
and needs the worker to treat the segment as invalid (to avoid training on frozen frames).

### Payload (ResetV1)

| Field | Type | Notes |
|---|---:|---|
| `reason` | `u32` | reason code |
| `reserved0` | `u32` | must be 0 |

No FDs are attached.

Reason codes:

- `1`: XWayland stalled/restarted
- `2`: compositor output reset
