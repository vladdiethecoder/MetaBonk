"""CUDA-side import + explicit sync for Synthetic Eye frames."""

from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from cuda.bindings import driver  # type: ignore

from .cuda_interop import (
    CudaExternalFrame,
    CudaExternalSemaphore,
    create_event,
    destroy_event,
    create_stream,
    import_dmabuf_as_buffer,
    import_dmabuf_as_mipmapped_array,
    import_external_semaphore_fd,
    query_event,
    record_event,
    signal_external_semaphore,
    stream_synchronize,
    wait_external_semaphore,
)
from .synthetic_eye_stream import SyntheticEyeFrame


@dataclass
class SyntheticEyeCudaHandle:
    frame: SyntheticEyeFrame
    ext_frame: CudaExternalFrame
    stream: driver.CUstream
    acquire: Optional[CudaExternalSemaphore] = None
    release: Optional[CudaExternalSemaphore] = None
    acquire_owned: bool = True
    release_owned: bool = True
    ext_frame_owned: bool = True

    def destroy(self) -> None:
        try:
            if self.acquire is not None and bool(self.acquire_owned):
                self.acquire.destroy()
        except Exception:
            pass
        try:
            if self.release is not None and bool(self.release_owned):
                self.release.destroy()
        except Exception:
            pass
        try:
            if bool(self.ext_frame_owned):
                self.ext_frame.destroy()
        except Exception:
            pass


@dataclass
class _PendingCleanup:
    """Resources that must stay alive until queued CUDA work completes."""

    event: driver.CUevent
    created_ts: float
    acquire: Optional[CudaExternalSemaphore] = None
    release: Optional[CudaExternalSemaphore] = None
    ext_frame: Optional[CudaExternalFrame] = None

    def destroy(self) -> None:
        try:
            if self.acquire is not None:
                self.acquire.destroy()
        except Exception:
            pass
        try:
            if self.release is not None:
                self.release.destroy()
        except Exception:
            pass
        try:
            if self.ext_frame is not None:
                self.ext_frame.destroy()
        except Exception:
            pass
        try:
            destroy_event(self.event)
        except Exception:
            pass


class SyntheticEyeCudaIngestor:
    def __init__(
        self,
        *,
        timeline_semaphores: bool = False,
        audit_log_path: Optional[str] = None,
        audit_interval_s: float = 2.0,
        stream: Optional[driver.CUstream] = None,
    ) -> None:
        self.timeline_semaphores = bool(timeline_semaphores)
        # Use a dedicated stream for DMA-BUF copy/import work. Do *not* share this stream with
        # "fence-only" handshakes: enqueueing acquire/release semaphore ops behind copies can
        # throttle the compositor frame rate to the (much slower) obs processing rate.
        self.stream = stream if stream is not None else create_stream(non_blocking=True)
        fence_priority = None
        try:
            high_prio = str(os.environ.get("METABONK_SYNTHETIC_EYE_FENCE_STREAM_HIGH_PRIORITY", "1") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        except Exception:
            high_prio = True
        if high_prio:
            try:
                err, least, greatest = driver.cuCtxGetStreamPriorityRange()
                if int(err) == 0:
                    fence_priority = int(greatest)
            except Exception:
                fence_priority = None
        self._fence_stream = create_stream(non_blocking=True, priority=fence_priority)
        self._audit_log_path = str(audit_log_path or "").strip() or None
        self._audit_interval_s = float(audit_interval_s)
        self._last_audit_ts = 0.0
        self.import_ok_count: int = 0
        self.import_fail_count: int = 0
        self.last_modifier: int = 0
        self.last_fourcc: int = 0
        self.last_frame_id: int = 0
        self.last_width: int = 0
        self.last_height: int = 0
        self.last_size_bytes: int = 0
        # The producer (Smithay Eye) waits on the consumer release semaphore before reusing a slot.
        # If we don't complete the CUDA signal promptly, the compositor stalls after a few frames.
        #
        # `cudaStreamSynchronize()` via cuda-python bindings may hold the GIL; use libcudart via ctypes
        # (which releases the GIL) to avoid freezing the worker's HTTP/heartbeat threads.
        self._strict_fence_sync = str(os.environ.get("METABONK_SYNTHETIC_EYE_STRICT_FENCE_SYNC", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # Never block the Synthetic Eye drain thread indefinitely. Cleanup is deferred and reaped
        # opportunistically using CUDA events + cuEventQuery(). When the queue grows too large,
        # treat it as a stall and let the worker restart.
        try:
            self._max_pending_cleanup = int(os.environ.get("METABONK_SYNTHETIC_EYE_MAX_PENDING_CLEANUP", "256"))
        except Exception:
            self._max_pending_cleanup = 256
        self._pending: list[_PendingCleanup] = []
        self._debug_fds = str(os.environ.get("METABONK_SYNTHETIC_EYE_DEBUG_FD", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._warned_no_fence: bool = False
        self._logged_quality_debug: bool = False
        # Cache CUDA external memory imports by dma-buf identity (device+inode).
        #
        # Synthetic Eye exports a small ring of slots and re-exports the same underlying dma-buf
        # objects each frame. Re-importing external memory on every frame is expensive and can
        # throttle export FPS under multi-worker load. Keep these imports alive for the lifetime
        # of the worker and only import once per unique dma-buf object.
        self._cache_extmem = str(os.environ.get("METABONK_SYNTHETIC_EYE_CACHE_EXTMEM", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._extmem_cache: dict[tuple[int, int], CudaExternalFrame] = {}
        # Cache CUDA external semaphore imports by fence identity (device+inode).
        #
        # Exported Vulkan semaphores are reused per slot in the compositor; re-importing them on
        # every frame is expensive and can dominate multi-worker throughput.
        self._cache_semaphores = str(os.environ.get("METABONK_SYNTHETIC_EYE_CACHE_SEMAPHORES", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._semaphore_cache: dict[tuple[int, int], CudaExternalSemaphore] = {}

    def _debug_fd(self, fd: int, name: str) -> None:
        if not self._debug_fds:
            return
        try:
            import stat

            st = os.fstat(int(fd))
            mode = oct(int(st.st_mode))
            print(
                f"[DEBUG] {name} fd={int(fd)} valid=True size={int(st.st_size)} mode={mode}",
                flush=True,
            )
        except OSError as e:
            print(f"[DEBUG] {name} fd={int(fd)} INVALID: {e}", flush=True)

    def _safe_import_semaphore(self, fd: int, name: str) -> tuple[Optional[CudaExternalSemaphore], bool]:
        """Best-effort import of external semaphores.

        Returns (semaphore, owned) where owned indicates whether the caller should destroy the
        imported semaphore after use.
        """
        try:
            fd_i = int(fd)
        except Exception:
            fd_i = -1
        if fd_i < 0:
            return None, False
        cache_key: Optional[tuple[int, int]] = None
        try:
            st = os.fstat(fd_i)
            cache_key = (int(st.st_dev), int(st.st_ino))
        except OSError:
            if self._debug_fds:
                print(f"[WARN] {name} fence fd invalid (fd={fd_i}); skipping", flush=True)
            return None, False
        if self._cache_semaphores and cache_key is not None:
            try:
                cached = self._semaphore_cache.get(cache_key)
            except Exception:
                cached = None
            if cached is not None:
                return cached, False
        try:
            sem = import_external_semaphore_fd(fd_i, timeline=self.timeline_semaphores)
        except Exception as e:
            if self._debug_fds:
                print(f"[WARN] {name} fence import failed (fd={fd_i}): {e}", flush=True)
            return None, False

        if self._cache_semaphores and cache_key is not None:
            try:
                self._semaphore_cache[cache_key] = sem
                return sem, False
            except Exception:
                pass
        return sem, True

    def _reap_pending(self) -> None:
        """Destroy resources whose CUDA work has completed (non-blocking)."""
        if not self._pending:
            return
        keep: list[_PendingCleanup] = []
        try:
            not_ready = int(getattr(driver.CUresult, "CUDA_ERROR_NOT_READY"))
        except Exception:
            not_ready = 600
        for p in self._pending:
            try:
                err_i = int(query_event(p.event))
                # Success => ready; NotReady/other => keep for now.
                if err_i == 0:
                    p.destroy()
                    continue
                if err_i == not_ready:
                    keep.append(p)
                else:
                    # Treat other errors as terminal; destroy to avoid unbounded growth.
                    p.destroy()
            except Exception:
                # If querying fails, keep for now.
                keep.append(p)
        self._pending = keep

        # If pending grows without being reaped, surface it as a hard failure.
        if len(self._pending) > max(32, int(self._max_pending_cleanup)):
            raise RuntimeError(f"synthetic_eye CUDA fence cleanup backlog too large: {len(self._pending)}")

    def on_reset(self) -> None:
        """Best-effort cleanup on compositor RESET (reason=2)."""
        try:
            self._reap_pending()
        except Exception:
            pass

    def note_frame(self, frame: SyntheticEyeFrame) -> None:
        """Record liveness for a frame without importing/syncing.

        Used for soak/health monitoring when capture/inference is disabled but we still want
        to prove the compositor is producing DMA-BUF frames and that the worker remains alive.
        """
        self.last_modifier = int(frame.modifier)
        self.last_fourcc = int(frame.drm_fourcc)
        self.last_frame_id = int(frame.frame_id)
        self.last_width = int(frame.width)
        self.last_height = int(frame.height)
        self.last_size_bytes = int(getattr(frame, "size_bytes", 0) or 0)
        self.import_ok_count += 1
        self._maybe_audit()

    def _maybe_audit(self) -> None:
        if not self._audit_log_path:
            return
        now = time.time()
        if (now - self._last_audit_ts) < self._audit_interval_s:
            return
        self._last_audit_ts = now
        try:
            Path(self._audit_log_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                extmem_cache = int(len(self._extmem_cache)) if self._cache_extmem else 0
            except Exception:
                extmem_cache = 0
            try:
                semaphore_cache = int(len(self._semaphore_cache)) if self._cache_semaphores else 0
            except Exception:
                semaphore_cache = 0
            try:
                pending = int(len(self._pending))
            except Exception:
                pending = 0
            with open(self._audit_log_path, "a", encoding="utf-8") as f:
                f.write(
                    "synthetic_eye_dmabuf "
                    f"import_ok={int(self.import_ok_count)} "
                    f"import_fail={int(self.import_fail_count)} "
                    f"last_frame_id={int(self.last_frame_id)} "
                    f"width={int(self.last_width)} "
                    f"height={int(self.last_height)} "
                    f"size_bytes={int(self.last_size_bytes)} "
                    f"drm_fourcc=0x{int(self.last_fourcc) & 0xFFFFFFFF:08x} "
                    f"modifier=0x{int(self.last_modifier) & 0xFFFFFFFFFFFFFFFF:016x} "
                    f"extmem_cache={int(extmem_cache)} "
                    f"semaphore_cache={int(semaphore_cache)} "
                    f"pending={int(pending)} "
                    f"ts={int(now)}\n"
                )
        except Exception:
            return

    def begin(self, frame: SyntheticEyeFrame) -> SyntheticEyeCudaHandle:
        """Import DMA-BUF + semaphores and wait on the acquire fence."""
        self._reap_pending()
        self.last_modifier = int(frame.modifier)
        self.last_fourcc = int(frame.drm_fourcc)
        self.last_frame_id = int(frame.frame_id)
        self.last_width = int(frame.width)
        self.last_height = int(frame.height)
        self.last_size_bytes = int(getattr(frame, "size_bytes", 0) or 0)
        if not self._logged_quality_debug:
            self._logged_quality_debug = True
            try:
                print(
                    "[QUALITY_DEBUG] "
                    f"synthetic_eye frame={int(frame.frame_id)} "
                    f"src={int(frame.width)}x{int(frame.height)} "
                    f"stride={int(frame.stride)} "
                    f"fourcc=0x{int(frame.drm_fourcc) & 0xFFFFFFFF:08x} "
                    f"modifier=0x{int(frame.modifier) & 0xFFFFFFFFFFFFFFFF:016x} "
                    f"target={str(os.environ.get('METABONK_STREAM_NVENC_TARGET_SIZE', '') or '').strip() or 'none'}",
                    flush=True,
                )
            except Exception:
                pass
        if self._debug_fds and int(frame.modifier) == 0:
            print(
                f"[WARN] Synthetic Eye modifier=0 (linear) for frame {int(frame.frame_id)}",
                flush=True,
            )
        dmabuf_fd = int(frame.dmabuf_fd)
        dmabuf_fd_dup: Optional[int] = None
        cache_key: Optional[tuple[int, int]] = None
        if self._cache_extmem:
            try:
                st = os.fstat(int(dmabuf_fd))
                cache_key = (int(st.st_dev), int(st.st_ino))
            except Exception:
                cache_key = None
        ext_frame_owned = True
        use_dup = str(os.environ.get("METABONK_SYNTHETIC_EYE_DUP_FD", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            ext_frame = None
            if cache_key is not None:
                try:
                    ext_frame = self._extmem_cache.get(cache_key)
                except Exception:
                    ext_frame = None
            if ext_frame is None:
                if use_dup:
                    try:
                        dmabuf_fd_dup = os.dup(dmabuf_fd)
                        dmabuf_fd = int(dmabuf_fd_dup)
                    except Exception:
                        dmabuf_fd_dup = None
                # Map DMA-BUF into CUDA. Prefer mipmapped array for tiled modifiers.
                if int(frame.modifier) != 0:
                    # RGBA8 (8 bits per channel). Order depends on producer format; treat as 4x8-bit for now.
                    ext_frame = import_dmabuf_as_mipmapped_array(
                        dmabuf_fd,
                        int(frame.size_bytes),
                        int(frame.width),
                        int(frame.height),
                    )
                else:
                    ext_frame = import_dmabuf_as_buffer(dmabuf_fd, int(frame.size_bytes))
                if cache_key is not None and self._cache_extmem:
                    try:
                        self._extmem_cache[cache_key] = ext_frame
                        ext_frame_owned = False
                    except Exception:
                        ext_frame_owned = True
                else:
                    ext_frame_owned = True
            else:
                ext_frame_owned = False
            # Import semaphores after the memory mapping succeeds.
            #
            # Some driver stacks treat external semaphore FDs as "consume-on-import"; if we import
            # the release fence first and then fail during memory import, we can no longer signal
            # release in a fallback path, deadlocking the producer's buffer pool.
            acquire, acquire_owned = self._safe_import_semaphore(frame.acquire_fence_fd, "ACQUIRE")
            release, release_owned = self._safe_import_semaphore(frame.release_fence_fd, "RELEASE")
            if acquire is not None:
                # Wait until compositor finished producing the frame.
                wait_external_semaphore(acquire, stream=self.stream, value=None)
            elif self._strict_fence_sync and not self._warned_no_fence:
                self._warned_no_fence = True
                print(
                    "[WARN] Synthetic Eye missing/invalid acquire fence; running unsynchronized",
                    flush=True,
                )
        except Exception:
            self.import_fail_count += 1
            self._maybe_audit()
            self._debug_fd(frame.dmabuf_fd, "DMABUF_FD")
            self._debug_fd(frame.acquire_fence_fd, "ACQUIRE_FENCE_FD")
            self._debug_fd(frame.release_fence_fd, "RELEASE_FENCE_FD")
            raise
        finally:
            if dmabuf_fd_dup is not None:
                try:
                    os.close(int(dmabuf_fd_dup))
                except Exception:
                    pass

        self.import_ok_count += 1
        self._maybe_audit()

        return SyntheticEyeCudaHandle(
            frame=frame,
            ext_frame=ext_frame,
            acquire=acquire,
            release=release,
            stream=self.stream,
            acquire_owned=bool(acquire_owned),
            release_owned=bool(release_owned),
            ext_frame_owned=bool(ext_frame_owned),
        )

    def end(self, handle: SyntheticEyeCudaHandle) -> None:
        """Signal release fence and cleanup imported CUDA objects."""
        self._reap_pending()
        if handle.release is not None:
            signal_external_semaphore(handle.release, stream=handle.stream, value=None)
        owned_acquire = handle.acquire if bool(getattr(handle, "acquire_owned", True)) else None
        owned_release = handle.release if bool(getattr(handle, "release_owned", True)) else None
        owned_ext_frame = handle.ext_frame if bool(getattr(handle, "ext_frame_owned", True)) else None
        if owned_acquire is None and owned_release is None and owned_ext_frame is None:
            return
        # Record an event after the release signal so we can destroy imported resources once
        # queued GPU work is done, without blocking the worker threads indefinitely.
        try:
            ev = create_event()
            record_event(ev, stream=handle.stream)
            # Store for deferred cleanup.
            self._pending.append(
                _PendingCleanup(
                    event=ev,
                    created_ts=time.time(),
                    acquire=owned_acquire,
                    release=owned_release,
                    ext_frame=owned_ext_frame,
                )
            )
        except Exception:
            # Fallback: if we can't create events, synchronize (may block, but preserves correctness).
            if self._strict_fence_sync:
                stream_synchronize(handle.stream)
            handle.destroy()

    def handshake_only(self, frame: SyntheticEyeFrame) -> None:
        """Service acquire/release fences without importing the DMA-BUF payload.

        This keeps the compositor buffer pool flowing even if the main rollout loop is busy.
        """
        self._reap_pending()
        self.last_modifier = int(frame.modifier)
        self.last_fourcc = int(frame.drm_fourcc)
        self.last_frame_id = int(frame.frame_id)
        self.last_width = int(frame.width)
        self.last_height = int(frame.height)
        self.last_size_bytes = int(getattr(frame, "size_bytes", 0) or 0)
        try:
            acquire, acquire_owned = self._safe_import_semaphore(frame.acquire_fence_fd, "ACQUIRE")
            release, release_owned = self._safe_import_semaphore(frame.release_fence_fd, "RELEASE")
            # Enqueue async wait/signal on a dedicated fence stream; do not synchronize per-frame.
            if acquire is not None:
                wait_external_semaphore(acquire, stream=self._fence_stream, value=None)
            if release is not None:
                signal_external_semaphore(release, stream=self._fence_stream, value=None)
            if bool(acquire_owned) or bool(release_owned):
                # Record event and defer destruction until the GPU stream has progressed.
                ev = create_event()
                record_event(ev, stream=self._fence_stream)
                self._pending.append(
                    _PendingCleanup(
                        event=ev,
                        created_ts=time.time(),
                        acquire=acquire if bool(acquire_owned) else None,
                        release=release if bool(release_owned) else None,
                    )
                )
        except Exception:
            self.import_fail_count += 1
            self._maybe_audit()
            raise

        self.import_ok_count += 1
        self._maybe_audit()
