from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path
from typing import Iterable, Optional


def _read_cmdline(pid: int) -> str:
    p = Path(f"/proc/{int(pid)}/cmdline")
    try:
        data = p.read_bytes()
    except Exception:
        return ""
    parts = [c.decode("utf-8", "replace") for c in data.split(b"\x00") if c]
    return " ".join(parts).strip()


def _iter_nvenc_sessions(*, gpu_index: int) -> Iterable[tuple[int, str]]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        sessions = pynvml.nvmlDeviceGetEncoderSessions(handle)
    except Exception:
        return []

    out: list[tuple[int, str]] = []
    for s in sessions or []:
        try:
            pid = int(getattr(s, "pid"))
        except Exception:
            continue
        try:
            st = str(getattr(s, "sessionType", "") or "")
        except Exception:
            st = ""
        out.append((pid, st))
    return out


def _pick_gpu_index(args_gpu: Optional[int]) -> int:
    if args_gpu is not None:
        return int(args_gpu)
    raw = str(os.environ.get("METABONK_NVML_GPU_INDEX", "") or "").strip()
    if raw:
        try:
            return int(raw)
        except Exception:
            return 0
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="List (and optionally kill) NVENC encoder session owners via NVML.")
    ap.add_argument("--gpu", type=int, default=None, help="NVML GPU index (default: METABONK_NVML_GPU_INDEX or 0)")
    ap.add_argument(
        "--kill",
        action="store_true",
        help="Send SIGTERM to suspected MetaBonk encoder owners (ffmpeg/gst-launch/python workers).",
    )
    ap.add_argument("--force", action="store_true", help="Use SIGKILL instead of SIGTERM (implies --kill).")
    ap.add_argument(
        "--only-metabonk",
        action="store_true",
        default=True,
        help="Only target processes that look like MetaBonk encoders (default: true).",
    )
    ap.add_argument(
        "--all",
        dest="only_metabonk",
        action="store_false",
        help="Target all encoder session owners (dangerous).",
    )
    args = ap.parse_args()

    gpu_index = _pick_gpu_index(args.gpu)
    sessions = list(_iter_nvenc_sessions(gpu_index=gpu_index))
    print(f"NVENC sessions (gpu_index={gpu_index}): {len(sessions)}")
    if not sessions:
        return 0

    for pid, st in sessions:
        cmd = _read_cmdline(pid)
        print(f"- pid={pid} type={st} cmd={cmd}")

    if not args.kill and not args.force:
        return 0

    sig = signal.SIGKILL if args.force else signal.SIGTERM
    killed = 0
    for pid, _st in sessions:
        cmd = _read_cmdline(pid).lower()
        if args.only_metabonk:
            if not any(tok in cmd for tok in ("metabonk", "megabonk", "src.worker.main", "ffmpeg", "gst-launch")):
                continue
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            continue
        except Exception as e:
            print(f"  WARN: failed to signal pid={pid}: {e}")
            continue
        killed += 1

    if killed:
        time.sleep(0.25)
        print(f"Signaled {killed} process(es) with {sig}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

