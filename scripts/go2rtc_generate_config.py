#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _add_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _dump_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _selinux_enabled() -> bool:
    if shutil.which("selinuxenabled") is None:
        return False
    try:
        proc = subprocess.run(
            ["selinuxenabled"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _relabel_fifo_dir_for_containers(fifo_dir: Path) -> None:
    """Best-effort: ensure FIFO dir is readable from containers on SELinux hosts."""
    if not _selinux_enabled():
        return
    if shutil.which("chcon") is None:
        return
    try:
        subprocess.run(
            ["chcon", "-R", "-t", "container_file_t", "-l", "s0", str(fifo_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


def main() -> int:
    repo_root = _add_repo_root()
    try:
        from src.streaming.fifo import ensure_fifo  # type: ignore
    except Exception as e:
        raise SystemExit("failed to import src.streaming.fifo (run from repo root)") from e

    parser = argparse.ArgumentParser(description="Generate go2rtc config for MetaBonk streams (FIFO or exec).")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("METABONK_DEFAULT_WORKERS", "6")))
    parser.add_argument("--instance-prefix", default=os.environ.get("METABONK_INSTANCE_PREFIX", "omega"))
    parser.add_argument(
        "--mode",
        default=os.environ.get("METABONK_GO2RTC_MODE", "fifo"),
        help="Stream source mode: fifo (default) or exec.",
    )
    parser.add_argument(
        "--exec-cmd-template",
        default=os.environ.get("METABONK_GO2RTC_EXEC_CMD", ""),
        help="Exec mode only. Command template; supports {instance_id}.",
    )
    parser.add_argument(
        "--exec-profile",
        default=os.environ.get("METABONK_GO2RTC_EXEC_PROFILE", ""),
        help="Exec mode only. Convenience profile: headless-agent or headless-agent-mpegts.",
    )
    parser.add_argument(
        "--exec-wrap",
        default=os.environ.get("METABONK_GO2RTC_EXEC_WRAP", "raw"),
        help="Exec mode only. raw (default) or mpegts.",
    )
    parser.add_argument(
        "--exec-wrapper",
        default=os.environ.get("METABONK_GO2RTC_EXEC_WRAPPER", "scripts/go2rtc_exec_mpegts.sh"),
        help="Exec mode only. Wrapper script path (used when --exec-wrap=mpegts).",
    )
    parser.add_argument(
        "--fifo-dir",
        default=os.environ.get("METABONK_STREAM_FIFO_DIR", str(repo_root / "temp" / "streams")),
        help="Host FIFO directory (created if missing).",
    )
    parser.add_argument(
        "--fifo-container",
        default=os.environ.get("METABONK_FIFO_CONTAINER", "mpegts"),
        help="FIFO container format: h264 (raw Annex-B) or mpegts.",
    )
    parser.add_argument(
        "--container-fifo-dir",
        default=os.environ.get("METABONK_GO2RTC_FIFO_DIR", "/streams"),
        help="FIFO directory path inside the go2rtc container (must match docker-compose mount).",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("METABONK_GO2RTC_CONFIG", str(repo_root / "temp" / "go2rtc.yaml")),
        help="Output go2rtc.yaml path.",
    )
    args = parser.parse_args()

    workers = max(0, int(args.workers))
    prefix = str(args.instance_prefix or "omega").strip() or "omega"
    fifo_dir = Path(args.fifo_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    container_fifo_dir = str(args.container_fifo_dir or "/streams").rstrip("/") or "/streams"
    fifo_container = str(args.fifo_container or "mpegts").strip().lower()
    if fifo_container in ("ts", "mpegts"):
        fifo_container = "mpegts"
        fifo_ext = "ts"
    else:
        fifo_container = "h264"
        fifo_ext = "h264"
    mode = str(args.mode or "fifo").strip().lower()
    exec_wrap = str(args.exec_wrap or "raw").strip().lower()
    exec_wrapper = str(args.exec_wrapper or "").strip()
    exec_profile = str(args.exec_profile or "").strip().lower()

    if mode not in ("fifo", "exec"):
        raise SystemExit(f"unsupported go2rtc mode: {mode}")

    instance_ids: List[str] = [f"{prefix}-{i}" for i in range(workers)]
    if mode == "fifo":
        fifo_dir.mkdir(parents=True, exist_ok=True)
        for iid in instance_ids:
            ensure_fifo(str(fifo_dir / f"{iid}.{fifo_ext}"))
        _relabel_fifo_dir_for_containers(fifo_dir)
    else:
        cmd_template = str(args.exec_cmd_template or "").strip()
        if exec_profile:
            if exec_profile in ("headless-agent", "headless", "headless_egl"):
                # Run the in-repo producer. Use bash so the script doesn't need exec bits.
                cmd_template = "bash /app/scripts/go2rtc_exec_headless_agent.sh {instance_id}"
                exec_wrap = "raw"
            elif exec_profile in ("egl-vpf", "headless-vpf", "headless-egl-vpf"):
                # Full EGL→CUDA interop→PyNvVideoCodec demo path (Annex‑B H.264 to stdout).
                # Requires optional deps inside the go2rtc exec image/venv:
                #   PyNvVideoCodec + CV-CUDA + CuPy.
                cmd_template = "bash /app/scripts/go2rtc_exec_headless_agent.sh {instance_id} --encoder vpf"
                exec_wrap = "raw"
            elif exec_profile in ("headless-agent-mpegts", "headless-mpegts"):
                cmd_template = "bash /app/scripts/go2rtc_exec_headless_agent.sh {instance_id}"
                exec_wrap = "mpegts"
                exec_wrapper = exec_wrapper or "scripts/go2rtc_exec_mpegts.sh"
            else:
                raise SystemExit(f"unsupported exec profile: {exec_profile}")
        if not cmd_template:
            raise SystemExit("exec mode requires --exec-cmd-template/--exec-profile (or METABONK_GO2RTC_EXEC_CMD)")
        if exec_wrap not in ("raw", "mpegts"):
            raise SystemExit(f"unsupported exec wrap: {exec_wrap}")
        if exec_wrap == "mpegts" and not exec_wrapper:
            raise SystemExit("exec wrap=mpegts requires --exec-wrapper path")

    # IMPORTANT: keep exec sources fixed to known FIFO paths inside the container.
    # Do not include any user-controlled input in the exec string.
    lines: List[str] = []
    lines.append("api:")
    lines.append('  listen: ":1984"')
    lines.append("rtsp:")
    lines.append('  listen: ":8554"')
    lines.append("webrtc:")
    lines.append('  listen: ":8555"')
    lines.append("streams:")
    for iid in instance_ids:
        if mode == "fifo":
            fifo_inside = f"{container_fifo_dir}/{iid}.{fifo_ext}"
            if fifo_container == "h264":
                # #video=h264 disables probing; #raw tells go2rtc to packetize as-is.
                lines.append(f"  {iid}: exec:cat {fifo_inside}#video=h264#raw")
            else:
                # MPEG-TS includes timing info; allow go2rtc to probe.
                lines.append(f"  {iid}: exec:cat {fifo_inside}")
        else:
            cmd = cmd_template.format(instance_id=iid)
            if exec_wrap == "mpegts":
                cmd = f"{exec_wrapper} -- {cmd}"
            # For raw elementary streams, disable probing. For MPEG-TS wrapping, let go2rtc probe.
            suffix = "#video=h264#raw" if exec_wrap == "raw" else ""
            lines.append(f"  {iid}: exec:{cmd}{suffix}")
    lines.append("")

    _dump_yaml(out_path, "\n".join(lines))
    print(f"[go2rtc_generate_config] wrote {out_path}")
    if mode == "fifo":
        print(f"[go2rtc_generate_config] fifo dir {fifo_dir}")
    else:
        print(f"[go2rtc_generate_config] exec cmd template {cmd_template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
