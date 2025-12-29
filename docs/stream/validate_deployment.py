#!/usr/bin/env python3
"""
MetaBonk NVENC Deployment Validation
Validates that NVENC session management is working correctly
"""

from __future__ import annotations

import argparse
import requests
import json
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def _nvml_gpu_index() -> Optional[int]:
    raw = str(os.environ.get("METABONK_NVML_GPU_INDEX", "") or "").strip()
    if raw:
        try:
            return int(raw)
        except Exception:
            return None
    cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if not cvd:
        return 0
    first = cvd.split(",")[0].strip()
    if not first:
        return 0
    try:
        return int(first)
    except Exception:
        return None


def _nvenc_limit_env() -> int:
    try:
        v = int(os.environ.get("METABONK_NVENC_MAX_SESSIONS", "0"))
    except Exception:
        v = 0
    return max(0, int(v))


def _nvml_nvenc_sessions_used(*, gpu_index: int) -> Optional[int]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        stats = pynvml.nvmlDeviceGetEncoderStats(handle)
        if isinstance(stats, tuple) and stats:
            return int(stats[0])
        if hasattr(stats, "sessionCount"):
            return int(getattr(stats, "sessionCount"))
        return None
    except Exception:
        return None


def _nvml_gpu_name(*, gpu_index: int) -> Optional[str]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            return name.decode("utf-8", "replace")
        return str(name)
    except Exception:
        return None


def _default_orchestrator_url() -> str:
    return str(os.environ.get("ORCHESTRATOR_URL") or os.environ.get("METABONK_ORCHESTRATOR_URL") or "http://127.0.0.1:8040").rstrip("/")


def get_workers_from_orchestrator(*, orch_url: str) -> List[Dict[str, Any]]:
    """Return worker records from orchestrator `/workers` (best-effort)."""
    try:
        resp = requests.get(f"{orch_url}/workers", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return []
    except Exception as e:
        print_error(f"Failed to get worker list from orchestrator ({orch_url}): {e}")
        return []

    out: List[Dict[str, Any]] = []
    for _wid, hb in data.items():
        if not isinstance(hb, dict):
            continue
        out.append(hb)
    return out


def get_worker_status(*, control_url: str) -> Dict[str, Any]:
    """Get individual worker status via worker `/status` (best-effort)."""
    try:
        resp = requests.get(f"{control_url.rstrip('/')}/status", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print_warning(f"Failed to get status from {control_url}: {e}")
        return {}

def validate_nvenc_sessions():
    """Validate NVENC session management"""
    print_header("1. NVENC SESSION MANAGEMENT")
    
    max_sessions = _nvenc_limit_env()
    allow_cpu = str(os.environ.get("METABONK_STREAM_ALLOW_CPU_FALLBACK", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    gpu_index = _nvml_gpu_index()
    if gpu_index is not None:
        name = _nvml_gpu_name(gpu_index=gpu_index) or "unknown"
        used = _nvml_nvenc_sessions_used(gpu_index=gpu_index)
        print(f"GPU: index={gpu_index} name={name}")
        print(f"NVML encoder sessions used: {used}")
    else:
        used = None
        print_warning("NVML GPU index unknown (set METABONK_NVML_GPU_INDEX if needed)")

    print(f"METABONK_NVENC_MAX_SESSIONS: {max_sessions}")
    print(f"METABONK_STREAM_ALLOW_CPU_FALLBACK: {1 if allow_cpu else 0}")

    if max_sessions <= 0:
        print_warning("NVENC limiting is disabled (METABONK_NVENC_MAX_SESSIONS=0). This is OK on pro GPUs, risky on GeForce.")
        return True

    if used is not None and int(used) > int(max_sessions):
        print_warning(f"NVENC sessions in use exceed MetaBonk limit ({used}>{max_sessions}); external encoders may be running.")
        return True

    print_success("NVENC limit is configured")
    return True

def validate_worker_streams():
    """Validate individual worker streaming status"""
    print_header("2. WORKER STREAMING STATUS")
    
    if not _CTX:
        print_error("internal error: validation context missing")
        return False

    workers = _CTX.get("workers", [])
    if not workers:
        print_error("No workers found")
        return False

    print(f"Found {len(workers)} workers\n")
    
    streaming_workers = 0
    failed_workers = 0
    optimal_backend_count = 0
    
    for hb in workers:
        instance_id = str(hb.get("instance_id") or "")
        control_url = str(hb.get("control_url") or "")
        status = get_worker_status(control_url=control_url) if control_url else {}
        
        if not status:
            print_error(f"{instance_id or control_url}: Status unavailable")
            failed_workers += 1
            continue
        
        stream_enabled = status.get('stream_enabled', False)
        stream_backend = status.get('stream_backend', 'unknown')
        stream_error = status.get('streamer_last_error') or status.get('stream_error')
        frames_fps = status.get('frames_fps', 0)
        frames_dropped = status.get('frames_dropped', 0)
        
        label = instance_id or str(status.get("instance_id") or control_url)
        print(f"\n{Colors.BOLD}{label}:{Colors.END}")
        print(f"  Stream Enabled: {stream_enabled}")
        print(f"  Backend: {stream_backend}")
        print(f"  FPS: {frames_fps:.1f}")
        print(f"  Dropped Frames: {frames_dropped}")
        try:
            nvenc_used = status.get("nvenc_sessions_used")
        except Exception:
            nvenc_used = None
        if nvenc_used is not None:
            print(f"  NVENC Sessions Used (NVML): {nvenc_used}")
        
        if stream_error:
            print_warning(f"  Error: {stream_error}")
            failed_workers += 1
        elif stream_enabled and 'gst:cuda' in stream_backend:
            print_success(f"  ✓ Optimal CUDA → GStreamer path")
            streaming_workers += 1
            optimal_backend_count += 1
        elif stream_enabled and 'ffmpeg' in stream_backend:
            print_warning(f"  Suboptimal FFmpeg path (should use GStreamer)")
            streaming_workers += 1
        elif stream_enabled:
            print_info(f"  Streaming active")
            streaming_workers += 1
        else:
            print_info(f"  Streaming disabled (graceful fallback)")
    
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Total Workers: {len(workers)}")
    print(f"  Streaming: {streaming_workers}")
    print(f"  Optimal Backend: {optimal_backend_count}")
    print(f"  Failed: {failed_workers}")
    
    if failed_workers > 0:
        print_error(f"{failed_workers} workers have errors")
        return False
    
    if optimal_backend_count > 0:
        print_success(f"{optimal_backend_count} workers using optimal CUDA path")
    
    return True

def validate_fps_quality():
    """Validate FPS and stream quality"""
    print_header("3. FPS & QUALITY VALIDATION")
    
    if not _CTX:
        print_error("internal error: validation context missing")
        return False

    workers = _CTX.get("workers", [])
    
    low_fps_workers = []
    high_drops_workers = []
    good_workers = []
    
    for hb in workers:
        control_url = str(hb.get("control_url") or "")
        status = get_worker_status(control_url=control_url) if control_url else {}
        
        if not status or not status.get('stream_enabled'):
            continue
        
        fps = status.get('frames_fps', 0)
        dropped = status.get('frames_dropped', 0)
        total = status.get('frames_ok', 0) + dropped
        
        drop_rate = (dropped / total * 100) if total > 0 else 0
        
        if fps < 30:
            low_fps_workers.append((worker_id, fps))
        elif drop_rate > 1.0:
            high_drops_workers.append((worker_id, drop_rate))
        elif fps >= 55:
            good_workers.append((str(status.get("instance_id") or control_url), fps))
    
    if good_workers:
        print_success(f"{len(good_workers)} workers achieving ≥55 FPS:")
        for worker_id, fps in good_workers:
            print(f"  • {worker_id}: {fps:.1f} FPS")
    
    if low_fps_workers:
        print_error(f"{len(low_fps_workers)} workers with low FPS (<30):")
        for worker_id, fps in low_fps_workers:
            print(f"  • {worker_id}: {fps:.1f} FPS")
    
    if high_drops_workers:
        print_warning(f"{len(high_drops_workers)} workers with high drop rate (>1%):")
        for worker_id, rate in high_drops_workers:
            print(f"  • {worker_id}: {rate:.1f}% dropped")
    
    return len(low_fps_workers) == 0

def validate_resolution():
    """Validate capture and stream resolutions"""
    print_header("4. RESOLUTION VALIDATION")
    
    if not _CTX:
        print_error("internal error: validation context missing")
        return False

    workers = _CTX.get("workers", [])
    for hb in workers[:1]:  # Check first worker as representative
        control_url = str(hb.get("control_url") or "")
        status = get_worker_status(control_url=control_url) if control_url else {}
        
        if not status:
            continue
        
        pixel_src_w = status.get('pixel_src_width', 0)
        pixel_src_h = status.get('pixel_src_height', 0)
        spectator_w = status.get('spectator_width', 0)
        spectator_h = status.get('spectator_height', 0)
        pixel_obs_w = status.get('pixel_obs_width', 0)
        pixel_obs_h = status.get('pixel_obs_height', 0)
        
        print(f"{str(status.get('instance_id') or control_url)}:")
        print(f"  Source Resolution: {pixel_src_w}x{pixel_src_h}")
        print(f"  Spectator Output: {spectator_w}x{spectator_h}")
        print(f"  Agent Observation: {pixel_obs_w}x{pixel_obs_h}")
        
        if pixel_src_w >= 1280 and pixel_src_h >= 720:
            print_success("Source resolution is adequate (≥720p)")
        elif pixel_src_w >= 640:
            print_warning("Source resolution is low (<720p)")
        else:
            print_error(f"Source resolution is critically low ({pixel_src_w}x{pixel_src_h})")
            return False
        
        if spectator_w == 1920 and spectator_h == 1080:
            print_success("Spectator output is 1080p")
        else:
            print_info(f"Spectator output is {spectator_w}x{spectator_h}")
    
    return True

def check_ui_accessibility():
    """Check if UI endpoints are accessible"""
    print_header("5. UI ACCESSIBILITY")

    if not _CTX:
        print_error("internal error: validation context missing")
        return False

    orch_url = str(_CTX.get("orch_url") or "").rstrip("/")
    ui_url = str(_CTX.get("ui_url") or "").rstrip("/")
    endpoints = [
        ("Orchestrator /workers", f"{orch_url}/workers"),
        ("Orchestrator stream diagnostics", f"{orch_url}/api/diagnostics/stream-quality"),
        ("Stream Page", f"{ui_url}/stream"),
        ("Neural Broadcast", f"{ui_url}/neural/broadcast"),
    ]
    
    all_ok = True
    
    for name, url in endpoints:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print_success(f"{name}: {url}")
            else:
                print_warning(f"{name}: {url} (HTTP {resp.status_code})")
                all_ok = False
        except Exception as e:
            print_error(f"{name}: {url} - {e}")
            all_ok = False
    
    return all_ok


_CTX: Optional[Dict[str, Any]] = None


def main():
    global _CTX
    ap = argparse.ArgumentParser(description="Validate MetaBonk NVENC deployment (best-effort).")
    ap.add_argument("--orch-url", default=_default_orchestrator_url(), help="Orchestrator base URL (default: ORCHESTRATOR_URL or http://127.0.0.1:8040)")
    ap.add_argument("--ui-url", default="http://127.0.0.1:5173", help="UI base URL (default: http://127.0.0.1:5173)")
    ap.add_argument("--workers", type=int, default=0, help="Fallback: number of workers to probe directly (uses base port)")
    ap.add_argument("--worker-host", default="127.0.0.1", help="Fallback: worker host for direct probing")
    ap.add_argument("--worker-base-port", type=int, default=5000, help="Fallback: base port for direct probing")
    args = ap.parse_args()

    orch_url = str(args.orch_url or "").rstrip("/")
    ui_url = str(args.ui_url or "").rstrip("/")

    workers = get_workers_from_orchestrator(orch_url=orch_url)
    if not workers and int(args.workers) > 0:
        host = str(args.worker_host or "127.0.0.1").strip() or "127.0.0.1"
        base_port = int(args.worker_base_port)
        workers = [
            {
                "instance_id": f"worker-{i}",
                "control_url": f"http://{host}:{base_port + i}",
            }
            for i in range(max(0, int(args.workers)))
        ]
    _CTX = {"orch_url": orch_url, "ui_url": ui_url, "workers": workers}

    print(f"\n{Colors.BOLD}MetaBonk NVENC Deployment Validation{Colors.END}")
    print(f"Testing session management and streaming quality\n")
    
    results = {
        'nvenc_sessions': validate_nvenc_sessions(),
        'worker_streams': validate_worker_streams(),
        'fps_quality': validate_fps_quality(),
        'resolution': validate_resolution(),
        'ui_access': check_ui_accessibility(),
    }
    
    print_header("FINAL RESULTS")
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test.replace('_', ' ').title()}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} tests passed{Colors.END}\n")
    
    if passed == total:
        print_success("🎉 All validation checks passed!")
        print_info("MetaBonk is ready for production use")
        return 0
    else:
        print_error(f"⚠️  {total - passed} validation checks failed")
        print_info("Review failures above and address issues")
        return 1

if __name__ == '__main__':
    sys.exit(main())
