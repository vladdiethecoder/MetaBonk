"""Video to Trajectory Pipeline for Behavioral Cloning.

Extracts demonstration trajectories from gameplay videos:
- Frame extraction at configurable FPS
- (Optional) proxy action signals from visual analysis (for debugging only)
- GPU-accelerated optical flow (CUDA when available)
- Curated training data for diffusion policy

Usage:
    python scripts/video_to_trajectory.py --video-dir gameplay_videos/
    python scripts/video_to_trajectory.py --video gameplay_videos/megabonk_gameplay_0.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import struct

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import cv2
    HAS_CV2 = True
    # Check for CUDA support in OpenCV
    HAS_CUDA_FLOW = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
except ImportError:
    HAS_CV2 = False
    HAS_CUDA_FLOW = False
    cv2 = None

try:
    import torch
    HAS_TORCH = True
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    TORCH_DEVICE = None


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    # Extraction settings
    target_fps: float = 45.0          # Frames per second to extract (45 FPS sweet spot)
    resize: Tuple[int, int] = (224, 224)  # Target frame size
    
    # Proxy action inference (debug only).
    # Real action labels should come from learned IDM labeling in `scripts/video_pretrain.py`.
    use_optical_flow: bool = False
    use_vlm_labeling: bool = False    # Requires VLM API; debug only
    
    # Output
    output_dir: str = "rollouts/video_demos"
    frames_per_chunk: int = 1000      # Frames per output file

    # Audio extraction
    extract_audio: bool = True
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    
    # Perception for action inference
    movement_threshold: float = 5.0   # Pixels for movement detection
    action_smoothing: int = 3         # Frames to smooth actions
    
    # GPU Acceleration
    use_gpu: bool = True              # Use CUDA when available
    batch_size: int = 64              # Batch size for GPU processing
    num_workers: int = 4              # Parallel video decoders
    progress_interval: int = 1000     # Print progress every N frames


class ActionInferrer:
    """Infer pseudo-actions from video frames using visual analysis.
    
    Since we don't have actual input recordings, we only infer a coarse
    movement proxy from optical flow. We intentionally avoid inferring
    game-specific semantics like aiming, firing, or abilities.
    
    Uses CUDA-accelerated optical flow when available for 10-50x speedup.
    """
    
    def __init__(self, cfg: VideoConfig):
        self.cfg = cfg
        self.prev_frame = None
        self.prev_gray = None
        self.use_cuda = cfg.use_gpu and HAS_CUDA_FLOW
        
        # Initialize CUDA optical flow if available
        if self.use_cuda:
            try:
                self.cuda_flow = cv2.cuda.FarnebackOpticalFlow.create(
                    numLevels=3,
                    pyrScale=0.5,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0
                )
                print("[ActionInferrer] Using CUDA-accelerated optical flow")
            except Exception as e:
                print(f"[ActionInferrer] CUDA flow init failed: {e}, using CPU")
                self.use_cuda = False
                self.cuda_flow = None
        else:
            self.cuda_flow = None
            if not HAS_CUDA_FLOW:
                print("[ActionInferrer] Using CPU optical flow (OpenCV CUDA not available)")
        
    def infer_action(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Infer action from frame comparison.
        
        Returns:
            action: [delta_x, delta_y, aux0, aux1, aux2, aux3] in [-1, 1]
              - Only delta_x/delta_y are inferred (generic optical-flow proxy).
              - aux* channels remain 0 (not inferred here).
        """
        action = np.zeros(6, dtype=np.float32)
        
        if not HAS_CV2 or prev_frame is None:
            return action
        
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow
        if self.use_cuda and self.cuda_flow is not None:
            # CUDA path - significantly faster
            try:
                prev_gpu = cv2.cuda_GpuMat()
                curr_gpu = cv2.cuda_GpuMat()
                prev_gpu.upload(prev_gray)
                curr_gpu.upload(gray)
                
                flow_gpu = self.cuda_flow.calc(prev_gpu, curr_gpu, None)
                flow = flow_gpu.download()
            except Exception:
                # Fall back to CPU on error
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
        else:
            # CPU path
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        
        # Global motion (camera/player movement)
        h, w = flow.shape[:2]
        center_region = flow[h//4:3*h//4, w//4:3*w//4]  # Center crop
        
        mean_flow = np.mean(center_region, axis=(0, 1))
        
        # Invert to get player movement (camera follows player)
        # Normalize to [-1, 1]
        action[0] = np.clip(-mean_flow[0] / self.cfg.movement_threshold, -1, 1)  # move_x
        action[1] = np.clip(-mean_flow[1] / self.cfg.movement_threshold, -1, 1)  # move_y

        return action


    
    def smooth_actions(
        self,
        actions: List[np.ndarray],
        window: int = 3,
    ) -> List[np.ndarray]:
        """Apply temporal smoothing to action sequence."""
        if len(actions) < window:
            return actions
        
        smoothed = []
        for i in range(len(actions)):
            start = max(0, i - window // 2)
            end = min(len(actions), i + window // 2 + 1)
            smoothed.append(np.mean(actions[start:end], axis=0))
        
        return smoothed


def _ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def _extract_audio_pcm(
    video_path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> Optional[np.ndarray]:
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        print("[VideoToTrajectory] ffmpeg not found; audio extraction disabled")
        return None

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(int(channels)),
        "-ar",
        str(int(sample_rate)),
        "-f",
        "s16le",
        "-",
    ]
    try:
        res = subprocess.run(cmd, check=False, capture_output=True)
    except Exception as exc:
        print(f"[VideoToTrajectory] ffmpeg failed: {exc}")
        return None
    if res.returncode != 0 or not res.stdout:
        err = res.stderr.decode("utf-8", errors="ignore") if res.stderr else ""
        print(f"[VideoToTrajectory] ffmpeg audio decode failed: {err.strip()}")
        return None

    audio = np.frombuffer(res.stdout, dtype=np.int16)
    if channels > 1:
        # Downmix to mono.
        try:
            audio = audio.reshape(-1, int(channels)).mean(axis=1).astype(np.int16)
        except Exception:
            audio = audio.astype(np.int16)
    return audio


class VideoToTrajectory:
    """Extract demonstration trajectories from gameplay videos."""
    
    def __init__(self, cfg: Optional[VideoConfig] = None):
        self.cfg = cfg or VideoConfig()
        self.action_inferrer = ActionInferrer(self.cfg)
        
        # Ensure output directory exists
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        
    def process_video(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = -1.0,
    ) -> Dict[str, Any]:
        """Process a video file and extract trajectories.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (-1 for full video)
            
        Returns:
            Dict with processing statistics
        """
        if not HAS_CV2:
            print("[VideoToTrajectory] OpenCV not available. Install with: pip install opencv-python")
            return {"error": "OpenCV required"}
        
        video_path = Path(video_path)
        if not video_path.exists():
            return {"error": f"Video not found: {video_path}"}
        
        print(f"[VideoToTrajectory] Processing: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}
        
        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / source_fps if source_fps > 0 else 0
        
        print(f"  Source: {source_fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")

        audio_pcm = None
        samples_per_frame = None
        if bool(self.cfg.extract_audio):
            audio_pcm = _extract_audio_pcm(
                video_path,
                sample_rate=int(self.cfg.audio_sample_rate),
                channels=int(self.cfg.audio_channels),
            )
            if audio_pcm is not None:
                samples_per_frame = max(1, int(round(float(self.cfg.audio_sample_rate) / max(self.cfg.target_fps, 1e-3))))
                print(
                    f"  Audio: {len(audio_pcm):,} samples @ {int(self.cfg.audio_sample_rate)} Hz "
                    f"(spf={samples_per_frame})"
                )
        
        # Calculate frame skip
        frame_skip = max(1, int(source_fps / self.cfg.target_fps))
        
        # Set start position
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        # Calculate end frame
        if end_time > 0:
            end_frame = int(end_time * source_fps)
        else:
            end_frame = total_frames

        # Extract trajectory (stream to disk in chunks to avoid huge in-memory arrays)
        trajectory_id = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        chunk_idx = 0
        chunk_frames: List[np.ndarray] = []
        chunk_proxy_actions: List[np.ndarray] = []
        chunk_audio: List[np.ndarray] = []
        outputs: List[str] = []
        
        prev_frame = None
        frame_idx = 0
        extracted = 0
        start_time_proc = time.time()
        last_progress_time = start_time_proc
        
        target_extract = int(total_frames / frame_skip)  # Expected frames
        
        while True:
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break
            
            frame_idx += 1
            
            # Skip frames for target FPS
            if frame_idx % frame_skip != 0:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame_resized = cv2.resize(frame_rgb, self.cfg.resize)
            
            # Infer action from frame difference
            if self.cfg.use_optical_flow or self.cfg.use_vlm_labeling:
                action = self.action_inferrer.infer_action(frame_resized, prev_frame)
                chunk_proxy_actions.append(action)

            if audio_pcm is not None and samples_per_frame is not None:
                # Align audio to extracted frame index.
                start = extracted * samples_per_frame
                end = start + samples_per_frame
                if start < len(audio_pcm):
                    seg = audio_pcm[start:end]
                    if len(seg) < samples_per_frame:
                        pad = samples_per_frame - len(seg)
                        seg = np.pad(seg, (0, pad), mode="constant")
                else:
                    seg = np.zeros((samples_per_frame,), dtype=np.int16)
                chunk_audio.append(seg.astype(np.int16, copy=False))

            chunk_frames.append(frame_resized)
            
            prev_frame = frame_resized.copy()
            extracted += 1

            # Flush chunk to disk
            if len(chunk_frames) >= int(self.cfg.frames_per_chunk):
                if chunk_proxy_actions:
                    chunk_proxy_actions = self.action_inferrer.smooth_actions(
                        chunk_proxy_actions, self.cfg.action_smoothing
                    )

                output_path = (
                    Path(self.cfg.output_dir)
                    / f"demo_{video_path.stem}_{trajectory_id}_chunk{chunk_idx:04d}.npz"
                )

                dones = np.zeros(len(chunk_frames), dtype=bool)
                if len(dones) > 0:
                    dones[-1] = True

                np.savez_compressed(
                    output_path,
                    observations=np.array(chunk_frames),
                    dones=dones,
                    video_source=str(video_path),
                    meta=json.dumps(
                        {
                            "episode_id": f"{video_path.stem}_{trajectory_id}",
                            "chunk_index": chunk_idx,
                            "action_labeling": "optical_flow_proxy" if chunk_proxy_actions else "unlabeled",
                            "reward_labeling": "unlabeled",
                            "audio_sample_rate": int(self.cfg.audio_sample_rate) if audio_pcm is not None else 0,
                            "audio_samples_per_frame": int(samples_per_frame or 0),
                            "audio_channels": int(self.cfg.audio_channels) if audio_pcm is not None else 0,
                        }
                    ),
                    **({"proxy_actions": np.array(chunk_proxy_actions)} if chunk_proxy_actions else {}),
                    **({"audio": np.array(chunk_audio)} if chunk_audio else {}),
                )

                outputs.append(str(output_path))
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"\n  Saved chunk {chunk_idx}: {output_path.name} ({file_size_mb:.1f} MB, {len(chunk_frames)} frames)")

                chunk_idx += 1
                chunk_frames = []
                chunk_proxy_actions = []
                chunk_audio = []
            
            # Progress update - in-place with carriage return
            now = time.time()
            if extracted % self.cfg.progress_interval == 0 or now - last_progress_time > 10:
                elapsed = now - start_time_proc
                fps_rate = extracted / elapsed if elapsed > 0 else 0
                remaining = (target_extract - extracted) / fps_rate if fps_rate > 0 else 0
                progress = (extracted / target_extract * 100) if target_extract > 0 else 0
                
                # Format ETA as HH:MM:SS.sss
                eta_h = int(remaining // 3600)
                eta_m = int((remaining % 3600) // 60)
                eta_s = remaining % 60
                eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:06.3f}"
                
                # In-place progress bar
                bar_width = 30
                filled = int(bar_width * progress / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                print(f"\r  {video_path.name} [{bar}] {progress:5.1f}% | {extracted:,}/{target_extract:,} | {fps_rate:.0f}fps | ETA {eta_str}", end="", flush=True)
                last_progress_time = now
        
        cap.release()
        
        # Final timing report (newline after in-place progress)
        total_time = time.time() - start_time_proc
        tt_h = int(total_time // 3600)
        tt_m = int((total_time % 3600) // 60)
        tt_s = total_time % 60
        time_str = f"{tt_h:02d}:{tt_m:02d}:{tt_s:06.3f}"
        print(f"\n  ✓ {video_path.name}: {extracted:,} frames in {time_str} ({extracted/total_time:.0f} fps)")

        
        if extracted == 0:
            return {"error": "No frames extracted"}
        
        # Flush final partial chunk.
        if chunk_frames:
            if chunk_proxy_actions:
                chunk_proxy_actions = self.action_inferrer.smooth_actions(
                    chunk_proxy_actions, self.cfg.action_smoothing
                )

            output_path = (
                Path(self.cfg.output_dir)
                / f"demo_{video_path.stem}_{trajectory_id}_chunk{chunk_idx:04d}.npz"
            )
            dones = np.zeros(len(chunk_frames), dtype=bool)
            if len(dones) > 0:
                dones[-1] = True

            # IMPORTANT: Do not write placeholder rewards or pretend proxy actions are real.
            # Rewards/actions are filled by learned labeling in `scripts/video_pretrain.py`.
            np.savez_compressed(
                output_path,
                observations=np.array(chunk_frames),
                dones=dones,
                video_source=str(video_path),
                meta=json.dumps(
                    {
                        "episode_id": f"{video_path.stem}_{trajectory_id}",
                        "chunk_index": chunk_idx,
                        "action_labeling": "optical_flow_proxy" if chunk_proxy_actions else "unlabeled",
                        "reward_labeling": "unlabeled",
                        "audio_sample_rate": int(self.cfg.audio_sample_rate) if audio_pcm is not None else 0,
                        "audio_samples_per_frame": int(samples_per_frame or 0),
                        "audio_channels": int(self.cfg.audio_channels) if audio_pcm is not None else 0,
                    }
                ),
                **({"proxy_actions": np.array(chunk_proxy_actions)} if chunk_proxy_actions else {}),
                **({"audio": np.array(chunk_audio)} if chunk_audio else {}),
            )

            outputs.append(str(output_path))
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  Saved chunk {chunk_idx}: {output_path.name} ({file_size_mb:.1f} MB, {len(chunk_frames)} frames)")
            chunk_idx += 1

        return {
            "video": str(video_path),
            "frames_extracted": extracted,
            "chunks": chunk_idx,
            "outputs": outputs,
        }
    
    def process_directory(
        self,
        video_dir: str,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".webm"),
    ) -> List[Dict[str, Any]]:
        """Process all videos in a directory."""
        video_dir = Path(video_dir)
        
        videos = []
        for ext in extensions:
            videos.extend(video_dir.glob(f"*{ext}"))
            # Also check .part files (partial downloads that might be playable)
            videos.extend(video_dir.glob(f"*{ext}.part"))
        
        # Human-readable phase introduction
        print("\n" + "="*60)
        print("📹 VIDEO TO TRAJECTORY EXTRACTION")
        print("="*60)
        print(f"Found {len(videos)} videos in {video_dir}")
        print()
        print("What's happening:")
        print("  • Extracting frames at {:.0f} FPS (from original video framerate)".format(self.cfg.target_fps))
        print("  • Inferring movement actions via optical flow analysis")
        print("  • Saving compressed .npz training datasets")
        print()
        print("Why this matters:")
        print("  → These frames become the AI's visual 'memories' of gameplay")
        print("  → Movement patterns teach the AI how humans navigate the game")
        print("="*60)
        
        phase_start = time.time()
        results = []
        video_times = []
        
        for i, video in enumerate(sorted(videos), 1):
            video_start = time.time()
            print(f"\n[{i}/{len(videos)}] {video.name}")
            
            result = self.process_video(str(video))
            
            video_elapsed = time.time() - video_start
            result["processing_time_s"] = video_elapsed
            video_times.append((video.name, video_elapsed))
            results.append(result)
        
        phase_elapsed = time.time() - phase_start
        
        # Summary
        total_frames = sum(r.get("frames_extracted", 0) for r in results)
        total_size = sum(r.get("file_size_mb", 0) for r in results)
        
        print(f"\n{'='*60}")
        print(f"[VideoToTrajectory] Phase Complete")
        print(f"{'='*60}")
        print(f"  Videos processed: {len(results)}")
        print(f"  Total frames: {total_frames:,}")
        print(f"  Total output size: {total_size:.1f} MB")
        print(f"  Total time: {phase_elapsed:.1f}s ({phase_elapsed/60:.1f} min)")
        print(f"\n  Per-video timing:")
        for name, t in video_times:
            print(f"    {name}: {t:.1f}s")
        print(f"{'='*60}")
        
        return results

    
    def create_training_manifest(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """Create manifest of all extracted trajectories."""
        rollout_dir = Path(self.cfg.output_dir)
        trajectories = list(rollout_dir.glob("demo_*.npz"))
        
        manifest = {
            "version": "1.0",
            "trajectories": [],
            "total_frames": 0,
            "config": {
                "target_fps": self.cfg.target_fps,
                "resize": self.cfg.resize,
                "audio": {
                    "enabled": bool(self.cfg.extract_audio),
                    "sample_rate": int(self.cfg.audio_sample_rate),
                    "channels": int(self.cfg.audio_channels),
                },
            },
        }
        
        for traj_path in trajectories:
            data = np.load(traj_path, allow_pickle=True)
            num_frames = len(data.get("observations", []))
            
            manifest["trajectories"].append({
                "path": str(traj_path),
                "frames": num_frames,
                "video_source": str(data.get("video_source", "unknown")),
            })
            manifest["total_frames"] += num_frames
        
        output_path = output_path or str(rollout_dir / "manifest.json")
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[VideoToTrajectory] Created manifest: {output_path}")
        print(f"  Trajectories: {len(manifest['trajectories'])}")
        print(f"  Total frames: {manifest['total_frames']}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Video to Trajectory Pipeline")
    parser.add_argument("--video-dir", type=str, default="gameplay_videos",
                        help="Directory containing gameplay videos")
    parser.add_argument("--video", type=str, help="Single video to process")
    parser.add_argument("--output-dir", type=str, default="rollouts/video_demos",
                        help="Output directory for trajectories")
    parser.add_argument("--fps", type=float, default=45.0,
                        help="Target FPS for extraction (default: 45 for action games)")
    parser.add_argument("--resize", type=int, nargs=2, default=[224, 224],
                        help="Target frame size (width height)")
    parser.add_argument("--start-s", type=float, default=0.0, help="Start time (seconds) for single-video mode")
    parser.add_argument("--end-s", type=float, default=-1.0, help="End time (seconds) for single-video mode")
    parser.add_argument("--frames-per-chunk", type=int, default=1000, help="Frames per output .npz chunk")
    parser.add_argument("--audio", action="store_true", default=True, help="Extract audio chunks alongside frames")
    parser.add_argument("--no-audio", action="store_false", dest="audio")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Audio sample rate for extraction")
    parser.add_argument("--audio-channels", type=int, default=1, help="Audio channels (1=mono)")
    
    args = parser.parse_args()
    
    cfg = VideoConfig(
        target_fps=args.fps,
        resize=tuple(args.resize),
        output_dir=args.output_dir,
        frames_per_chunk=int(args.frames_per_chunk),
        extract_audio=bool(args.audio),
        audio_sample_rate=int(args.audio_sample_rate),
        audio_channels=int(args.audio_channels),
    )
    
    pipeline = VideoToTrajectory(cfg)
    
    if args.video:
        # Process single video
        result = pipeline.process_video(args.video, start_time=float(args.start_s), end_time=float(args.end_s))
        print(f"\nResult: {result}")
    else:
        # Process directory
        results = pipeline.process_directory(args.video_dir)
        
        # Create manifest
        pipeline.create_training_manifest()
    
    print("\n[VideoToTrajectory] Done!")
    print(f"Trajectories saved to: {cfg.output_dir}")
    print(f"\nTo train diffusion policy:")
    print(f"  python scripts/train_sima2.py --phase 1")


if __name__ == "__main__":
    main()
