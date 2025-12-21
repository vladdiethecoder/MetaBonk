"""Video-based pretraining utilities (VPT-style, offline).

This module wires up the missing pieces for "video-only" pretraining:
  - learned inverse dynamics (IDM) action labeling
  - learned reward-from-video (self-supervised temporal ranking)
  - conversion to vector rollouts for world-model + dreaming
  - skill discovery via SkillVQVAE trained on labeled action sequences

All components are intentionally *game-agnostic*:
  - No hard-coded key bindings, aiming heuristics, or game-specific rewards.
  - Rewards are learned from temporal ordering (or other supervision provided
    by the caller), not from hand-authored gameplay rules.
"""

from __future__ import annotations

import time
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def iter_npz_files(root: str | Path) -> Iterator[Path]:
    root_p = Path(root)
    if not root_p.exists():
        return iter(())
    yield from sorted(root_p.glob("*.npz"))


def list_npz_files(root: str | Path, *, sort: bool = False) -> List[Path]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    # `os.scandir` is substantially faster than `glob` for large directories.
    paths: List[Path] = []
    try:
        with os.scandir(root_p) as it:
            for ent in it:
                if ent.is_file() and ent.name.endswith(".npz"):
                    paths.append(Path(ent.path))
    except FileNotFoundError:
        return []
    if sort:
        paths.sort()
    return paths


def _safe_np_load(path: Path):
    # Prefer safe non-pickle loads; fall back for legacy/object arrays.
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)


def _load_npz_selected(path: Path, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    data = _safe_np_load(path)
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        if k in data.files:
            out[k] = data[k]
    return out


def _load_npz_all(path: Path) -> Dict[str, np.ndarray]:
    data = _safe_np_load(path)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = data[k]
    return out


def _ensure_uint8_frames(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    # Handle float frames in [0,1] or [0,255].
    f = frames
    if np.issubdtype(f.dtype, np.floating):
        mx = float(np.nanmax(f)) if f.size else 0.0
        if mx <= 1.5:
            f = f * 255.0
    f = np.clip(f, 0.0, 255.0).astype(np.uint8)
    return f


def _to_torch_frames(
    frames_hwc: np.ndarray,
    *,
    device: "torch.device",
    out_hw: Optional[Tuple[int, int]] = None,
    channels_last: bool = False,
) -> "torch.Tensor":
    # [B,H,W,C] uint8 -> [B,C,H,W] float in [0,1]
    f = torch.from_numpy(frames_hwc).to(device=device)
    if f.dtype != torch.uint8:
        f = f.to(dtype=torch.uint8)
    f = f.permute(0, 3, 1, 2).contiguous()
    f = f.to(dtype=torch.float32) / 255.0
    if out_hw is not None:
        oh, ow = int(out_hw[0]), int(out_hw[1])
        if int(f.shape[-2]) != oh or int(f.shape[-1]) != ow:
            # Interpolate in float space; keeps training robust when data was extracted at a different size.
            f = F.interpolate(f, size=(oh, ow), mode="bilinear", align_corners=False)
    if channels_last:
        f = f.contiguous(memory_format=torch.channels_last)
    return f


def _as_button_targets(action: "torch.Tensor") -> "torch.Tensor":
    # Accept either {0,1} or [-1,1] style labels.
    btn = action
    if btn.numel() == 0:
        return btn
    if torch.min(btn) < 0.0:
        return (btn > 0.0).to(dtype=torch.float32)
    return torch.clamp(btn, 0.0, 1.0)


@dataclass(frozen=True)
class IDMTrainConfig:
    npz_dir: str = "rollouts/video_demos"
    out_ckpt: str = "checkpoints/idm.pt"
    device: str = "cuda"
    frame_size: Tuple[int, int] = (224, 224)
    action_dim: int = 6
    context: int = 3
    batch_size: int = 32
    steps: int = 2000
    # If set (>0), skip NPZ files larger than this many bytes (helps avoid pathological huge single-trajectory files).
    max_npz_bytes: int = 0
    # Data perf: amortize expensive NPZ decompression by sampling multiple steps per loaded file.
    steps_per_load: int = 50
    prefetch: bool = True
    # Compute perf.
    tf32: bool = True
    channels_last: bool = True
    compile: bool = False
    # If enabled (CUDA only), cache the currently loaded episode on GPU and sample batches via indexing.
    # This can dramatically increase utilization by removing per-step CPU->GPU transfers.
    cache_on_device: bool = False
    # Store cached frames in fp16 (default) to reduce GPU memory + bandwidth.
    cache_fp16: bool = True
    # Safety limit: if >0, skip caching when estimated bytes exceed this.
    cache_max_bytes: int = 0
    lr: float = 3e-4
    seed: int = 0


@dataclass(frozen=True)
class IDMLabelConfig:
    in_npz_dir: str = "rollouts/video_demos"
    out_npz_dir: str = "rollouts/video_demos_labeled"
    idm_ckpt: str = "checkpoints/idm.pt"
    device: str = "cuda"
    context: int = 3
    batch_size: int = 64
    prefetch: bool = True
    compress_output: bool = True
    cache_on_device: bool = False
    cache_fp16: bool = True
    cache_max_bytes: int = 0
    # Optional progress peeking
    peek_every: int = 0
    peek_samples: int = 5


@dataclass(frozen=True)
class RewardTrainConfig:
    npz_dir: str = "rollouts/video_demos"
    out_ckpt: str = "checkpoints/video_reward_model.pt"
    device: str = "cuda"
    frame_size: Tuple[int, int] = (224, 224)
    embed_dim: int = 256
    batch_size: int = 64
    steps: int = 5000
    max_npz_bytes: int = 0
    steps_per_load: int = 50
    prefetch: bool = True
    tf32: bool = True
    channels_last: bool = True
    compile: bool = False
    cache_on_device: bool = False
    cache_fp16: bool = True
    cache_max_bytes: int = 0
    lr: float = 3e-4
    seed: int = 0


@dataclass(frozen=True)
class RewardLabelConfig:
    in_npz_dir: str = "rollouts/video_demos_labeled"
    out_npz_dir: str = "rollouts/video_demos_labeled"
    reward_ckpt: str = "checkpoints/video_reward_model.pt"
    device: str = "cuda"
    batch_size: int = 128
    # Reward is delta(progress_score) by default.
    reward_scale: float = 1.0
    prefetch: bool = True
    compress_output: bool = True
    cache_on_device: bool = False
    cache_fp16: bool = True
    cache_max_bytes: int = 0
    peek_every: int = 0
    peek_samples: int = 5


@dataclass(frozen=True)
class ExportRolloutsConfig:
    in_npz_dir: str = "rollouts/video_demos_labeled"
    out_pt_dir: str = "rollouts/video_rollouts"
    reward_ckpt: str = "checkpoints/video_reward_model.pt"
    # Optional: audio token embedding to append to obs vectors.
    audio_token_ckpt: str = ""
    audio_embed_scale: float = 1.0
    audio_optional: bool = True
    device: str = "cuda"
    batch_size: int = 128
    prefetch: bool = True
    cache_on_device: bool = False
    cache_fp16: bool = True
    cache_max_bytes: int = 0


@dataclass(frozen=True)
class SkillLabelConfig:
    in_npz_dir: str = "rollouts/video_demos_labeled"
    out_npz_dir: str = "rollouts/video_demos_labeled"
    skill_ckpt: str = "checkpoints/skill_vqvae.pt"
    device: str = "cuda"
    seq_len: int = 16
    stride: int = 8
    batch_size: int = 256
    compress_output: bool = True
    peek_every: int = 0
    peek_samples: int = 5


@dataclass(frozen=True)
class AudioTokenTrainConfig:
    npz_dir: str = "rollouts/video_demos_labeled"
    out_ckpt: str = "checkpoints/audio_vqvae.pt"
    device: str = "cuda"
    sample_rate: int = 16000
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    n_mels: int = 64
    context_frames: int = 1
    num_codes: int = 512
    code_dim: int = 64
    encoder_hidden: int = 256
    commitment_cost: float = 0.25
    mel_eps: float = 1e-5
    batch_size: int = 128
    steps: int = 4000
    max_npz_bytes: int = 0
    steps_per_load: int = 50
    prefetch: bool = True
    tf32: bool = True
    compile: bool = False
    seed: int = 0


@dataclass(frozen=True)
class AudioLabelConfig:
    in_npz_dir: str = "rollouts/video_demos_labeled"
    out_npz_dir: str = "rollouts/video_demos_labeled"
    audio_ckpt: str = "checkpoints/audio_vqvae.pt"
    device: str = "cuda"
    batch_size: int = 256
    context_frames: int = 1
    stride: int = 1
    compress_output: bool = True
    peek_every: int = 0
    peek_samples: int = 5


@dataclass(frozen=True)
class InspectLabelsConfig:
    npz_dir: str = "rollouts/video_demos_labeled"
    num_files: int = 3
    samples_per_file: int = 5
    seed: int = 0
    topk: int = 10
    json_out: str = ""


@dataclass(frozen=True)
class ShardNPZConfig:
    """Split large trajectory NPZ files into smaller chunk NPZ files.

    This is a pragmatic performance fix for datasets saved as single huge `.npz`
    trajectories, where per-step training repeatedly decompresses massive arrays.
    """

    in_npz_dir: str = "rollouts/video_demos"
    out_npz_dir: str = "rollouts/video_demos_sharded"
    frames_per_chunk: int = 1000
    compress: bool = True
    workers: int = 1
    delete_source: bool = False
    overwrite: bool = False


def _shard_one_npz_file(
    *,
    path: Path,
    out_dir: Path,
    frames_per_chunk: int,
    compress: bool,
    delete_source: bool,
    overwrite: bool,
) -> None:
    if "_chunk" in path.stem:
        return

    import zipfile
    from numpy.lib import format as npyfmt

    def _read_exact(fp, n: int) -> bytes:
        buf = bytearray(n)
        mv = memoryview(buf)
        off = 0
        while off < n:
            got = fp.read(n - off)
            if not got:
                raise EOFError("unexpected EOF while reading npy data")
            mv[off : off + len(got)] = got
            off += len(got)
        return bytes(buf)

    fps = max(1, int(frames_per_chunk))

    # Load all non-observation arrays (typically small) without ever touching `observations`.
    other: Dict[str, np.ndarray] = {}
    with _safe_np_load(path) as npz:
        if "observations" not in npz.files:
            return
        for k in npz.files:
            if k == "observations":
                continue
            try:
                other[k] = np.asarray(npz[k])
            except Exception:
                continue

    with zipfile.ZipFile(path, "r") as zf:
        obs_member = "observations.npy"
        if obs_member not in zf.namelist():
            return

        with zf.open(obs_member, "r") as f:
            version = npyfmt.read_magic(f)
            if version == (1, 0):
                shape, fortran_order, dtype = npyfmt.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran_order, dtype = npyfmt.read_array_header_2_0(f)
            else:
                # Unknown/unsupported .npy version.
                return

            if not isinstance(shape, tuple) or len(shape) != 4:
                return
            T, H, W, C = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
            if T <= 0 or C != 3:
                return

            num_chunks = (T + fps - 1) // fps
            if not overwrite:
                # If already fully sharded, skip without decompressing.
                if all((out_dir / f"{path.stem}_chunk{ci:04d}.npz").exists() for ci in range(num_chunks)):
                    return

            print(f"[video_pretraining] Shard streaming -> {path} (T={T}, chunk={fps}, chunks={num_chunks})")

            for chunk_idx in range(num_chunks):
                start = chunk_idx * fps
                end = min(T, (chunk_idx + 1) * fps)
                n = end - start
                out_path = out_dir / f"{path.stem}_chunk{chunk_idx:04d}.npz"

                nbytes = n * H * W * C * int(dtype.itemsize)
                raw = _read_exact(f, nbytes)
                obs_chunk = np.frombuffer(raw, dtype=dtype).reshape((n, H, W, C), order="F" if fortran_order else "C")
                if fortran_order:
                    obs_chunk = np.ascontiguousarray(obs_chunk)

                if out_path.exists() and not overwrite:
                    continue

                chunk: Dict[str, np.ndarray] = {"observations": obs_chunk}
                for k, arr in other.items():
                    if arr.ndim >= 1 and int(arr.shape[0]) == T:
                        chunk[k] = arr[start:end]
                    else:
                        chunk[k] = arr

                # Ensure chunk-local dones mark the end of each chunk for downstream tooling.
                if "dones" in chunk:
                    d = np.asarray(chunk["dones"])
                    if d.ndim == 1 and d.shape[0] == n and d.dtype != np.bool_:
                        d = d.astype(np.bool_)
                    if d.ndim == 1 and d.shape[0] == n:
                        d = d.copy()
                        d[-1] = True
                        chunk["dones"] = d
                elif n > 0:
                    d = np.zeros((n,), dtype=np.bool_)
                    d[-1] = True
                    chunk["dones"] = d

                if compress:
                    np.savez_compressed(out_path, **chunk)
                else:
                    np.savez(out_path, **chunk)
                print(f"[video_pretraining] Sharded -> {out_path}")

    if delete_source and not overwrite:
        try:
            if all((out_dir / f"{path.stem}_chunk{ci:04d}.npz").exists() for ci in range(num_chunks)):
                path.unlink()
                print(f"[video_pretraining] Deleted source -> {path}")
        except Exception:
            pass


def shard_npz_demos(cfg: ShardNPZConfig) -> Path:
    in_dir = Path(cfg.in_npz_dir)
    out_dir = Path(cfg.out_npz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_npz_files(in_dir))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    workers = max(1, int(getattr(cfg, "workers", 1)))
    if workers == 1:
        for path in files:
            _shard_one_npz_file(
                path=path,
                out_dir=out_dir,
                frames_per_chunk=int(cfg.frames_per_chunk),
                compress=bool(cfg.compress),
                delete_source=bool(cfg.delete_source),
                overwrite=bool(cfg.overwrite),
            )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Note: sharding can be RAM heavy; use workers>1 only if you have ample system RAM.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _shard_one_npz_file,
                    path=path,
                    out_dir=out_dir,
                    frames_per_chunk=int(cfg.frames_per_chunk),
                    compress=bool(cfg.compress),
                    delete_source=bool(cfg.delete_source),
                    overwrite=bool(cfg.overwrite),
                )
                for path in files
            ]
            for f in as_completed(futs):
                f.result()

    return out_dir


if TORCH_AVAILABLE:

    class TemporalRankRewardModel(nn.Module):
        """Self-supervised reward model from temporal ordering.

        Trains a scalar "progress score" such that later frames should score
        higher than earlier frames within the same trajectory:
            score(t2) > score(t1) for t2 > t1

        Rewards can then be derived as score deltas.
        """

        def __init__(self, frame_size: Tuple[int, int] = (224, 224), embed_dim: int = 256):
            super().__init__()
            self.frame_size = tuple(int(x) for x in frame_size)
            self.embed_dim = int(embed_dim)

            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4, padding=2),  # 224 -> 56
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 56 -> 28
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 28 -> 14
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 14 -> 7
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.embed = nn.Linear(256, self.embed_dim)
            self.head = nn.Linear(self.embed_dim, 1)

        def forward(self, x: "torch.Tensor", *, return_embed: bool = False):
            # x: [B,C,H,W] float in [0,1]
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            z = self.embed(h)
            score = self.head(torch.tanh(z)).squeeze(-1)
            if return_embed:
                return score, z
            return score


def _device_from_str(device: str) -> "torch.device":
    assert TORCH_AVAILABLE
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_idm_batch(
    frames: np.ndarray,
    actions: np.ndarray,
    *,
    context: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(frames.shape[0])
    lo = context
    hi = T - context
    if hi <= lo:
        raise ValueError("trajectory too short for IDM context window")
    idx = rng.integers(lo, hi, size=(batch_size,))
    offsets = np.arange(-context, context + 1, dtype=np.int64)
    win_idx = idx[:, None] + offsets[None, :]
    # Fancy indexing builds [B, 2*context+1, H, W, C] without a Python loop.
    win = frames[win_idx]
    act = actions[idx].astype(np.float32)
    return win, act


def _configure_torch_perf(*, device: "torch.device", tf32: bool) -> None:
    if not TORCH_AVAILABLE:
        return
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if tf32:
            # New TF32 API (torch>=2.9); fall back to legacy.
            try:
                torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _maybe_compile(model: "torch.nn.Module", enabled: bool) -> "torch.nn.Module":
    if not TORCH_AVAILABLE or not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    try:
        return compile_fn(model)  # type: ignore[misc]
    except Exception:
        return model


def _supports_fused_adamw() -> bool:
    if not TORCH_AVAILABLE:
        return False
    try:
        import inspect

        return "fused" in inspect.signature(torch.optim.AdamW).parameters
    except Exception:
        return False


def _submit_prefetch(
    executor: Optional[ThreadPoolExecutor],
    fn,
    *args,
    **kwargs,
) -> Optional[Future]:
    if executor is None:
        return None
    return executor.submit(fn, *args, **kwargs)


def _estimate_frame_cache_bytes(*, T: int, H: int, W: int, fp16: bool) -> int:
    # Cached as [T,3,H,W] float.
    bytes_per = 2 if fp16 else 4
    return int(T) * 3 * int(H) * int(W) * bytes_per


def _cache_episode_on_device(
    *,
    frames_u8: np.ndarray,
    actions_f32: Optional[np.ndarray],
    device: "torch.device",
    frame_size: Tuple[int, int],
    channels_last: bool,
    fp16: bool,
    max_bytes: int,
) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    if not TORCH_AVAILABLE or device.type != "cuda":
        return None, None
    if frames_u8.ndim != 4 or frames_u8.shape[-1] != 3:
        return None, None

    T = int(frames_u8.shape[0])
    out_h, out_w = int(frame_size[0]), int(frame_size[1])
    est = _estimate_frame_cache_bytes(T=T, H=out_h, W=out_w, fp16=bool(fp16))

    # If the user provided an explicit cap, honor it.
    if int(max_bytes) > 0 and est > int(max_bytes):
        return None, None

    # Otherwise, be conservative and only cache if it fits comfortably into free memory.
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        if est > int(free_b * 0.60):
            return None, None
    except Exception:
        pass

    # Move once, then sample via GPU indexing.
    f = torch.from_numpy(frames_u8)
    # [T,H,W,C] -> [T,3,H,W] float in [0,1]
    f = f.to(device=device, dtype=torch.uint8)
    f = f.permute(0, 3, 1, 2).contiguous()
    f = f.to(dtype=torch.float16 if fp16 else torch.float32) / 255.0
    if int(f.shape[-2]) != out_h or int(f.shape[-1]) != out_w:
        f = F.interpolate(f, size=(out_h, out_w), mode="bilinear", align_corners=False)
    if channels_last:
        f = f.contiguous(memory_format=torch.channels_last)

    a_t: Optional["torch.Tensor"] = None
    if actions_f32 is not None and actions_f32.size:
        a_t = torch.from_numpy(actions_f32).to(device=device, dtype=torch.float32)

    return f, a_t


def _summarize_actions(actions: np.ndarray) -> Dict[str, float]:
    if actions is None:
        return {"T": 0.0}
    a = np.asarray(actions)
    if a.ndim != 2 or a.size == 0:
        return {"T": float(a.shape[0]) if a.ndim else 0.0}
    a = a.astype(np.float32, copy=False)
    T, D = int(a.shape[0]), int(a.shape[1])
    out: Dict[str, float] = {"T": float(T), "D": float(D)}
    out["a_min"] = float(np.min(a))
    out["a_max"] = float(np.max(a))
    if D >= 2:
        move = a[:, :2]
        mag = np.linalg.norm(move, axis=-1)
        out["move_mag_mean"] = float(np.mean(mag))
        out["move_mag_p95"] = float(np.quantile(mag, 0.95))
        out["move_near0_frac"] = float(np.mean(mag < 0.05))
    if D > 2:
        btn = a[:, 2:]
        out["btn_mean"] = float(np.mean(btn))
        out["btn_on_frac_0.5"] = float(np.mean(btn > 0.5))
        out["btn_on_frac_0.1"] = float(np.mean(btn > 0.1))
    return out


def _summarize_rewards(rewards: np.ndarray) -> Dict[str, float]:
    if rewards is None:
        return {"T": 0.0}
    r = np.asarray(rewards).reshape(-1)
    if r.size == 0:
        return {"T": 0.0}
    r = r.astype(np.float32, copy=False)
    return {
        "T": float(r.shape[0]),
        "r_mean": float(np.mean(r)),
        "r_std": float(np.std(r)),
        "r_min": float(np.min(r)),
        "r_max": float(np.max(r)),
        "r_p05": float(np.quantile(r, 0.05)),
        "r_p95": float(np.quantile(r, 0.95)),
        "r_zero_frac": float(np.mean(np.isclose(r, 0.0))),
    }


def _summarize_tokens(tokens: np.ndarray, *, topk: int = 10) -> Dict[str, Any]:
    if tokens is None:
        return {"T": 0, "unique": 0, "unlabeled_frac": 1.0, "top": []}
    t = np.asarray(tokens).reshape(-1)
    if t.size == 0:
        return {"T": 0, "unique": 0, "unlabeled_frac": 1.0, "top": []}
    t = t.astype(np.int64, copy=False)
    unlabeled = t < 0
    labeled = t[~unlabeled]
    out: Dict[str, Any] = {"T": int(t.shape[0]), "unlabeled_frac": float(np.mean(unlabeled))}
    if labeled.size == 0:
        out["unique"] = 0
        out["top"] = []
        return out
    vals, counts = np.unique(labeled, return_counts=True)
    order = np.argsort(-counts)
    k = max(0, int(topk))
    out["unique"] = int(vals.shape[0])
    out["top"] = [(int(vals[i]), int(counts[i])) for i in order[:k]]
    return out


def train_idm(cfg: IDMTrainConfig) -> Path:
    """Train an inverse-dynamics model on (frames -> action) supervision.

    This is a generic supervised trainer; for video-only datasets without real
    inputs, you must provide some action labels (e.g., from a small instrumented
    dataset) before this can learn meaningful button/aim semantics.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to train IDM")

    from src.imitation import InverseDynamicsModel, VPTConfig

    npz_paths = list_npz_files(cfg.npz_dir, sort=False)
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")
    max_npz_bytes = int(getattr(cfg, "max_npz_bytes", 0) or 0)
    if max_npz_bytes > 0:
        filtered = [p for p in npz_paths if p.stat().st_size <= max_npz_bytes]
        if filtered:
            npz_paths = filtered

    device = _device_from_str(cfg.device)
    rng = np.random.default_rng(int(cfg.seed))

    _configure_torch_perf(device=device, tf32=bool(cfg.tf32))

    vpt_cfg = VPTConfig(frame_size=cfg.frame_size, action_dim=cfg.action_dim, idm_context=cfg.context)
    model = InverseDynamicsModel(vpt_cfg).to(device)
    if bool(cfg.channels_last) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile(model, enabled=bool(cfg.compile) and device.type == "cuda")
    
    # Fused optimizer for GPU efficiency
    use_fused = device.type == "cuda" and _supports_fused_adamw()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), fused=use_fused)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
        use_fused = False
    
    # AMP for mixed precision training
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    steps = int(cfg.steps)
    bs = int(cfg.batch_size)
    ctx = int(cfg.context)
    steps_per_load = max(1, int(getattr(cfg, "steps_per_load", 1)))
    
    start_time = time.time()
    log_interval = max(100, steps // 20)  # ~20 logs per run
    
    print(
        f"[train_idm] Starting: {steps} steps, batch_size={bs}, AMP={use_amp}, fused={use_fused}, "
        f"steps_per_load={steps_per_load}, prefetch={bool(cfg.prefetch)}"
    )

    def _load_idm_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = _load_npz_selected(path, ("observations", "actions", "proxy_actions"))
        frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
        actions = np.asarray(data.get("actions"))
        if actions.size == 0:
            actions = np.asarray(data.get("proxy_actions"))
        if actions.size == 0:
            raise ValueError(f"{path.name} has no actions/proxy_actions; cannot train IDM")
        return frames, actions

    cache_on_device = bool(getattr(cfg, "cache_on_device", False)) and device.type == "cuda"
    cache_fp16 = bool(getattr(cfg, "cache_fp16", True))
    cache_max_bytes = int(getattr(cfg, "cache_max_bytes", 0) or 0)
    cache_requested = cache_on_device

    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if bool(cfg.prefetch) else None
    try:
        # Load the first episode (and prefetch the second) before entering the hot loop.
        cur_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
        next_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
        next_future = _submit_prefetch(executor, _load_idm_npz, next_path)
        frames, actions = _load_idm_npz(cur_path)
        frames_cache, actions_cache = _cache_episode_on_device(
            frames_u8=frames,
            actions_f32=np.asarray(actions, dtype=np.float32),
            device=device,
            frame_size=tuple(int(x) for x in cfg.frame_size),
            channels_last=bool(cfg.channels_last) and device.type == "cuda",
            fp16=cache_fp16,
            max_bytes=cache_max_bytes,
        )
        if cache_requested:
            if frames_cache is None:
                print("[train_idm] GPU cache: skipped (episode too large / low free VRAM / non-CUDA)")
            else:
                print(
                    f"[train_idm] GPU cache: enabled (frames={tuple(frames_cache.shape)}, dtype={frames_cache.dtype})"
                )
        if cache_on_device and frames_cache is None:
            cache_on_device = False

        # Sample many minibatches from the same loaded episode to amortize NPZ decompression.
        for step in range(steps):
            if step > 0 and (step % steps_per_load) == 0:
                if next_future is not None:
                    try:
                        frames, actions = next_future.result()
                    except Exception:
                        # Fallback to synchronous load if prefetch failed.
                        frames, actions = _load_idm_npz(next_path)
                else:
                    frames, actions = _load_idm_npz(next_path)
                cur_path = next_path
                next_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
                next_future = _submit_prefetch(executor, _load_idm_npz, next_path)
                frames_cache, actions_cache = _cache_episode_on_device(
                    frames_u8=frames,
                    actions_f32=np.asarray(actions, dtype=np.float32),
                    device=device,
                    frame_size=tuple(int(x) for x in cfg.frame_size),
                    channels_last=bool(cfg.channels_last) and device.type == "cuda",
                    fp16=cache_fp16,
                    max_bytes=cache_max_bytes,
                )
                if bool(getattr(cfg, "cache_on_device", False)) and frames_cache is not None:
                    cache_on_device = True
                else:
                    cache_on_device = False

            if cache_on_device and frames_cache is not None and actions_cache is not None:
                T = int(frames_cache.shape[0])
                lo = ctx
                hi = T - ctx
                if hi <= lo:
                    continue
                idx = torch.randint(lo, hi, (bs,), device=device)
                offsets = torch.arange(-ctx, ctx + 1, device=device, dtype=torch.int64)
                win_idx = idx[:, None] + offsets[None, :]
                flat_idx = win_idx.reshape(-1)
                win_flat_t = frames_cache.index_select(0, flat_idx)
                if bool(cfg.channels_last) and device.type == "cuda":
                    win_flat_t = win_flat_t.contiguous(memory_format=torch.channels_last)
                win_t = win_flat_t.view(bs, 2 * ctx + 1, 3, win_flat_t.shape[-2], win_flat_t.shape[-1])
                act_t = actions_cache.index_select(0, idx).to(dtype=torch.float32)
            else:
                win, act = _sample_idm_batch(frames, actions, context=ctx, batch_size=bs, rng=rng)
                win_flat = win.reshape(bs * (2 * ctx + 1), *win.shape[2:])
                win_t = _to_torch_frames(
                    win_flat,
                    device=device,
                    out_hw=tuple(int(x) for x in cfg.frame_size),
                    channels_last=bool(cfg.channels_last) and device.type == "cuda",
                )
                win_t = win_t.view(bs, 2 * ctx + 1, 3, win_t.shape[-2], win_t.shape[-1])
                act_t = torch.from_numpy(act).to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
        
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(win_t)
                move_pred = out["movement"]
                btn_logits = out["buttons"]

                move_tgt = act_t[:, :2]
                btn_tgt = _as_button_targets(act_t[:, 2:])

                loss_move = F.mse_loss(move_pred, move_tgt)
                loss_btn = torch.tensor(0.0, device=device)
                if btn_logits.numel() and btn_tgt.numel():
                    loss_btn = F.binary_cross_entropy_with_logits(btn_logits, btn_tgt)

                loss = loss_move + loss_btn

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed
                eta = (steps - step - 1) / rate if rate > 0 else 0
                print(
                    f"[train_idm] {step+1}/{steps} ({100*(step+1)/steps:.0f}%) | "
                    f"loss={float(loss.item()):.4f} (move={float(loss_move.item()):.4f}, btn={float(loss_btn.item()):.4f}) | "
                    f"{rate:.1f} step/s | ETA: {eta:.0f}s"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    total_time = time.time() - start_time
    print(f"[train_idm] Completed {steps} steps in {total_time:.1f}s ({steps/total_time:.1f} step/s)")

    out_path = Path(cfg.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vpt_cfg,
        },
        out_path,
    )
    print(f"[train_idm] Saved checkpoint to {out_path}")
    return out_path


@torch.no_grad()
def label_actions(cfg: IDMLabelConfig) -> Path:
    """Label/overwrite actions for .npz demos using a trained IDM."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to label actions")

    from src.imitation import InverseDynamicsModel

    ckpt = torch.load(cfg.idm_ckpt, map_location="cpu", weights_only=False)
    vpt_cfg = ckpt.get("config")
    if vpt_cfg is None:
        raise ValueError("Invalid IDM checkpoint: missing config")

    device = _device_from_str(cfg.device)
    _configure_torch_perf(device=device, tf32=True)
    model = InverseDynamicsModel(vpt_cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_amp = device.type == "cuda"

    in_dir = Path(cfg.in_npz_dir)
    out_dir = Path(cfg.out_npz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = int(cfg.context)
    bs = int(cfg.batch_size)
    compress_output = bool(getattr(cfg, "compress_output", True))
    cache_on_device = bool(getattr(cfg, "cache_on_device", False)) and device.type == "cuda"
    cache_fp16 = bool(getattr(cfg, "cache_fp16", True))
    cache_max_bytes = int(getattr(cfg, "cache_max_bytes", 0) or 0)

    paths = list_npz_files(in_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    # CPU prefetch (next file load) to overlap I/O with GPU compute.
    def _load_for_label(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        data = _load_npz_all(path)
        frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
        return data, frames

    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if bool(getattr(cfg, "prefetch", True)) else None
    peek_every = max(0, int(getattr(cfg, "peek_every", 0) or 0))
    peek_samples = max(0, int(getattr(cfg, "peek_samples", 5) or 0))
    peek_every = max(0, int(getattr(cfg, "peek_every", 0) or 0))
    peek_samples = max(0, int(getattr(cfg, "peek_samples", 5) or 0))

    try:
        cur_path = paths[0]
        cur_data, cur_frames = _load_for_label(cur_path)
        next_future: Optional[Future] = None
        if executor is not None and len(paths) > 1:
            next_future = executor.submit(_load_for_label, paths[1])

        for idx_path, path in enumerate(paths):
            if idx_path == 0:
                data, frames = cur_data, cur_frames
            else:
                if next_future is not None:
                    data, frames = next_future.result()
                else:
                    data, frames = _load_for_label(path)

            if executor is not None:
                next_idx = idx_path + 1
                if next_idx + 1 < len(paths):
                    next_future = executor.submit(_load_for_label, paths[next_idx + 1])
                else:
                    next_future = None

            T = int(frames.shape[0])
            act_dim = int(getattr(vpt_cfg, "action_dim", 6))
            pred_actions = np.zeros((T, act_dim), dtype=np.float32)

            if T <= 0:
                continue

            frame_size = tuple(int(x) for x in getattr(vpt_cfg, "frame_size", (224, 224)))
            frames_cache, _ = _cache_episode_on_device(
                frames_u8=frames,
                actions_f32=None,
                device=device,
                frame_size=frame_size,
                channels_last=device.type == "cuda",
                fp16=cache_fp16,
                max_bytes=cache_max_bytes,
            )
            use_gpu_cache = cache_on_device and frames_cache is not None

            padded = None if use_gpu_cache else np.pad(frames, ((ctx, ctx), (0, 0), (0, 0), (0, 0)), mode="edge")
            offsets_np = np.arange(-ctx, ctx + 1, dtype=np.int64)

            with torch.inference_mode():
                for start in range(0, T, bs):
                    end = min(T, start + bs)
                    bsz = end - start
                    if bsz <= 0:
                        continue

                    if use_gpu_cache and frames_cache is not None:
                        center = torch.arange(start, end, device=device, dtype=torch.int64)
                        offsets = torch.arange(-ctx, ctx + 1, device=device, dtype=torch.int64)
                        win_idx = center[:, None] + offsets[None, :]
                        # Edge padding semantics (mode="edge"): clamp to valid range.
                        win_idx = win_idx.clamp_(0, int(frames_cache.shape[0]) - 1)
                        flat = win_idx.reshape(-1)
                        win_flat_t = frames_cache.index_select(0, flat)
                        win_t = win_flat_t.view(bsz, 2 * ctx + 1, 3, win_flat_t.shape[-2], win_flat_t.shape[-1])
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            out = model(win_t)
                        move = out["movement"]
                        btn = torch.sigmoid(out["buttons"]) if out["buttons"].numel() else torch.zeros(
                            move.shape[0], 0, device=device
                        )
                        a = torch.cat([move, btn], dim=-1).clamp(-1.0, 1.0).to("cpu").numpy().astype(np.float32)
                        pred_actions[start:end] = a
                    else:
                        assert padded is not None
                        base = np.arange(start, end, dtype=np.int64)
                        # windows are centered at i with ctx-padding at both ends.
                        win_idx = (base + ctx)[:, None] + offsets_np[None, :]
                        win = padded[win_idx]  # [B,W,H,W,C]
                        win_t = _to_torch_frames(
                            win.reshape(bsz * (2 * ctx + 1), *win.shape[2:]),
                            device=device,
                            out_hw=frame_size,
                            channels_last=device.type == "cuda",
                        )
                        win_t = win_t.view(bsz, 2 * ctx + 1, 3, win_t.shape[-2], win_t.shape[-1])
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            out = model(win_t)
                        move = out["movement"]
                        btn = torch.sigmoid(out["buttons"]) if out["buttons"].numel() else torch.zeros(
                            move.shape[0], 0, device=device
                        )
                        a = torch.cat([move, btn], dim=-1).clamp(-1.0, 1.0).to("cpu").numpy().astype(np.float32)
                        pred_actions[start:end] = a

            out_path = out_dir / path.name
            save_fn = np.savez_compressed if compress_output else np.savez
            save_fn(
                out_path,
                **{k: v for k, v in data.items() if k != "actions"},
                actions=pred_actions,
            )
            print(f"[video_pretraining] Labeled actions -> {out_path}")
            if peek_every and ((idx_path + 1) % peek_every == 0):
                summ = _summarize_actions(pred_actions)
                msg = " ".join(f"{k}={v:.4g}" for k, v in summ.items())
                print(f"[video_pretraining] Peek actions: {out_path.name} {msg}")
                if peek_samples and T > 0:
                    picks = np.linspace(0, max(0, T - 1), num=min(peek_samples, T), dtype=np.int64)
                    for t in picks:
                        print(f"[video_pretraining]  t={int(t)} action={pred_actions[int(t)].tolist()}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return out_dir


def train_reward_model(cfg: RewardTrainConfig) -> Path:
    """Train a self-supervised temporal-ranking reward model on video demos."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to train reward model")

    npz_paths = list_npz_files(cfg.npz_dir, sort=False)
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")
    max_npz_bytes = int(getattr(cfg, "max_npz_bytes", 0) or 0)
    if max_npz_bytes > 0:
        filtered = [p for p in npz_paths if p.stat().st_size <= max_npz_bytes]
        if filtered:
            npz_paths = filtered

    device = _device_from_str(cfg.device)
    rng = np.random.default_rng(int(cfg.seed))

    _configure_torch_perf(device=device, tf32=bool(cfg.tf32))

    model = TemporalRankRewardModel(frame_size=cfg.frame_size, embed_dim=cfg.embed_dim).to(device)
    if bool(cfg.channels_last) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile(model, enabled=bool(cfg.compile) and device.type == "cuda")
    
    # Fused optimizer for GPU efficiency
    use_fused = device.type == "cuda" and _supports_fused_adamw()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), fused=use_fused)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
        use_fused = False
    
    # AMP for mixed precision training  
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    bs = int(cfg.batch_size)
    steps = int(cfg.steps)
    steps_per_load = max(1, int(getattr(cfg, "steps_per_load", 1)))
    
    start_time = time.time()
    log_interval = max(100, steps // 20)
    
    print(
        f"[train_reward_model] Starting: {steps} steps, batch_size={bs}, AMP={use_amp}, fused={use_fused}, "
        f"steps_per_load={steps_per_load}, prefetch={bool(cfg.prefetch)}"
    )

    def _load_reward_npz(path: Path) -> np.ndarray:
        data = _load_npz_selected(path, ("observations",))
        frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
        return frames

    cache_on_device = bool(getattr(cfg, "cache_on_device", False)) and device.type == "cuda"
    cache_fp16 = bool(getattr(cfg, "cache_fp16", True))
    cache_max_bytes = int(getattr(cfg, "cache_max_bytes", 0) or 0)
    cache_requested = cache_on_device

    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if bool(cfg.prefetch) else None
    model.train()
    try:
        cur_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
        next_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
        next_future = _submit_prefetch(executor, _load_reward_npz, next_path)
        frames = _load_reward_npz(cur_path)
        frames_cache, _ = _cache_episode_on_device(
            frames_u8=frames,
            actions_f32=None,
            device=device,
            frame_size=tuple(int(x) for x in cfg.frame_size),
            channels_last=bool(cfg.channels_last) and device.type == "cuda",
            fp16=cache_fp16,
            max_bytes=cache_max_bytes,
        )
        if cache_requested:
            if frames_cache is None:
                print("[train_reward_model] GPU cache: skipped (episode too large / low free VRAM / non-CUDA)")
            else:
                print(
                    f"[train_reward_model] GPU cache: enabled (frames={tuple(frames_cache.shape)}, dtype={frames_cache.dtype})"
                )
        if cache_on_device and frames_cache is None:
            cache_on_device = False

        for step in range(steps):
            if step > 0 and (step % steps_per_load) == 0:
                if next_future is not None:
                    try:
                        frames = next_future.result()
                    except Exception:
                        frames = _load_reward_npz(next_path)
                else:
                    frames = _load_reward_npz(next_path)
                cur_path = next_path
                next_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
                next_future = _submit_prefetch(executor, _load_reward_npz, next_path)
                frames_cache, _ = _cache_episode_on_device(
                    frames_u8=frames,
                    actions_f32=None,
                    device=device,
                    frame_size=tuple(int(x) for x in cfg.frame_size),
                    channels_last=bool(cfg.channels_last) and device.type == "cuda",
                    fp16=cache_fp16,
                    max_bytes=cache_max_bytes,
                )
                if bool(getattr(cfg, "cache_on_device", False)) and frames_cache is not None:
                    cache_on_device = True
                else:
                    cache_on_device = False

            T = int(frames_cache.shape[0] if (cache_on_device and frames_cache is not None) else frames.shape[0])
            if T < 2:
                continue

            if cache_on_device and frames_cache is not None:
                # Sample ordered pairs within a trajectory on-GPU.
                t1 = torch.randint(0, T - 1, (bs,), device=device)
                max_d = (T - 1) - t1  # >=1 since t1 <= T-2
                u = torch.rand((bs,), device=device)
                dt = (u * max_d.to(dtype=torch.float32)).floor().to(dtype=torch.int64) + 1
                t2 = t1 + dt

                f1_t = frames_cache.index_select(0, t1)
                f2_t = frames_cache.index_select(0, t2)
                if bool(cfg.channels_last) and device.type == "cuda":
                    f1_t = f1_t.contiguous(memory_format=torch.channels_last)
                    f2_t = f2_t.contiguous(memory_format=torch.channels_last)
            else:
                # Sample ordered pairs within a trajectory on-CPU.
                t1 = rng.integers(0, T - 1, size=(bs,))
                max_d = np.maximum(1, (T - 1) - t1)
                dt = rng.integers(1, max_d + 1)
                t2 = t1 + dt

                f1 = frames[t1]
                f2 = frames[t2]

                f1_t = _to_torch_frames(
                    f1,
                    device=device,
                    out_hw=tuple(int(x) for x in cfg.frame_size),
                    channels_last=bool(cfg.channels_last) and device.type == "cuda",
                )
                f2_t = _to_torch_frames(
                    f2,
                    device=device,
                    out_hw=tuple(int(x) for x in cfg.frame_size),
                    channels_last=bool(cfg.channels_last) and device.type == "cuda",
                )

            opt.zero_grad(set_to_none=True)
        
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                s1 = model(f1_t)
                s2 = model(f2_t)

                # Bradley-Terry / logistic ranking loss: want s2 > s1.
                logits = s2 - s1
                loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed
                eta = (steps - step - 1) / rate if rate > 0 else 0
                print(
                    f"[train_reward_model] {step+1}/{steps} ({100*(step+1)/steps:.0f}%) | "
                    f"loss={float(loss.item()):.4f} | {rate:.1f} step/s | ETA: {eta:.0f}s"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    total_time = time.time() - start_time
    print(f"[train_reward_model] Completed {steps} steps in {total_time:.1f}s ({steps/total_time:.1f} step/s)")

    out_path = Path(cfg.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "frame_size": cfg.frame_size,
                "embed_dim": int(cfg.embed_dim),
            },
        },
        out_path,
    )
    print(f"[train_reward_model] Saved checkpoint to {out_path}")

    return out_path


@torch.no_grad()
def label_rewards(cfg: RewardLabelConfig) -> Path:
    """Write rewards (and progress scores) into labeled .npz demos."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to label rewards")

    ckpt = torch.load(cfg.reward_ckpt, map_location="cpu", weights_only=False)
    conf = ckpt.get("config") or {}
    frame_size = tuple(conf.get("frame_size") or (224, 224))
    embed_dim = int(conf.get("embed_dim") or 256)

    device = _device_from_str(cfg.device)
    model = TemporalRankRewardModel(frame_size=frame_size, embed_dim=embed_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_amp = device.type == "cuda"

    in_dir = Path(cfg.in_npz_dir)
    out_dir = Path(cfg.out_npz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = int(cfg.batch_size)
    scale = float(cfg.reward_scale)
    compress_output = bool(getattr(cfg, "compress_output", True))
    cache_on_device = bool(getattr(cfg, "cache_on_device", False)) and device.type == "cuda"
    cache_fp16 = bool(getattr(cfg, "cache_fp16", True))
    cache_max_bytes = int(getattr(cfg, "cache_max_bytes", 0) or 0)

    paths = list_npz_files(in_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    def _load_for_label(path: Path) -> Dict[str, np.ndarray]:
        return _load_npz_all(path)

    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if bool(getattr(cfg, "prefetch", True)) else None
    peek_every = max(0, int(getattr(cfg, "peek_every", 0) or 0))
    peek_samples = max(0, int(getattr(cfg, "peek_samples", 5) or 0))

    try:
        cur = _load_for_label(paths[0])
        next_future: Optional[Future] = None
        if executor is not None and len(paths) > 1:
            next_future = executor.submit(_load_for_label, paths[1])

        for i, path in enumerate(paths):
            if i == 0:
                data = cur
            else:
                if next_future is not None:
                    data = next_future.result()
                else:
                    data = _load_for_label(path)

            if executor is not None:
                if i + 2 < len(paths):
                    next_future = executor.submit(_load_for_label, paths[i + 2])
                else:
                    next_future = None

            frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
            T = int(frames.shape[0])
            scores = np.zeros((T,), dtype=np.float32)

            frames_cache, _ = _cache_episode_on_device(
                frames_u8=frames,
                actions_f32=None,
                device=device,
                frame_size=tuple(int(x) for x in frame_size),
                channels_last=device.type == "cuda",
                fp16=cache_fp16,
                max_bytes=cache_max_bytes,
            )
            use_gpu_cache = cache_on_device and frames_cache is not None

            with torch.inference_mode():
                for start in range(0, T, bs):
                    if use_gpu_cache and frames_cache is not None:
                        ft = frames_cache[start : start + bs]
                    else:
                        f = frames[start : start + bs]
                        ft = _to_torch_frames(f, device=device, out_hw=tuple(int(x) for x in frame_size), channels_last=device.type == "cuda")
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        s = model(ft).to("cpu").numpy().astype(np.float32)
                    scores[start : start + len(s)] = s

            rewards = np.zeros((T,), dtype=np.float32)
            if T >= 2:
                rewards[:-1] = (scores[1:] - scores[:-1]) * scale
            if T > 0:
                rewards[-1] = 0.0

            dones = np.asarray(data.get("dones")) if "dones" in data else None
            if dones is None or dones.size != T:
                dones = np.zeros((T,), dtype=bool)
                if T > 0:
                    dones[-1] = True

            out_path = out_dir / path.name
            save_fn = np.savez_compressed if compress_output else np.savez
            save_fn(
                out_path,
                **{k: v for k, v in data.items() if k not in ("rewards", "dones", "progress_scores")},
                rewards=rewards,
                dones=dones,
                progress_scores=scores,
            )
            print(f"[video_pretraining] Labeled rewards -> {out_path}")
            if peek_every and ((i + 1) % peek_every == 0):
                rs = _summarize_rewards(rewards)
                msg = " ".join(f"{k}={v:.4g}" for k, v in rs.items())
                print(f"[video_pretraining] Peek rewards: {out_path.name} {msg}")
                if peek_samples and T > 0:
                    picks = np.linspace(0, max(0, T - 1), num=min(peek_samples, T), dtype=np.int64)
                    for t in picks:
                        print(
                            f"[video_pretraining]  t={int(t)} score={float(scores[int(t)]):.4g} reward={float(rewards[int(t)]):.4g}"
                        )
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return out_dir


@torch.no_grad()
def export_pt_rollouts(cfg: ExportRolloutsConfig) -> Path:
    """Export .pt rollouts (vector obs) from labeled video .npz demos.

    Observations are embeddings produced by the reward model's encoder.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to export rollouts")

    ckpt = torch.load(cfg.reward_ckpt, map_location="cpu", weights_only=False)
    conf = ckpt.get("config") or {}
    frame_size = tuple(conf.get("frame_size") or (224, 224))
    embed_dim = int(conf.get("embed_dim") or 256)

    device = _device_from_str(cfg.device)
    model = TemporalRankRewardModel(frame_size=frame_size, embed_dim=embed_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_amp = device.type == "cuda"

    audio_codebook = None
    audio_code_dim = 0
    audio_ckpt_path = str(getattr(cfg, "audio_token_ckpt", "") or "")
    if audio_ckpt_path:
        try:
            from src.learner.audio_tokens import AudioVQVAE

            audio_ckpt = torch.load(audio_ckpt_path, map_location="cpu", weights_only=False)
            audio_cfg = audio_ckpt.get("config")
            if audio_cfg is not None:
                audio_model = AudioVQVAE(audio_cfg)
                audio_model.load_state_dict(audio_ckpt["model_state_dict"])
                audio_model.eval()
                audio_codebook = audio_model.quantizer.codebook.weight.detach().cpu().numpy().astype(np.float32)
                audio_code_dim = int(audio_codebook.shape[-1])
                print(f"[video_pretraining] Audio tokenizer loaded ({audio_code_dim}D) -> {audio_ckpt_path}")
        except Exception as e:
            print(f"[video_pretraining] Audio tokenizer load failed: {e}")
            audio_codebook = None
            audio_code_dim = 0

    in_dir = Path(cfg.in_npz_dir)
    out_dir = Path(cfg.out_pt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = int(cfg.batch_size)
    cache_on_device = bool(getattr(cfg, "cache_on_device", False)) and device.type == "cuda"
    cache_fp16 = bool(getattr(cfg, "cache_fp16", True))
    cache_max_bytes = int(getattr(cfg, "cache_max_bytes", 0) or 0)

    paths = list_npz_files(in_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    def _load_for_export(path: Path) -> Dict[str, np.ndarray]:
        # Need to carry actions/rewards/dones through.
        return _load_npz_all(path)

    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if bool(getattr(cfg, "prefetch", True)) else None

    try:
        cur = _load_for_export(paths[0])
        next_future: Optional[Future] = None
        if executor is not None and len(paths) > 1:
            next_future = executor.submit(_load_for_export, paths[1])

        for i, path in enumerate(paths):
            if i == 0:
                data = cur
            else:
                if next_future is not None:
                    data = next_future.result()
                else:
                    data = _load_for_export(path)

            if executor is not None:
                if i + 2 < len(paths):
                    next_future = executor.submit(_load_for_export, paths[i + 2])
                else:
                    next_future = None

            frames = _ensure_uint8_frames(np.asarray(data.get("observations")))
            if "actions" not in data:
                raise ValueError(f"{path.name} missing actions; run IDM labeling first")
            if "rewards" not in data:
                raise ValueError(f"{path.name} missing rewards; run reward labeling first")
            actions = np.asarray(data.get("actions"), dtype=np.float32)
            rewards = np.asarray(data.get("rewards"), dtype=np.float32)
            dones = np.asarray(data.get("dones")) if "dones" in data else np.zeros((len(frames),), dtype=bool)

            T = int(frames.shape[0])
            obs_emb = np.zeros((T, embed_dim), dtype=np.float32)

            frames_cache, _ = _cache_episode_on_device(
                frames_u8=frames,
                actions_f32=None,
                device=device,
                frame_size=tuple(int(x) for x in frame_size),
                channels_last=device.type == "cuda",
                fp16=cache_fp16,
                max_bytes=cache_max_bytes,
            )
            use_gpu_cache = cache_on_device and frames_cache is not None

            with torch.inference_mode():
                for start in range(0, T, bs):
                    if use_gpu_cache and frames_cache is not None:
                        ft = frames_cache[start : start + bs]
                    else:
                        f = frames[start : start + bs]
                        ft = _to_torch_frames(
                            f,
                            device=device,
                            out_hw=tuple(int(x) for x in frame_size),
                            channels_last=device.type == "cuda",
                        )
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        _s, z = model(ft, return_embed=True)
                    obs_emb[start : start + z.shape[0]] = z.to("cpu").numpy().astype(np.float32)

            audio_tokens = None
            if audio_codebook is not None and audio_code_dim > 0:
                audio_tokens = data.get("audio_tokens")
                if audio_tokens is None and not bool(getattr(cfg, "audio_optional", True)):
                    raise ValueError(f"{path.name} missing audio_tokens; run audio labeling first")
                if audio_tokens is not None:
                    tok = np.asarray(audio_tokens, dtype=np.int64).reshape(-1)
                    if tok.shape[0] != T:
                        raise ValueError(f"{path.name} audio_tokens length {tok.shape[0]} != T={T}")
                    emb = np.zeros((T, audio_code_dim), dtype=np.float32)
                    valid = tok >= 0
                    if valid.any():
                        safe = np.clip(tok[valid], 0, audio_codebook.shape[0] - 1)
                        emb[valid] = audio_codebook[safe]
                    scale = float(getattr(cfg, "audio_embed_scale", 1.0) or 1.0)
                    if scale != 1.0:
                        emb *= scale
                    obs_emb = np.concatenate([obs_emb, emb], axis=-1)

            ep = {
                "observations": torch.from_numpy(obs_emb).float(),
                "actions": torch.from_numpy(actions).float(),
                "rewards": torch.from_numpy(rewards).float(),
                "dones": torch.from_numpy(dones.astype(np.bool_)),
            }
            if audio_tokens is not None:
                ep["audio_tokens"] = torch.from_numpy(np.asarray(audio_tokens).astype(np.int32))
            out_path = out_dir / f"{path.stem}.pt"
            torch.save(ep, out_path)
            print(f"[video_pretraining] Exported rollout -> {out_path}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    return out_dir


@torch.no_grad()
def label_skill_tokens(cfg: SkillLabelConfig) -> Path:
    """Add discrete skill token sequences to labeled .npz demos.

    Uses a trained SkillVQVAE to encode sliding windows of action sequences
    into token IDs. The output `skill_tokens` is a per-step int array with:
      - -1 for steps not covered by any window
      - a non-negative token id elsewhere
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to label skill tokens")

    from src.learner.skill_tokens import SkillVQVAE

    ckpt = torch.load(cfg.skill_ckpt, map_location="cpu", weights_only=False)
    vq_cfg = ckpt.get("config")
    if vq_cfg is None:
        raise ValueError("Invalid skill checkpoint: missing config")

    device = _device_from_str(cfg.device)
    model = SkillVQVAE(vq_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    in_dir = Path(cfg.in_npz_dir)
    out_dir = Path(cfg.out_npz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = int(cfg.seq_len)
    stride = int(cfg.stride)
    bs = int(getattr(cfg, "batch_size", 256))
    compress_output = bool(getattr(cfg, "compress_output", True))

    paths = list_npz_files(in_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    save_fn = np.savez_compressed if compress_output else np.savez
    peek_every = max(0, int(getattr(cfg, "peek_every", 0) or 0))
    peek_samples = max(0, int(getattr(cfg, "peek_samples", 5) or 0))

    with torch.inference_mode():
        for file_i, path in enumerate(paths):
            data = _load_npz_all(path)
            actions = np.asarray(data.get("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] < seq_len:
                continue

            T = int(actions.shape[0])
            tokens = np.full((T,), -1, dtype=np.int32)

            starts = list(range(0, T - seq_len + 1, max(1, stride)))
            if not starts:
                continue

            for i in range(0, len(starts), bs):
                batch_starts = starts[i : i + bs]
                chunks = np.stack([actions[s : s + seq_len] for s in batch_starts], axis=0).astype(np.float32)
                a_t = torch.from_numpy(chunks).to(device=device, dtype=torch.float32)
                _zq, idx = model.encode(a_t)
                toks = idx.to("cpu", non_blocking=False).numpy().astype(np.int32).reshape(-1)
                for s, tok in zip(batch_starts, toks):
                    tokens[s : s + seq_len] = int(tok)

            out_path = out_dir / path.name
            save_fn(
                out_path,
                **{k: v for k, v in data.items() if k != "skill_tokens"},
                skill_tokens=tokens,
            )
            print(f"[video_pretraining] Labeled skills -> {out_path}")
            if peek_every and ((file_i + 1) % peek_every == 0):
                summ = _summarize_tokens(tokens, topk=10)
                print(
                    f"[video_pretraining] Peek skills: {out_path.name} "
                    f"T={summ.get('T')} unlabeled_frac={float(summ.get('unlabeled_frac', 1.0)):.4g} "
                    f"unique={summ.get('unique')} top={summ.get('top')}"
                )
                if peek_samples and T > 0:
                    picks = np.linspace(0, max(0, T - 1), num=min(peek_samples, T), dtype=np.int64)
                    for t in picks:
                        print(f"[video_pretraining]  t={int(t)} skill={int(tokens[int(t)])}")

    return out_dir


def train_audio_tokens(cfg: AudioTokenTrainConfig) -> Path:
    """Train audio VQ-VAE tokenizer on per-frame audio chunks."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to train audio tokens")

    from src.learner.audio_tokens import AudioMelSpec, AudioVQVAE, AudioVQConfig

    npz_paths = list_npz_files(cfg.npz_dir, sort=False)
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")

    max_npz_bytes = int(getattr(cfg, "max_npz_bytes", 0) or 0)
    if max_npz_bytes > 0:
        filtered = [p for p in npz_paths if p.stat().st_size <= max_npz_bytes]
        if filtered:
            npz_paths = filtered

    rng = np.random.default_rng(int(cfg.seed))
    device = _device_from_str(cfg.device)

    def _load_audio_npz(path: Path) -> Optional[np.ndarray]:
        data = _load_npz_selected(path, ("audio",))
        if "audio" not in data:
            return None
        audio = np.asarray(data.get("audio"))
        if audio.ndim != 2 or audio.shape[0] <= 0:
            return None
        return audio

    # Find a sample audio segment to infer segment length.
    sample_audio = None
    for p in npz_paths:
        sample_audio = _load_audio_npz(p)
        if sample_audio is not None:
            break
    if sample_audio is None:
        raise ValueError(f"No audio arrays found in {cfg.npz_dir}")

    samples_per_frame = int(sample_audio.shape[1])
    context_frames = max(1, int(cfg.context_frames))
    segment_samples = samples_per_frame * context_frames

    vq_cfg = AudioVQConfig(
        sample_rate=int(cfg.sample_rate),
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop_length),
        win_length=int(cfg.win_length),
        n_mels=int(cfg.n_mels),
        segment_samples=int(segment_samples),
        num_codes=int(cfg.num_codes),
        code_dim=int(cfg.code_dim),
        encoder_hidden=int(cfg.encoder_hidden),
        commitment_cost=float(cfg.commitment_cost),
        mel_eps=float(cfg.mel_eps),
    )

    model = AudioVQVAE(vq_cfg).to(device)
    mel_spec = AudioMelSpec(vq_cfg).to(device)
    if bool(cfg.compile) and device.type == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception:
            pass

    if bool(cfg.tf32) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if bool(getattr(cfg, "prefetch", True)) else None

    try:
        cur = _load_audio_npz(npz_paths[0])
        next_future: Optional[Future] = None
        if executor is not None and len(npz_paths) > 1:
            next_future = executor.submit(_load_audio_npz, npz_paths[1])

        steps_per_load = max(1, int(cfg.steps_per_load))
        total_steps = int(cfg.steps)
        bs = int(cfg.batch_size)
        step = 0
        while step < total_steps:
            # Rotate current buffer
            if cur is None:
                # Advance until we get a valid audio array.
                if next_future is not None:
                    cur = next_future.result()
                else:
                    cur = _load_audio_npz(npz_paths[int(rng.integers(0, len(npz_paths)))])
            if cur is None:
                step += 1
                continue

            if executor is not None:
                next_path = npz_paths[int(rng.integers(0, len(npz_paths)))]
                next_future = executor.submit(_load_audio_npz, next_path)

            T = int(cur.shape[0])
            max_start = max(0, T - context_frames)

            for _ in range(steps_per_load):
                if step >= total_steps:
                    break
                if max_start <= 0:
                    step += 1
                    continue

                starts = rng.integers(0, max_start + 1, size=bs)
                segments = np.stack(
                    [
                        cur[s : s + context_frames].reshape(-1)
                        for s in starts
                    ],
                    axis=0,
                ).astype(np.float32)
                wave = torch.from_numpy(segments).to(device=device, dtype=torch.float32) / 32768.0
                mel = mel_spec(wave)
                _recon, _idx, loss, metrics = model.forward_mel(mel)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if (step + 1) % 200 == 0 or step == 0:
                    usage = float(metrics.get("codebook_utilization", torch.tensor(0.0)).detach().item())
                    print(
                        f"[video_pretraining] AudioVQ step {step+1}/{total_steps} "
                        f"loss={float(loss.detach().item()):.4f} usage={usage:.3f}"
                    )
                step += 1

            # Move to prefetched audio next.
            if next_future is not None:
                cur = next_future.result()
            else:
                cur = None
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    out = Path(cfg.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vq_cfg,
        },
        out,
    )
    print(f"[video_pretraining] Saved audio VQ-VAE -> {out}")
    return out


@torch.no_grad()
def label_audio_tokens(cfg: AudioLabelConfig) -> Path:
    """Add discrete audio token sequences to labeled .npz demos."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to label audio tokens")

    from src.learner.audio_tokens import AudioMelSpec, AudioVQVAE

    ckpt = torch.load(cfg.audio_ckpt, map_location="cpu", weights_only=False)
    vq_cfg = ckpt.get("config")
    if vq_cfg is None:
        raise ValueError("Invalid audio tokenizer checkpoint: missing config")

    device = _device_from_str(cfg.device)
    model = AudioVQVAE(vq_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mel_spec = AudioMelSpec(vq_cfg).to(device)

    in_dir = Path(cfg.in_npz_dir)
    out_dir = Path(cfg.out_npz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_frames = max(1, int(cfg.context_frames))
    stride = max(1, int(cfg.stride))
    bs = int(cfg.batch_size)
    compress_output = bool(getattr(cfg, "compress_output", True))

    paths = list_npz_files(in_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {in_dir}")

    save_fn = np.savez_compressed if compress_output else np.savez
    peek_every = max(0, int(getattr(cfg, "peek_every", 0) or 0))
    peek_samples = max(0, int(getattr(cfg, "peek_samples", 5) or 0))

    for file_i, path in enumerate(paths):
        data = _load_npz_all(path)
        audio = np.asarray(data.get("audio")) if "audio" in data else None
        if audio is None or audio.ndim != 2:
            continue

        T = int(audio.shape[0])
        samples_per_frame = int(audio.shape[1])
        expected_samples = int(getattr(vq_cfg, "segment_samples", 0) or 0)
        if expected_samples > 0:
            derived = max(1, expected_samples // max(1, samples_per_frame))
            if derived != context_frames:
                context_frames = derived
        tokens = np.full((T,), -1, dtype=np.int32)
        max_start = T - context_frames
        if max_start < 0:
            continue

        starts = list(range(0, max_start + 1, stride))
        if not starts:
            continue

        for i in range(0, len(starts), bs):
            batch_starts = starts[i : i + bs]
            segments = np.stack(
                [audio[s : s + context_frames].reshape(-1) for s in batch_starts],
                axis=0,
            ).astype(np.float32)
            wave = torch.from_numpy(segments).to(device=device, dtype=torch.float32) / 32768.0
            mel = mel_spec(wave)
            _zq, idx = model.encode_mel(mel)
            toks = idx.to("cpu", non_blocking=False).numpy().astype(np.int32).reshape(-1)
            for s, tok in zip(batch_starts, toks):
                tokens[s : s + context_frames] = int(tok)

        out_path = out_dir / path.name
        save_fn(
            out_path,
            **{k: v for k, v in data.items() if k != "audio_tokens"},
            audio_tokens=tokens,
        )
        print(f"[video_pretraining] Labeled audio -> {out_path}")
        if peek_every and ((file_i + 1) % peek_every == 0):
            summ = _summarize_tokens(tokens, topk=10)
            print(
                f"[video_pretraining] Peek audio: {out_path.name} "
                f"T={summ.get('T')} unlabeled_frac={float(summ.get('unlabeled_frac', 1.0)):.4g} "
                f"unique={summ.get('unique')} top={summ.get('top')}"
            )
            if peek_samples and T > 0:
                picks = np.linspace(0, max(0, T - 1), num=min(peek_samples, T), dtype=np.int64)
                for t in picks:
                    print(f"[video_pretraining]  t={int(t)} audio_tok={int(tokens[int(t)])}")

    return out_dir


def inspect_labels(cfg: InspectLabelsConfig) -> Dict[str, Any]:
    """Summarize labeled actions/rewards/skill tokens for a few NPZ files.

    Intended as a fast progress check (not a full dataset analysis).
    """
    paths = list_npz_files(cfg.npz_dir, sort=True)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {cfg.npz_dir}")

    rng = np.random.default_rng(int(cfg.seed))
    n = max(1, min(int(cfg.num_files), len(paths)))
    idxs = rng.choice(len(paths), size=n, replace=False)
    idxs.sort()

    report: Dict[str, Any] = {"dir": str(cfg.npz_dir), "files": []}
    for i in idxs:
        p = paths[int(i)]
        d = _load_npz_selected(p, ("actions", "rewards", "skill_tokens", "audio_tokens", "progress_scores"))
        file_rep: Dict[str, Any] = {"file": p.name}
        if "actions" in d:
            file_rep["actions"] = _summarize_actions(d["actions"])
        if "rewards" in d:
            file_rep["rewards"] = _summarize_rewards(d["rewards"])
        if "skill_tokens" in d:
            file_rep["skills"] = _summarize_tokens(d["skill_tokens"], topk=int(cfg.topk))
        if "audio_tokens" in d:
            file_rep["audio"] = _summarize_tokens(d["audio_tokens"], topk=int(cfg.topk))
        report["files"].append(file_rep)

        parts = [f"[inspect] {p.name}"]
        if "actions" in file_rep:
            a = file_rep["actions"]
            parts.append(
                f"actions(D={int(a.get('D', 0))} move_mag_mean={a.get('move_mag_mean', 0.0):.3g} "
                f"btn_on@0.5={a.get('btn_on_frac_0.5', 0.0):.3g})"
            )
        if "rewards" in file_rep:
            r = file_rep["rewards"]
            parts.append(f"rewards(mean={r.get('r_mean', 0.0):.3g} std={r.get('r_std', 0.0):.3g})")
        if "skills" in file_rep:
            s = file_rep["skills"]
            parts.append(f"skills(unlabeled={s.get('unlabeled_frac', 1.0):.3g} unique={s.get('unique', 0)})")
        if "audio" in file_rep:
            a = file_rep["audio"]
            parts.append(f"audio(unlabeled={a.get('unlabeled_frac', 1.0):.3g} unique={a.get('unique', 0)})")
        print(" | ".join(parts))

        samples = max(0, int(cfg.samples_per_file))
        if samples > 0:
            T = 0
            if "actions" in d and np.asarray(d["actions"]).ndim == 2:
                T = int(d["actions"].shape[0])
            elif "rewards" in d:
                T = int(np.asarray(d["rewards"]).reshape(-1).shape[0])
            elif "skill_tokens" in d:
                T = int(np.asarray(d["skill_tokens"]).reshape(-1).shape[0])
            elif "audio_tokens" in d:
                T = int(np.asarray(d["audio_tokens"]).reshape(-1).shape[0])
            if T > 0:
                picks = np.linspace(0, max(0, T - 1), num=min(samples, T), dtype=np.int64)
                for t in picks:
                    row = [f"t={int(t)}"]
                    if "actions" in d:
                        row.append(f"action={np.asarray(d['actions'])[int(t)].astype(np.float32).tolist()}")
                    if "rewards" in d:
                        row.append(f"reward={float(np.asarray(d['rewards']).reshape(-1)[int(t)]):.4g}")
                    if "skill_tokens" in d:
                        row.append(f"skill={int(np.asarray(d['skill_tokens']).reshape(-1)[int(t)])}")
                    if "audio_tokens" in d:
                        row.append(f"audio_tok={int(np.asarray(d['audio_tokens']).reshape(-1)[int(t)])}")
                    print("[inspect]  " + " ".join(row))

    if cfg.json_out:
        out = Path(cfg.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(report, indent=2))
        print(f"[inspect] Wrote report -> {out}")

    return report
