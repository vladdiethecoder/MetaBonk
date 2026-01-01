"""CuTile observations for MetaBonk.

In MetaBonk, the "CuTile" contract means:
- GPU-resident preprocessing (no CPU readbacks in the production path)
- Tile-based downsampling using the `cuda.tile` + CuPy stack when available
- Deterministic, integer-scale downsample for stable learning inputs

This implementation targets the existing pixel-policy path by returning an
RGB CHW `torch.uint8` tensor suitable for `VisionActorCritic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CuTileObsConfig:
    out_size: Tuple[int, int] = (128, 128)
    tile: int = 32


class CuTileObservations:
    """GPU cuTile observation extractor (uint8 CHW)."""

    def __init__(self, *, cfg: Optional[CuTileObsConfig] = None) -> None:
        self.cfg = cfg or CuTileObsConfig()

        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"CuTileObservations requires torch: {e}") from e

        from src.worker.gpu_preprocess import HAS_CUTILE

        if not torch.cuda.is_available():  # pragma: no cover
            raise RuntimeError("CuTileObservations requires torch CUDA (torch.cuda.is_available() is False)")
        if not HAS_CUTILE:  # pragma: no cover
            raise RuntimeError("CuTileObservations requires cuTile+CuPy (install cuda-tile + cupy-cuda13x)")

        out_h, out_w = (int(self.cfg.out_size[0]), int(self.cfg.out_size[1]))
        tile = int(self.cfg.tile)
        if out_h <= 0 or out_w <= 0:
            raise ValueError("CuTileObsConfig.out_size must be positive")
        if (out_h % tile) != 0 or (out_w % tile) != 0:
            raise ValueError(f"CuTile out_size must be multiples of {tile} (got {out_h}x{out_w})")

    def extract_from_chw_u8(self, frame_chw_u8: "torch.Tensor") -> "torch.Tensor":
        """Extract a CuTile observation from a CHW uint8 CUDA tensor."""
        import torch  # type: ignore

        if frame_chw_u8.ndim != 3:
            raise ValueError(f"expected CHW, got shape={tuple(frame_chw_u8.shape)}")
        if frame_chw_u8.dtype != torch.uint8:
            raise ValueError(f"expected uint8 input, got dtype={frame_chw_u8.dtype}")
        if not frame_chw_u8.is_cuda:
            raise RuntimeError("CuTileObservations requires a CUDA tensor input")

        out_h, out_w = (int(self.cfg.out_size[0]), int(self.cfg.out_size[1]))

        # 1) Center-crop to square.
        from src.worker.frame_normalizer import center_crop_aspect_chw, center_crop_hw_chw

        cropped = center_crop_aspect_chw(frame_chw_u8, 1.0)

        # 2) Center-crop again to ensure integer-scale downsample uses the center
        #    (cuTile kernel reads from top-left of the input tensor).
        _c, h, w = int(cropped.shape[0]), int(cropped.shape[1]), int(cropped.shape[2])
        scale_h = max(1, int(h) // int(out_h))
        scale_w = max(1, int(w) // int(out_w))
        used_h = int(out_h) * int(scale_h)
        used_w = int(out_w) * int(scale_w)
        cropped = center_crop_hw_chw(cropped, out_h=used_h, out_w=used_w)

        # 3) cuTile downsample+normalize (returns float CHW in [0,1]).
        from src.worker.gpu_preprocess import PreprocessConfig, preprocess_frame

        y_f = preprocess_frame(
            cropped,
            cfg=PreprocessConfig(out_size=(out_h, out_w), to_grayscale=False),
            backend="cutile",
            device=str(frame_chw_u8.device),
        )

        # 4) Return uint8 CHW for the vision policy.
        y_u8 = (y_f.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8).contiguous()
        return y_u8


def compare_observation_sizes(*, src_hw: Tuple[int, int] = (1080, 1920), out_hw: Tuple[int, int] = (128, 128)) -> dict:
    """Return a small summary comparing raw RGB vs downsampled RGB sizes."""
    src_h, src_w = (int(src_hw[0]), int(src_hw[1]))
    out_h, out_w = (int(out_hw[0]), int(out_hw[1]))
    pix_bytes = src_h * src_w * 3
    out_bytes = out_h * out_w * 3
    return {
        "src_mb": float(pix_bytes) / 1e6,
        "out_mb": float(out_bytes) / 1e6,
        "reduction_x": float(pix_bytes) / float(max(1, out_bytes)),
    }


__all__ = ["CuTileObsConfig", "CuTileObservations", "compare_observation_sizes"]

