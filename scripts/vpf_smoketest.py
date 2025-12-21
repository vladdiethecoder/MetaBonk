#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="PyNvVideoCodec (VPF) encode smoketest (H.264 Annex-B)")
    parser.add_argument("--out", default="/tmp/metabonk_vpf_smoketest.h264")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bitrate", default="6M")
    parser.add_argument("--gop", type=int, default=60)
    args = parser.parse_args()

    try:
        import PyNvVideoCodec as nvc  # type: ignore
        import cvcuda  # type: ignore
        import cupy as cp  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing deps. Install: pip install PyNvVideoCodec cvcuda-cu13 cupy-cuda13x"
        ) from e

    w = int(args.width)
    h = int(args.height)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a CV-CUDA RGB tensor we can fill via a CuPy view without host round-trips.
    rgb = cvcuda.Tensor((h, w, 3), cvcuda.Type.U8, cvcuda.TensorLayout.HWC)
    rgb_view = cp.asarray(rgb.cuda())

    enc = nvc.CreateEncoder(
        width=w,
        height=h,
        fmt="NV12",
        codec="h264",
        usecpuinputbuffer=False,
        preset=os.environ.get("METABONK_VPF_PRESET", "P1"),
        tuning_info=os.environ.get("METABONK_VPF_TUNING", "low_latency"),
        bitrate=str(args.bitrate),
        fps=str(int(args.fps)),
        gop=str(int(args.gop)),
    )

    with out_path.open("wb") as f:
        for i in range(int(args.frames)):
            rgb_view.fill(0)
            x0 = (i * 7) % w
            rgb_view[:, x0 : x0 + 12, 0] = 255
            nv12 = cvcuda.cvtcolor(rgb, cvcuda.ColorConversion.RGB2YUV_NV12)
            bs = enc.Encode(nv12)
            if bs:
                f.write(bs)
        tail = enc.EndEncode()
        if tail:
            f.write(tail)

    print(f"[vpf_smoketest] wrote: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

