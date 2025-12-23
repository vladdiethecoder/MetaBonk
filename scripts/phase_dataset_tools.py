#!/usr/bin/env python3
"""Utilities for inspecting MetaBonk phase datasets.

The phase dataset is written as .npz files containing:
  - frames: uint8 array (T,H,W,3)
  - label: string
  - meta: JSON string
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional


def _iter_npz_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.npz")):
        if p.is_file():
            yield p


def cmd_counts(root: Path) -> int:
    if not root.exists():
        print(f"missing: {root}")
        return 2
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        count = sum(1 for _ in d.glob("*.npz"))
        print(f"{d.name}: {count}")
    total = sum(1 for _ in _iter_npz_files(root))
    print(f"total: {total}")
    return 0


def _load_npz(path: Path):
    import numpy as np

    data = np.load(path, allow_pickle=False)
    frames = data["frames"]
    label = str(data.get("label", ""))
    meta_raw = data.get("meta")
    meta = None
    if meta_raw is not None:
        try:
            meta = json.loads(str(meta_raw))
        except Exception:
            meta = {"raw": str(meta_raw)}
    return frames, label, meta


def cmd_export(
    root: Path,
    out_dir: Path,
    *,
    label: Optional[str],
    n: int,
    seed: int,
    frame_idx: str,
) -> int:
    if not root.exists():
        print(f"missing: {root}")
        return 2
    if label:
        root = root / label
    if not root.exists():
        print(f"missing: {root}")
        return 2
    files = list(root.glob("*.npz"))
    if not files:
        print(f"no .npz files in {root}")
        return 2
    rnd = random.Random(seed)
    rnd.shuffle(files)
    files = files[: max(1, int(n))]
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for idx, p in enumerate(files):
        frames, lbl, meta = _load_npz(p)
        t = int(frames.shape[0]) if hasattr(frames, "shape") and len(frames.shape) >= 1 else 0
        if not t:
            continue
        if frame_idx == "first":
            ti = 0
        elif frame_idx == "last":
            ti = t - 1
        else:
            ti = t // 2
        img = Image.fromarray(frames[ti].astype("uint8"))
        out_path = out_dir / f"{idx:03d}_{lbl}_{p.stem}.jpg"
        img.save(out_path)
        if meta:
            meta_path = out_dir / f"{idx:03d}_{lbl}_{p.stem}.json"
            try:
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            except Exception:
                pass
    print(f"wrote: {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get("METABONK_PHASE_DATASET_DIR", "temp/phase_dataset"),
        help="Dataset root directory",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_counts = sub.add_parser("counts", help="Print counts per label")
    _ = p_counts

    p_export = sub.add_parser("export", help="Export sample frames as JPG (and meta JSON)")
    p_export.add_argument("--label", default=None, help="Label directory under root")
    p_export.add_argument("--out", default="temp/phase_dataset_previews", help="Output directory")
    p_export.add_argument("--n", type=int, default=20, help="Number of samples")
    p_export.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    p_export.add_argument(
        "--frame",
        choices=("first", "mid", "last"),
        default="mid",
        help="Which frame in the clip to export",
    )

    args = parser.parse_args()
    root = Path(str(args.root)).expanduser()
    if args.cmd == "counts":
        return cmd_counts(root)
    if args.cmd == "export":
        out = Path(str(args.out)).expanduser()
        return cmd_export(
            root,
            out,
            label=str(args.label) if args.label else None,
            n=int(args.n),
            seed=int(args.seed),
            frame_idx=str(args.frame),
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

