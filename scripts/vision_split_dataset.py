#!/usr/bin/env python3
"""Split an image-folder dataset into train/val splits.

Input layout:
  dataset_root/
    menu/
    combat/
    reward/
    selection/

Output layout:
  out_root/
    train/<class>/
    val/<class>/
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _iter_images(path: Path) -> Iterable[Path]:
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _copy_file(src: Path, dst: Path, use_link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_link:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Split image-folder dataset into train/val")
    ap.add_argument("--input", required=True, help="Dataset root with class subfolders")
    ap.add_argument("--out", default="data/menu_dataset", help="Output root")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--copy", action="store_true", help="Copy instead of hardlink")
    args = ap.parse_args()

    src_root = Path(args.input)
    if not src_root.exists():
        raise SystemExit(f"input not found: {src_root}")
    out_root = Path(args.out)
    rng = random.Random(int(args.seed))
    classes = [p for p in src_root.iterdir() if p.is_dir()]
    if not classes:
        raise SystemExit("no class subfolders found")

    for cls in classes:
        images = list(_iter_images(cls))
        if not images:
            continue
        rng.shuffle(images)
        split = int(len(images) * float(args.val))
        val_imgs = images[:split]
        train_imgs = images[split:]
        for p in train_imgs:
            _copy_file(p, out_root / "train" / cls.name / p.name, use_link=not args.copy)
        for p in val_imgs:
            _copy_file(p, out_root / "val" / cls.name / p.name, use_link=not args.copy)
        print(f"[split] {cls.name}: train={len(train_imgs)} val={len(val_imgs)}")

    print(f"[split] output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
