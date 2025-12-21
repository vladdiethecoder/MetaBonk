#!/usr/bin/env python3
"""Train a YOLO model for MegaBonk UI/world detection.

Usage:
  python scripts/train_yolo.py --data configs/megabonk_dataset.yaml

This script applies UI-specific augmentation overrides:
  - mosaic=0.0 (disable)
  - fliplr=0.0 (disable)
  - scale=0.5
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> int:
    p = argparse.ArgumentParser(description="Train YOLO for MegaBonk")
    p.add_argument("--weights", default="yolo11n.pt", help="Base weights to fine-tune")
    p.add_argument("--data", default="configs/megabonk_dataset.yaml", help="Dataset YAML")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="cuda, cpu, or device id")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        mosaic=0.0,
        fliplr=0.0,
        scale=0.5,
        hsv_s=0.7,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

