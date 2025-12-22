#!/usr/bin/env python3
"""Menu benchmark harness.

Expected dataset layout:
  dataset/
    images/
      0001.jpg
      0002.jpg
    labels.json

labels.json format:
  {
    "0001.jpg": {"goal": "Start Run", "expected_label": "btn_confirm"},
    "0002.jpg": {"goal": "Continue", "expected_text": "Continue"}
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from PIL import Image

from src.vision.som_preprocess import SoMPreprocessor
from src.control.menu_reasoner import MenuReasoner


def main() -> int:
    ap = argparse.ArgumentParser(description="Menu benchmark for SoM + VLM")
    ap.add_argument("--dataset", required=True, help="Dataset root with images/ and labels.json")
    ap.add_argument("--max", type=int, default=0, help="Limit number of samples")
    args = ap.parse_args()

    root = Path(args.dataset)
    labels_path = root / "labels.json"
    images_dir = root / "images"
    if not labels_path.exists():
        raise SystemExit("labels.json missing")
    if not images_dir.exists():
        raise SystemExit("images/ missing")

    labels: Dict[str, Any] = json.loads(labels_path.read_text())
    som = SoMPreprocessor()
    reasoner = MenuReasoner()

    total = 0
    correct = 0

    for name, meta in labels.items():
        if args.max and total >= args.max:
            break
        img_path = images_dir / name
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        overlay, _, mapping = som.process(img)
        goal = str(meta.get("goal") or "Start Run")
        hint = str(meta.get("hint") or "")
        action = reasoner.infer_action(overlay, mapping, goal=goal, hint=hint)
        total += 1
        expected_label = str(meta.get("expected_label") or "")
        expected_text = str(meta.get("expected_text") or "")
        if action and action.target_id:
            chosen = None
            for item in mapping:
                if int(item.get("id", -1)) == int(action.target_id):
                    chosen = item
                    break
            if chosen:
                label = str(chosen.get("label") or "")
                text = str(chosen.get("text") or "")
                if expected_label and expected_label == label:
                    correct += 1
                elif expected_text and expected_text.lower() in text.lower():
                    correct += 1
        print(f"{name}: action={action} expected_label={expected_label} expected_text={expected_text}")

    acc = (correct / total) if total else 0.0
    print(f"accuracy={acc:.3f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
