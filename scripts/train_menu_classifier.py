#!/usr/bin/env python3
"""Train menu/gameplay classifier for MegaBonk UI states.

Dataset layout:
  dataset_root/
    train/<class>/*.jpg
    val/<class>/*.jpg
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.vision.menu_classifier import build_menu_model


def _transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _epoch(model, loader, opt, device, train: bool) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = crit(logits, y)
            if train:
                loss.backward()
                opt.step()
        loss_sum += float(loss.detach().cpu()) * x.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))
    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    return loss_avg, acc


def main() -> int:
    ap = argparse.ArgumentParser(description="Train menu classifier")
    ap.add_argument("--data", default="data/menu_dataset", help="Dataset root")
    ap.add_argument("--out", default="checkpoints/menu_classifier.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=os.environ.get("METABONK_MENU_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--pretrained", action="store_true", help="Use pretrained backbone weights")
    args = ap.parse_args()

    root = Path(args.data)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("dataset must contain train/ and val/ subfolders")

    train_ds = datasets.ImageFolder(str(train_dir), transform=_transforms(True))
    val_ds = datasets.ImageFolder(str(val_dir), transform=_transforms(False))
    num_classes = len(train_ds.classes)
    if num_classes < 2:
        raise SystemExit("need at least 2 classes to train")

    train_loader = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch), shuffle=False, num_workers=2)

    device = torch.device(str(args.device))
    model = build_menu_model(num_classes=num_classes, pretrained=bool(args.pretrained)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    best = 0.0
    for epoch in range(int(args.epochs)):
        tr_loss, tr_acc = _epoch(model, train_loader, opt, device, train=True)
        va_loss, va_acc = _epoch(model, val_loader, opt, device, train=False)
        print(f"[menu] epoch {epoch+1}/{args.epochs} train_loss={tr_loss:.4f} acc={tr_acc:.3f} val_loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc >= best:
            best = va_acc
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, out)
            print(f"[menu] saved {out} (best acc={best:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
