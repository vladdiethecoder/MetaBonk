#!/usr/bin/env python3
"""MAE-style self-supervised pretraining for MetaBonkVisionEncoder.

Example:
  python scripts/pretrain_vision_mae.py --data-dir /tmp/metabonk_frames --out checkpoints/vision_encoder_mae.pt

Notes:
  - Prefer training on the same resolution you will use for pixel PPO rollouts
    (e.g., 128x128) and set a smaller patch size (e.g., 4) so the token grid has
    enough spatial capacity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from src.agent.vision.encoder import MetaBonkVisionEncoder, VisionEncoderConfig
from src.agent.pretraining.mae import MaskedAutoencoder, MAEConfig


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, size: Tuple[int, int]):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"data dir not found: {root}")
        self.size = (int(size[0]), int(size[1]))
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        self.files: List[Path] = [p for p in sorted(self.root.rglob("*")) if p.suffix.lower() in exts]
        if not self.files:
            raise FileNotFoundError(f"no images found under: {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from PIL import Image  # local import

        p = self.files[int(idx)]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
        x = torch.from_numpy(__import__("numpy").asarray(img, dtype="uint8"))  # (H,W,3)
        x = x.permute(2, 0, 1).contiguous()  # (3,H,W) uint8
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory of PNG/JPEG frames")
    ap.add_argument("--out", default="checkpoints/vision_encoder_mae.pt", help="Output checkpoint path")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--size", default="128x128", help="Training image size, e.g. 128x128")
    ap.add_argument("--patch", type=int, default=4, help="Encoder patch size (on stem output)")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--stem-width", type=int, default=256)
    ap.add_argument("--decoder-dim", type=int, default=256)
    ap.add_argument("--decoder-depth", type=int, default=2)
    ap.add_argument("--decoder-heads", type=int, default=8)
    args = ap.parse_args()

    try:
        h_s, w_s = str(args.size).lower().split("x")
        size = (int(h_s), int(w_s))
    except Exception as e:
        raise SystemExit(f"invalid --size={args.size!r}: {e}") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ImageFolderDataset(args.data_dir, size=size)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    enc_cfg = VisionEncoderConfig(
        patch_size=int(args.patch),
        embed_dim=int(args.embed_dim),
        depth=int(args.depth),
        num_heads=int(args.heads),
        output_dim=int(args.embed_dim),
        stem_width=int(args.stem_width),
    )
    encoder = MetaBonkVisionEncoder(enc_cfg).to(device)
    mae = MaskedAutoencoder(
        encoder,
        cfg=MAEConfig(
            mask_ratio=float(args.mask_ratio),
            decoder_dim=int(args.decoder_dim),
            decoder_depth=int(args.decoder_depth),
            decoder_heads=int(args.decoder_heads),
        ),
    ).to(device)

    opt = torch.optim.AdamW(mae.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    mae.train()
    for epoch in range(int(args.epochs)):
        loss_ema = None
        for x in dl:
            x = x.to(device, non_blocking=True)
            loss = mae(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lv = float(loss.detach().item())
            loss_ema = lv if loss_ema is None else (0.95 * float(loss_ema) + 0.05 * lv)
        print(f"[mae] epoch={epoch+1}/{int(args.epochs)} loss={loss_ema if loss_ema is not None else 0.0:.6f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "metabonk_vision_mae",
        "encoder_cfg": enc_cfg.__dict__,
        "mae_cfg": {
            "mask_ratio": float(args.mask_ratio),
            "decoder_dim": int(args.decoder_dim),
            "decoder_depth": int(args.decoder_depth),
            "decoder_heads": int(args.decoder_heads),
        },
        "encoder_state_dict": encoder.state_dict(),
    }
    torch.save(payload, out_path)
    print(f"[mae] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

