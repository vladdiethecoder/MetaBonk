#!/usr/bin/env python3
"""Record on-policy (human) experience to .npz: (frames, real inputs).

This connects to the BonkLink BepInEx plugin and records episodes as `.npz`
files suitable for action-conditioned pixel world-model training:
  - observations: uint8 frames [T, H, W, 3]
  - actions: float32 vectors [T, A] from the plugin's input snapshot

Requirements (in-game plugin):
  - Enable JPEG capture (default true)
  - Enable input snapshot:
      [Capture]
      EnableInputSnapshot = true
  - If the game crashes on boot with `connection from unknown thread`, rebuild/reinstall BonkLink:
      python scripts/update_plugins.py --game-dir "/path/to/steamapps/common/Megabonk"

Then train the world model directly on these files:
  python scripts/train_generative_world_model.py --npz-dir <out_dir> --phase all
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError("PIL/Pillow is required (pip install pillow)") from e

# Ensure repo root on sys.path.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bridge.bonklink_client import BonkLinkClient  # type: ignore


def _decode_frame_bytes(frame: bytes) -> Optional[np.ndarray]:
    try:
        # Raw RGB frame from BonkLink: [MBRF][w:int32][h:int32][c:int32][payload bytes]
        if frame and len(frame) >= 16 and frame[:4] == b"MBRF":
            import struct

            w, h, c = struct.unpack_from("<3i", frame, 4)
            if w <= 0 or h <= 0:
                return None
            if c not in (3, 4):
                return None
            raw = frame[16:]
            need = w * h * c
            if len(raw) < need:
                return None
            arr = np.frombuffer(raw[:need], dtype=np.uint8).reshape((h, w, c))
            if c == 4:
                arr = arr[:, :, :3]
            return np.asarray(arr, dtype=np.uint8)

        import io

        im = Image.open(io.BytesIO(frame))
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def _action_vec_from_state(st) -> np.ndarray:
    # Fixed 10D action vector, all floats:
    #   move(x,y), look(x,y), fire, ability, interact, ui_click, click_nx, click_ny
    try:
        mx, my = getattr(st, "input_move", (0.0, 0.0)) or (0.0, 0.0)
    except Exception:
        mx, my = 0.0, 0.0
    try:
        lx, ly = getattr(st, "input_look", (0.0, 0.0)) or (0.0, 0.0)
    except Exception:
        lx, ly = 0.0, 0.0
    fire = 1.0 if bool(getattr(st, "input_fire", False)) else 0.0
    ability = 1.0 if bool(getattr(st, "input_ability", False)) else 0.0
    interact = 1.0 if bool(getattr(st, "input_interact", False)) else 0.0
    ui_click = 1.0 if bool(getattr(st, "input_ui_click", False)) else 0.0
    try:
        cnx, cny = getattr(st, "input_click_norm", (0.0, 0.0)) or (0.0, 0.0)
    except Exception:
        cnx, cny = 0.0, 0.0
    return np.asarray([mx, my, lx, ly, fire, ability, interact, ui_click, cnx, cny], dtype=np.float32)


def _save_episode(
    out_dir: Path,
    prefix: str,
    frames: List[np.ndarray],
    actions: List[np.ndarray],
    meta: dict,
    *,
    compress: bool,
) -> Optional[Path]:
    if not frames or not actions:
        return None
    T = min(len(frames), len(actions))
    if T <= 0:
        return None

    obs = np.stack(frames[:T], axis=0).astype(np.uint8, copy=False)
    act = np.stack(actions[:T], axis=0).astype(np.float32, copy=False)

    ts = int(time.time() * 1000)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{ts}.npz"

    payload = {
        "observations": obs,
        "actions": act,
        "meta": json.dumps(meta, separators=(",", ":"), sort_keys=True),
    }

    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Record (frames, real inputs) to .npz via BonkLink")
    ap.add_argument("--host", default=os.environ.get("METABONK_BONKLINK_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("METABONK_BONKLINK_PORT", "5555")))
    ap.add_argument("--out-dir", default=os.environ.get("METABONK_RECORD_DIR", "rollouts/onpolicy_npz"))
    ap.add_argument("--prefix", default="onpolicy")
    ap.add_argument("--hz", type=float, default=float(os.environ.get("METABONK_RECORD_HZ", "10")))
    ap.add_argument("--max-frames", type=int, default=int(os.environ.get("METABONK_RECORD_MAX_FRAMES", "6000")))
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--compress", action="store_true", default=True)
    ap.add_argument("--no-compress", action="store_false", dest="compress")
    ap.add_argument("--split-on-not-playing", action="store_true", default=True)
    ap.add_argument("--no-split-on-not-playing", action="store_false", dest="split_on_not_playing")
    ap.add_argument("--dedupe-frames", action="store_true", default=True)
    ap.add_argument("--no-dedupe-frames", action="store_false", dest="dedupe_frames")
    args = ap.parse_args()

    hz = max(0.5, float(args.hz))
    min_dt = 1.0 / hz

    client = BonkLinkClient(host=str(args.host), port=int(args.port))
    if not client.connect(timeout_s=3.0):
        print(f"[record] failed to connect to BonkLink at {args.host}:{args.port}")
        print("[record] is the game running with the BonkLink plugin enabled?")
        return 2

    out_dir = Path(args.out_dir)
    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    last_save = None
    last_frame_sig = None
    last_keep_ts = 0.0
    episode_idx = 0
    input_snapshot_seen = False

    def flush(reason: str):
        nonlocal frames, actions, last_save, episode_idx
        T = min(len(frames), len(actions))
        if T < int(args.min_frames):
            frames = []
            actions = []
            return
        meta = {
            "reason": reason,
            "episode_idx": int(episode_idx),
            "hz": float(hz),
            "action_format": "move(xy),look(xy),fire,ability,interact,ui_click,click_norm(xy)",
            "action_dim": 10,
            "input_snapshot_seen": bool(input_snapshot_seen),
            "note": "EnableInputSnapshot=true in BonkLink config for real inputs.",
        }
        p = _save_episode(out_dir, str(args.prefix), frames, actions, meta, compress=bool(args.compress))
        if p:
            print(f"[record] saved {p.name} (T={T}) reason={reason}")
            last_save = p
            episode_idx += 1
        frames = []
        actions = []

    print(f"[record] connected. recording to {out_dir} at ~{hz} Hz (Ctrl+C to stop)")

    try:
        while True:
            # Always drain the socket fast; only *store* at the requested Hz.
            pkt = client.read_state_frame(timeout_ms=50)
            if pkt is None:
                time.sleep(0.01)
                continue
            st, jpg = pkt

            if getattr(st, "input_ui_click", False) or getattr(st, "input_fire", False) or getattr(st, "input_ability", False) or getattr(st, "input_interact", False):
                input_snapshot_seen = True

            if not jpg:
                # No frames being sent; keep waiting.
                continue

            # Dedupe identical JPEGs (common when capture Hz < update Hz).
            if args.dedupe_frames:
                sig = (len(jpg), jpg[:32], jpg[-32:])
                if last_frame_sig == sig:
                    # Still update keep time to preserve action alignment timing a bit.
                    last_keep_ts = time.time()
                    continue
                last_frame_sig = sig

            # Subsample to target Hz.
            now = time.time()
            if (now - last_keep_ts) < min_dt:
                continue

            fr = _decode_frame_bytes(jpg)
            if fr is None or fr.ndim != 3 or fr.shape[2] != 3:
                continue

            frames.append(fr)
            actions.append(_action_vec_from_state(st))
            last_keep_ts = time.time()

            # Split/flush conditions.
            if int(args.max_frames) and len(frames) >= int(args.max_frames):
                flush("max_frames")
                continue

            if args.split_on_not_playing and not bool(getattr(st, "is_playing", True)):
                flush("not_playing")
                continue
    except KeyboardInterrupt:
        print("[record] Ctrl+C; flushing…")
        flush("interrupt")
        return 0
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
