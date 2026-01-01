"""YOLO Vision service for MegaBonk.

FastAPI microservice that performs object detection on frames and returns
structured detections for workers. This recovery implementation supports:

- `image_b64`: base64‑encoded RGB/PNG/JPEG image.
- `shm_name` + `width`/`height`: POSIX shared‑memory RGB buffer (uint8).

Zero‑copy DMABuf inference is implemented on the worker side; for now the
vision service expects CPU‑addressable images.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
import uvicorn

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

try:
    from multiprocessing import shared_memory
except Exception:  # pragma: no cover
    shared_memory = None  # type: ignore

from src.common.device import resolve_device
from src.common.schemas import PredictRequest, PredictResponse


app = FastAPI(title="MetaBonk Vision Service")

_model: Optional[Any] = None
_device: Optional[str] = None


def _load_model(weights: str) -> Any:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    return YOLO(weights)


def _decode_b64_image(image_b64: str) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow not installed")
    # Strip data URL prefix if present.
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _read_shm_rgb(name: str, width: int, height: int) -> "Image.Image":
    if shared_memory is None or np is None or Image is None:
        raise RuntimeError("shared_memory/numpy/Pillow not installed")
    shm = shared_memory.SharedMemory(name=name)
    try:
        expected = width * height * 3
        buf = shm.buf[:expected]
        arr = np.ndarray((height, width, 3), dtype=np.uint8, buffer=buf)
        return Image.fromarray(arr, mode="RGB")
    finally:
        shm.close()


def _run_yolo(frame: "Image.Image") -> Tuple[List[Dict[str, Any]], float]:
    assert _model is not None
    start = time.perf_counter()
    dev = (_device or "").strip() or None
    # Force CPU when requested (helps avoid VRAM pressure that can crash a running game).
    if dev:
        results = _model.predict(frame, verbose=False, device=dev)
    else:
        results = _model.predict(frame, verbose=False)
    dets: List[Dict[str, Any]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            try:
                cls_id = int(b.cls.item()) if hasattr(b, "cls") else -1
                conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                xyxy = (
                    b.xyxy.cpu().tolist()[0]
                    if hasattr(b, "xyxy") and b.xyxy is not None
                    else None
                )
                dets.append({"cls": cls_id, "conf": conf, "xyxy": xyxy})
            except Exception:
                continue
    latency_ms = (time.perf_counter() - start) * 1000.0
    return dets, latency_ms


@app.get("/")
async def read_root():
    return {"message": "MetaBonk Vision Service is running"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if req.image_b64:
        try:
            frame = _decode_b64_image(req.image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image_b64: {e}") from e
    elif req.shm_name and req.width and req.height:
        try:
            frame = _read_shm_rgb(req.shm_name, req.width, req.height)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid shm image: {e}") from e
    else:
        raise HTTPException(status_code=400, detail="Provide image_b64 or shm_name+width+height")

    dets, latency_ms = _run_yolo(frame)
    # Recovery vision service returns detections only. Downstream code may also
    # consume a `metrics` dict when a richer visual model is used.
    return PredictResponse(detections=dets, latency_ms=latency_ms, metrics={})


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaBonk vision service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default=os.environ.get("METABONK_VISION_DEVICE", ""))
    args = parser.parse_args()

    global _model
    global _device
    _device = resolve_device(str(args.device or "").strip(), context="vision")
    _model = _load_model(args.weights)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
