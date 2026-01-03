#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_rgb(path: Path) -> "Any":
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Requires Pillow + numpy") from e

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _norm_text(s: str) -> str:
    return " ".join(str(s or "").split())


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _find_confirm(hints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_conf = -1.0
    for h in hints:
        text = _norm_text(h.get("text", "")).lower()
        if "confirm" not in text:
            continue
        conf = _as_float(h.get("confidence", 0.0), 0.0)
        if conf > best_conf:
            best_conf = conf
            best = h
    return best


def _cv_fallback_hints(frame: "Any") -> List[Dict[str, Any]]:
    """Best-effort CV/OCR fallback when VLM is unavailable."""
    from src.worker.vlm_hint_generator import detect_ui_patterns, merge_hints

    dets = detect_ui_patterns(frame, include_ocr=True)

    h, w = int(frame.shape[0]), int(frame.shape[1])
    out: List[Dict[str, Any]] = []

    for d in dets:
        if str(d.get("type") or "").strip().lower() != "ocr_text":
            continue
        bb = d.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        txt = _norm_text(d.get("text") or "")
        if not txt:
            continue
        conf = _as_float(d.get("confidence", 0.0), 0.0)
        pri = 1 if "confirm" in txt.lower() else 2
        out.append(
            {
                "element_type": "button" if pri == 1 else "text_prompt",
                "text": txt,
                "location": {"x": int(max(0, min(w - 1, cx))), "y": int(max(0, min(h - 1, cy)))},
                "priority": int(pri),
                "reasoning": "OCR detected text",
                "confidence": float(max(0.0, min(1.0, conf))),
                "source": "cv_ocr",
            }
        )

    # Add non-OCR shape candidates as lower-priority hints.
    try:
        shapes = [d for d in dets if str(d.get("type") or "").strip().lower() != "ocr_text"]
        out = merge_hints(out, shapes)
    except Exception:
        pass
    return list(out)


def _run_one(
    path: Path,
    *,
    context: Dict[str, Any],
    min_conf: float,
    json_out: bool,
    require_confirm: bool,
) -> Tuple[bool, List[Dict[str, Any]]]:
    frame = _load_rgb(path)
    hints: List[Dict[str, Any]] = []
    used_fallback = False

    try:
        from src.worker.vlm_hint_generator import generate_hints

        hints = list(generate_hints(frame, context=context) or [])
    except Exception as e:
        used_fallback = True
        print(f"[WARN] {path}: VLM hint generation failed ({e}); using CV/OCR fallback", file=sys.stderr)
        try:
            hints = _cv_fallback_hints(frame)
        except Exception as e2:
            print(f"[ERROR] {path}: CV/OCR fallback failed ({e2})", file=sys.stderr)
            hints = []

    # Sort: priority asc, confidence desc.
    hints_sorted = sorted(
        hints,
        key=lambda h: (
            _as_int(h.get("priority", 2), 2),
            -_as_float(h.get("confidence", 0.0), 0.0),
            str(h.get("element_type") or ""),
        ),
    )

    confirm = _find_confirm(hints_sorted)
    ok = True
    if require_confirm:
        ok = confirm is not None and _as_float(confirm.get("confidence", 0.0), 0.0) >= float(min_conf)

    if json_out:
        payload = {
            "image": str(path),
            "used_fallback": bool(used_fallback),
            "confirm": confirm,
            "hints": hints_sorted,
            "pass": bool(ok),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"== {path} ==")
        if confirm is not None:
            c_text = _norm_text(confirm.get("text", ""))
            c_loc = confirm.get("location") or {}
            print(
                f"confirm: text={c_text!r} conf={_as_float(confirm.get('confidence'), 0.0):.2f} "
                f"loc=({c_loc.get('x')},{c_loc.get('y')}) pri={_as_int(confirm.get('priority', 2), 2)}"
            )
        else:
            print("confirm: NOT FOUND")

        print("top hints:")
        for h in hints_sorted[:10]:
            loc = h.get("location") or {}
            print(
                f"  - pri={_as_int(h.get('priority', 2), 2)} conf={_as_float(h.get('confidence', 0.0), 0.0):.2f} "
                f"type={str(h.get('element_type') or ''):>16} text={_norm_text(h.get('text',''))!r} "
                f"xy=({loc.get('x')},{loc.get('y')}) src={h.get('source','')}"
            )

        if require_confirm:
            if ok:
                print(f"RESULT: PASS (confirm >= {min_conf:.2f})")
            else:
                print(f"RESULT: FAIL (expected confirm >= {min_conf:.2f})")
        else:
            print("RESULT: OK")

    return (bool(ok), hints_sorted)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate game-agnostic VLM UI hint detection.")
    ap.add_argument("images", nargs="+", help="Input image paths (png/jpg).")
    ap.add_argument("--min-confidence", type=float, default=0.60, help="Min confidence required for CONFIRM detection.")
    ap.add_argument(
        "--no-require-confirm",
        action="store_true",
        help="Do not fail if CONFIRM is not detected (still prints hints).",
    )
    ap.add_argument("--json", action="store_true", help="Output JSON per image.")
    ap.add_argument(
        "--context-state",
        default="menu",
        help="Context state string passed to hint generator (default: menu).",
    )

    args = ap.parse_args()
    require_confirm = not bool(args.no_require_confirm)

    ok_all = True
    for p in args.images:
        path = Path(p)
        if not path.exists():
            print(f"[ERROR] missing image: {path}", file=sys.stderr)
            ok_all = False
            continue
        ok, _hints = _run_one(
            path,
            context={"state": str(args.context_state)},
            min_conf=float(args.min_confidence),
            json_out=bool(args.json),
            require_confirm=bool(require_confirm),
        )
        ok_all = ok_all and bool(ok)

    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())

