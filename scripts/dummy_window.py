#!/usr/bin/env python3
"""Minimal X11 window for validating worker isolation without launching the game."""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Spawn a simple X11 window for isolation tests.")
    ap.add_argument("--title", default="Megabonk Dummy")
    ap.add_argument("--class-name", default="megabonk-dummy")
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = run until closed.")
    args = ap.parse_args()

    try:
        import tkinter as tk
    except Exception as exc:
        print(f"[dummy_window] tkinter unavailable: {exc}", file=sys.stderr)
        return 1

    pid = os.getpid()
    display = os.environ.get("DISPLAY", "")

    root = tk.Tk()
    root.title(str(args.title))
    try:
        root.wm_class(str(args.class_name), str(args.class_name))
    except Exception:
        pass

    root.geometry("640x360")
    label = tk.Label(
        root,
        text=f"dummy window\npid={pid}\nDISPLAY={display}\n",
        font=("Helvetica", 14),
    )
    label.pack(expand=True, fill="both")

    print(f"[dummy_window] pid={pid} DISPLAY={display} title={args.title}", flush=True)

    if args.duration_s and args.duration_s > 0:
        root.after(int(args.duration_s * 1000), root.destroy)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
