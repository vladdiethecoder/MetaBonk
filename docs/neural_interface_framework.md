# Architecting the Neural Interface

This guide documents the production-ready path for Linux input injection and
Vision-Language menu reasoning in MetaBonk. It provides a runnable framework
for multi-instance agents with per-instance inputs and open-source VLM support.

## 1) Input Injection Options

### A. X11 per-instance input (recommended for multi-instance)
Uses `xdotool` per X11 display. With `MEGABONK_USE_XVFB=1`, each worker gets its
own Xvfb display, so input is isolated per instance.

Required env:

```
export METABONK_INPUT_BACKEND=xdotool
export MEGABONK_USE_XVFB=1
```

Optional (window targeting):

```
export METABONK_INPUT_XDO_WINDOW="gamescope"
export METABONK_INPUT_XDO_CLASS="gamescope"
```

### B. Kernel uinput (single-display or focused-window only)
Uses `/dev/uinput` for OS-level keyboard/mouse injection. Requires permissions
and is global to the host display (not per instance).

Run the setup script once:

```
./scripts/setup_uinput.sh
newgrp uinput
```

Then set:

```
export METABONK_INPUT_BACKEND=uinput
```

## 2) VLM Menu Reasoning (Open Source)

MetaBonk’s menu reasoning supports an Ollama backend.

Download a vision-capable model:

```
./scripts/setup_vlm_ollama.sh
```

Configure the worker to use it:

```
export METABONK_USE_SWITCHING_CONTROLLER=1
export METABONK_VLM_BACKEND=ollama
export METABONK_VLM_MODEL=llava:7b
```

Optional strobe/flash stabilization (EMA blend for menu frames):

```
export METABONK_MENU_FRAME_ALPHA=0.5
```

## 3) Vision Menu Classifier

If no trained menu classifier weights are provided, the vision service now
falls back to an OCR-based heuristic to determine menu vs gameplay. This is
enabled by default (no extra configuration required).

## 4) Multi-instance Launch (example)

```
export MEGABONK_USE_XVFB=1
export METABONK_INPUT_BACKEND=xdotool
export METABONK_USE_SWITCHING_CONTROLLER=1
export METABONK_VLM_BACKEND=ollama
export METABONK_VLM_MODEL=qwen2.5-vl:7b
export METABONK_INPUT_BUTTONS="W,A,S,D,SPACE,ENTER,ESC,LEFT,RIGHT,UP,DOWN"

./start --mode train --workers 4 --no-ui --game-dir "/path/to/Megabonk"
```

## 5) Notes

- Use `METABONK_MENU_GOAL` to change the menu objective (e.g. “Start Run”).
- For non-Xvfb setups, set `METABONK_INPUT_DISPLAY=:1` and `METABONK_INPUT_XAUTHORITY=/path/to/xauth`.
- `METABONK_MENU_LOG=1` enables menu actions and debug logs.
