#!/usr/bin/env python3
"""Watch a normal Steam session and train from visuals (storage-minimal).

This connects to the BonkLink BepInEx plugin running inside Megabonk and:
  - captures frames from your compositor via the ScreenCast portal (PipeWire)
  - runs vision (/predict) to build MetaBonk observations
  - pushes visual-only rollouts to the learner (/push_visual_rollout) to train the world model

No raw video is persisted; batches are applied online and discarded.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from typing import List, Optional, Tuple

import requests
from PIL import Image

from pathlib import Path

# Ensure repo root is on sys.path when executed as a script (so `import src.*` works).
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.worker.perception import construct_observation

try:
    import dbus  # type: ignore
    from dbus.mainloop.glib import DBusGMainLoop  # type: ignore
except Exception:  # pragma: no cover
    dbus = None  # type: ignore
    DBusGMainLoop = None  # type: ignore

try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst  # type: ignore
except Exception:  # pragma: no cover
    GLib = None  # type: ignore
    Gst = None  # type: ignore


def _acquire_singleton_lock(repo_root: Path) -> Optional[Path]:
    """Prevent multiple concurrent watchers (which can overwhelm the game/plugin)."""
    lock_path = repo_root / "temp" / "watch_visual.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        finally:
            os.close(fd)
        return lock_path
    except FileExistsError:
        # Best-effort: if the lock is stale, remove it.
        try:
            txt = lock_path.read_text().strip()
            pid = int(txt) if txt else 0
        except Exception:
            pid = 0
        if pid > 1:
            try:
                os.kill(pid, 0)
                # Still running.
                return None
            except Exception:
                pass
        try:
            lock_path.unlink()
        except Exception:
            return None
        # Retry once.
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            return lock_path
        except Exception:
            return None


def _jpeg_size(jpg: bytes) -> Optional[Tuple[int, int]]:
    try:
        im = Image.open(io.BytesIO(jpg))
        return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def _portal_request(
    bus: "dbus.SessionBus",
    iface: "dbus.Interface",
    method: str,
    *args,
    timeout_s: float = 30.0,
    options: Optional[dict] = None,
) -> dict:
    """Call an xdg-desktop-portal method and wait for its Request.Response."""
    if dbus is None or DBusGMainLoop is None or GLib is None:
        raise RuntimeError("dbus/GLib not available (required for portal capture)")

    token = f"metabonk{os.getpid()}{int(time.time() * 1000)}"
    opts = dict(options or {})
    opts["handle_token"] = token
    # dbus requires signature 'sv' dictionary
    opts = dbus.Dictionary(opts, signature="sv")

    req_path = getattr(iface, method)(*args, opts)
    req_obj = bus.get_object("org.freedesktop.portal.Desktop", req_path)
    req = dbus.Interface(req_obj, "org.freedesktop.portal.Request")

    out: dict = {}
    loop = GLib.MainLoop()

    def _on_response(response: int, results: "dbus.Dictionary"):  # type: ignore[name-defined]
        out["response"] = int(response)
        try:
            out["results"] = {str(k): results[k] for k in results.keys()}
        except Exception:
            out["results"] = {}
        try:
            loop.quit()
        except Exception:
            pass

    req.connect_to_signal("Response", _on_response)

    def _timeout():
        out["response"] = 2
        out["results"] = {}
        try:
            loop.quit()
        except Exception:
            pass
        return False

    GLib.timeout_add(int(max(1, timeout_s) * 1000), _timeout)
    loop.run()
    return out


def _portal_open_screencast_fd(*, types_mask: int = 2) -> Tuple[int, int, callable]:
    """Open a ScreenCast portal session and return (pw_fd, node_id, cleanup_fn)."""
    if dbus is None or DBusGMainLoop is None or GLib is None:
        raise RuntimeError("dbus/GLib not available (required for portal capture)")

    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    portal = bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
    sc = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")

    # 1) Create session
    r1 = _portal_request(bus, sc, "CreateSession", timeout_s=30.0, options={"session_handle_token": f"s{os.getpid()}"})
    if int(r1.get("response", 2)) != 0:
        raise RuntimeError("portal CreateSession denied/canceled")
    session_handle = str((r1.get("results") or {}).get("session_handle") or "")
    if not session_handle:
        raise RuntimeError("portal CreateSession missing session_handle")

    # 2) Select sources (default: window capture)
    r2 = _portal_request(
        bus,
        sc,
        "SelectSources",
        session_handle,
        timeout_s=60.0,
        options={
            "types": dbus.UInt32(int(types_mask)),
            "multiple": dbus.Boolean(False),
            # ScreenCast portal cursor mode is a bitmask; 0 is invalid on some portals.
            # 2 ("embedded") is widely supported (see AvailableCursorModes property).
            "cursor_mode": dbus.UInt32(2),
        },
    )
    if int(r2.get("response", 2)) != 0:
        raise RuntimeError("portal SelectSources denied/canceled")

    # 3) Start (this is where the user picks the window/monitor)
    r3 = _portal_request(
        bus,
        sc,
        "Start",
        session_handle,
        "",  # parent_window (empty is acceptable for CLI tools)
        timeout_s=120.0,
        options={},
    )
    if int(r3.get("response", 2)) != 0:
        raise RuntimeError("portal Start denied/canceled")
    streams = (r3.get("results") or {}).get("streams") or []
    node_id = None
    try:
        # streams: array of (u node_id, a{sv} props)
        if streams:
            node_id = int(streams[0][0])
    except Exception:
        node_id = None
    if not node_id:
        raise RuntimeError("portal Start returned no streams/node_id")

    # 4) Open PipeWire remote FD
    try:
        fd_obj = sc.OpenPipeWireRemote(session_handle, dbus.Dictionary({}, signature="sv"))
        try:
            pw_fd = int(fd_obj.take())  # type: ignore[attr-defined]
        except Exception:
            pw_fd = int(fd_obj)
    except Exception as e:
        raise RuntimeError(f"portal OpenPipeWireRemote failed: {e}")

    # Cleanup: close session
    def _cleanup():
        try:
            sess_obj = bus.get_object("org.freedesktop.portal.Desktop", session_handle)
            sess = dbus.Interface(sess_obj, "org.freedesktop.portal.Session")
            sess.Close()
        except Exception:
            pass
        try:
            os.close(int(pw_fd))
        except Exception:
            pass

    return int(pw_fd), int(node_id), _cleanup


def _gst_pull_rgb_frames(*, pw_fd: int, node_id: int, width: int, height: int, hz: float):
    """Yield RGB frames (bytes, (w,h)) from a PipeWire portal stream."""
    if Gst is None:
        raise RuntimeError("GStreamer GI bindings not available")
    Gst.init(None)

    target_caps = f"video/x-raw,format=RGB,width={int(width)},height={int(height)}"

    # Prefer target-object, fall back to deprecated path if needed.
    pipelines = [
        f"pipewiresrc fd={int(pw_fd)} target-object={int(node_id)} do-timestamp=true ! "
        f"videoconvert ! videoscale ! {target_caps} ! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true",
        f"pipewiresrc fd={int(pw_fd)} path={int(node_id)} do-timestamp=true ! "
        f"videoconvert ! videoscale ! {target_caps} ! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true",
    ]

    pipeline = None
    sink = None
    last_err = None
    for pstr in pipelines:
        try:
            pipeline = Gst.parse_launch(pstr)
            sink = pipeline.get_by_name("sink")
            pipeline.set_state(Gst.State.PLAYING)
            # Quick sanity pull.
            sample = sink.emit("try-pull-sample", int(1e9 * 2))  # 2s
            if sample is None:
                raise RuntimeError("no frames (portal stream not producing)")
            # Put sample back in the loop by yielding it first.
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w = int(caps.get_value("width"))
            h = int(caps.get_value("height"))
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("failed to map sample buffer")
            data0 = bytes(mapinfo.data)
            buf.unmap(mapinfo)
            yield data0, (w, h)
            break
        except Exception as e:
            last_err = str(e)
            try:
                if pipeline:
                    pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            pipeline = None
            sink = None
            continue

    if pipeline is None or sink is None:
        raise RuntimeError(f"failed to start PipeWire capture pipeline: {last_err or 'unknown error'}")

    min_dt = 1.0 / max(1.0, float(hz))
    try:
        while True:
            t0 = time.time()
            sample = sink.emit("try-pull-sample", int(1e9 * 2))
            if sample is None:
                # portal stream can stall when window is minimized; keep trying
                time.sleep(0.05)
                continue
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w = int(caps.get_value("width"))
            h = int(caps.get_value("height"))
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            data = bytes(mapinfo.data)
            buf.unmap(mapinfo)
            yield data, (w, h)
            dt = time.time() - t0
            if dt < min_dt:
                time.sleep(min_dt - dt)
    finally:
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="MetaBonk: watch me play (online demo training)")
    ap.add_argument("--capture", choices=["portal"], default="portal")
    ap.add_argument("--portal-types", choices=["window", "monitor"], default="window")
    ap.add_argument("--portal-width", type=int, default=int(os.environ.get("METABONK_WATCH_W", "640")))
    ap.add_argument("--portal-height", type=int, default=int(os.environ.get("METABONK_WATCH_H", "360")))
    ap.add_argument("--vision-url", default=os.environ.get("VISION_URL", "http://127.0.0.1:8050"))
    ap.add_argument("--learner-url", default=os.environ.get("LEARNER_URL", "http://127.0.0.1:8061"))
    ap.add_argument("--policy-name", default=os.environ.get("POLICY_NAME", "Greed"))
    ap.add_argument("--obs-dim", type=int, default=int(os.environ.get("METABONK_DEFAULT_OBS_DIM", "204")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("METABONK_DEMO_BATCH", "32")))
    ap.add_argument("--max-elements", type=int, default=32)
    ap.add_argument("--hz", type=float, default=float(os.environ.get("METABONK_DEMO_HZ", "10")))
    args = ap.parse_args()

    lock = _acquire_singleton_lock(_repo_root)
    if lock is None:
        raise SystemExit("another watch_visual instance is already running (lock: temp/watch_visual.lock)")

    vision_url = str(args.vision_url).rstrip("/")
    learner_url = str(args.learner_url).rstrip("/")

    sess = requests.Session()
    sess.headers.update({"User-Agent": "metabonk-watch-play"})

    obs_buf: List[List[float]] = []

    min_dt = 1.0 / max(1.0, float(args.hz))
    types_mask = 2 if str(args.portal_types) == "window" else 1
    print(f"[watch] starting portal capture ({args.portal_types}); select the Megabonk window when prompted…")

    pw_fd = None
    node_id = None
    portal_cleanup = None
    try:
        pw_fd, node_id, portal_cleanup = _portal_open_screencast_fd(types_mask=int(types_mask))
        print(f"[watch] portal stream acquired (node_id={node_id}); pushing visual rollouts to {learner_url}/push_visual_rollout")

        for rgb, frame_size in _gst_pull_rgb_frames(
            pw_fd=int(pw_fd),
            node_id=int(node_id),
            width=int(args.portal_width),
            height=int(args.portal_height),
            hz=float(args.hz),
        ):
            # Encode to JPEG for the vision service (small and simple).
            try:
                w, h = frame_size
                im = Image.frombytes("RGB", (w, h), rgb)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80, optimize=False)
                jpg = buf.getvalue()
            except Exception:
                continue

            img_b64 = base64.b64encode(jpg).decode("ascii")
            try:
                r = sess.post(f"{vision_url}/predict", json={"image_b64": img_b64}, timeout=(1.0, 2.0))
                if not r.ok:
                    continue
                payload = r.json()
                dets = payload.get("detections") or []
            except Exception:
                continue

            obs, action_mask = construct_observation(
                dets,
                obs_dim=int(args.obs_dim),
                frame_size=frame_size,
                max_elements=int(args.max_elements),
            )

            obs_buf.append(obs)

            if len(obs_buf) >= int(args.batch):
                try:
                    rr = sess.post(
                        f"{learner_url}/push_visual_rollout",
                        json={
                            "policy_name": str(args.policy_name),
                            "obs": obs_buf,
                        },
                        timeout=(1.0, 3.0),
                    )
                    if rr.ok:
                        m = rr.json().get("losses") or {}
                        print(
                            f"[watch] demo batch {len(obs_buf)}: "
                            f"wm_total={m.get('wm_total', '—')} wm_recon={m.get('wm_recon', '—')} wm_kl={m.get('wm_kl', '—')}"
                        )
                except Exception:
                    pass
                obs_buf.clear()
    except KeyboardInterrupt:
        print("[watch] Ctrl+C; stopping…")
        return 0
    finally:
        try:
            if portal_cleanup:
                portal_cleanup()
        except Exception:
            pass
        try:
            if lock and lock.exists():
                lock.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
