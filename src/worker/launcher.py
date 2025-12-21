"""Gamescope/game + save-data launcher.

This module is responsible for two things:
  1) Launch-time environment setup for MegaBonk under Gamescope/Sidecar.
  2) Selecting and wiring persistent save data into each instance.

Post‑mortem (Dec 2025) fix:
Megabonk stores critical JSON saves directly in the Steam Cloud ID folder on
native Linux, not under a nested ``Saves/`` directory. The scorer now validates
both layouts by explicitly checking for ``progression.json`` and ``stats.json``
either in ``<id>/Saves/`` or at ``<id>/`` root.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SaveCandidate:
    name: str
    path: Path
    score: int
    mtime: float


CRITICAL_FILES: Tuple[str, str] = ("progression.json", "stats.json")
DEFAULT_SAVE_ROOT = Path("~/.config/unity3d/Ved/Megabonk/Saves/CloudDir").expanduser()

def _parse_pw_cli_list(text: str) -> list[dict]:
    out: list[dict] = []
    cur: Optional[dict] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("id "):
            if cur:
                out.append(cur)
            m = re.match(r"id\\s+(\\d+)(?:,\\s*type\\s+([^\\s]+))?", line)
            if not m:
                cur = None
                continue
            type_raw = (m.group(2) or "").split("/")[0]
            cur = {"id": int(m.group(1)), "type": type_raw, "props": {}}
            continue
        if cur is None or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().rstrip(",")
        if val.startswith("\"") and val.endswith("\""):
            val = val[1:-1]
        cur["props"][key] = val
    if cur:
        out.append(cur)
    return out


def _pw_cli_objects(kind: str) -> list[dict]:
    if not shutil.which("pw-cli"):
        return []
    try:
        out = subprocess.check_output(["pw-cli", "ls", kind], stderr=subprocess.DEVNULL, timeout=1.5)
    except Exception:
        return []
    parsed = _parse_pw_cli_list(out.decode("utf-8", "replace"))
    objs: list[dict] = []
    for entry in parsed:
        typ = entry.get("type")
        if not typ:
            if kind.lower() == "node":
                typ = "PipeWire:Interface:Node"
            elif kind.lower() == "port":
                typ = "PipeWire:Interface:Port"
            else:
                typ = f"PipeWire:Interface:{kind}"
        objs.append({"id": entry.get("id"), "type": typ, "info": {"props": entry.get("props") or {}}})
    return objs


def _pipewire_objects() -> list[dict]:
    if shutil.which("pw-dump"):
        try:
            out = subprocess.check_output(["pw-dump"], stderr=subprocess.DEVNULL, timeout=1.0)
            data = json.loads(out.decode("utf-8", "replace"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    nodes = _pw_cli_objects("Node")
    ports = _pw_cli_objects("Port")
    return nodes + ports


def _find_gamescope_capture_target() -> Optional[str]:
    data = _pipewire_objects()
    if not data:
        return None
    candidates: list[str] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "") != "PipeWire:Interface:Port":
            continue
        props = (obj.get("info") or {}).get("props") or {}
        if not isinstance(props, dict):
            continue
        for key in ("object.path", "port.alias"):
            val = props.get(key)
            if isinstance(val, str) and val.startswith("gamescope:capture_"):
                candidates.append(val)
    if not candidates:
        return None
    # Stable ordering: capture_0 first, then higher indices.
    def _idx(name: str) -> int:
        m = re.search(r"capture_(\\d+)", name)
        return int(m.group(1)) if m else 0
    candidates = sorted(set(candidates), key=_idx)
    pref_idx = os.environ.get("METABONK_PIPEWIRE_CAPTURE_INDEX")
    if pref_idx and pref_idx.isdigit():
        want = f"gamescope:capture_{int(pref_idx)}"
        if want in candidates:
            return want
    return candidates[0]


def _valid_json_file(p: Path) -> bool:
    """Cheap sanity check: file exists, non‑empty, looks like JSON."""
    try:
        if not p.is_file() or p.stat().st_size == 0:
            return False
        with p.open("rb") as f:
            head = f.read(50).lstrip()
        return head.startswith(b"{")
    except Exception:
        return False


def _last_mtime(dir_path: Path) -> float:
    """Latest mtime of any critical file under dir (depth<=2)."""
    latest = 0.0
    for fn in CRITICAL_FILES:
        for p in dir_path.rglob(fn):
            try:
                rel_parts = p.relative_to(dir_path).parts
                if len(rel_parts) <= 3:  # depth <=2 plus filename
                    latest = max(latest, p.stat().st_mtime)
            except Exception:
                continue
    return latest


def score_save_directory(target_path: Path) -> int:
    """Score a candidate save directory.

    Supports two layouts:
      - Windows/legacy: <id>/Saves/*.json
      - Linux/native:  <id>/{progression.json,stats.json}
    """
    score = 0

    nested = target_path / "Saves"
    if nested.is_dir():
        json_files = list(nested.glob("*.json"))
        score += len(json_files) * 10
        for fn, weight in (("progression.json", 100), ("stats.json", 50)):
            if _valid_json_file(nested / fn):
                score += weight
    else:
        for fn, weight in (("progression.json", 100), ("stats.json", 50)):
            if _valid_json_file(target_path / fn):
                score += weight

        # Future‑proofing: limited depth search for non‑standard layouts.
        if score == 0:
            found: Dict[str, Path] = {}
            for fn in CRITICAL_FILES:
                for p in target_path.rglob(fn):
                    try:
                        if len(p.relative_to(target_path).parts) <= 3 and _valid_json_file(p):
                            found[fn] = p
                            break
                    except Exception:
                        continue
            if "progression.json" in found:
                score += 100
            if "stats.json" in found:
                score += 50

    return score


def select_save_candidate(
    save_root: Optional[str] = None,
    save_override: Optional[str] = None,
) -> Optional[SaveCandidate]:
    """Find the best save directory."""
    candidates: List[SaveCandidate] = []

    if save_override:
        ov = Path(save_override).expanduser()
        if ov.is_file():
            ov = ov.parent
        if ov.is_dir():
            sc = score_save_directory(ov)
            candidates.append(
                SaveCandidate(name=f"override_{ov.name}", path=ov, score=sc, mtime=_last_mtime(ov))
            )

    root = Path(save_root).expanduser() if save_root else DEFAULT_SAVE_ROOT
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            sc = score_save_directory(child)
            if sc > 0:
                candidates.append(
                    SaveCandidate(name=child.name, path=child, score=sc, mtime=_last_mtime(child))
                )

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c.score, c.mtime), reverse=True)
    return candidates[0]


def ensure_symlink(src: Path, dest: Path) -> None:
    """Create/replace dest -> src symlink safely."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        # Only replace if directory is empty (avoid destroying data).
        try:
            if dest.is_dir() and not any(dest.iterdir()):
                shutil.rmtree(dest)
            else:
                return
        except Exception:
            return
    dest.symlink_to(src)


@dataclass
class GameLauncher:
    instance_id: str
    display: Optional[str] = None
    use_gamescope: bool = True
    sidecar_port: Optional[int] = None
    selected_save: Optional[SaveCandidate] = None
    proc: Optional[Any] = None
    log_path: Optional[Path] = None

    def launch(self) -> None:
        # Real implementation execs gamescope + game binary; recovery mode uses
        # an env-configurable command so the launcher is still useful in dev.
        self.use_gamescope = os.environ.get("MEGABONK_USE_GAMESCOPE", "1") == "1"
        if self.sidecar_port is None:
            sp = os.environ.get("MEGABONK_SIDECAR_PORT")
            self.sidecar_port = int(sp) if sp else None

        # Save selection / wiring.
        save_root = os.environ.get("MEGABONK_SAVE_ROOT")
        save_override = os.environ.get("MEGABONK_SAVE_OVERRIDE")
        self.selected_save = select_save_candidate(save_root=save_root, save_override=save_override)

        if os.environ.get("MEGABONK_LOG_SAVES", "0") == "1":
            if self.selected_save:
                print(
                    f"[launcher] save candidate selected: {self.selected_save.name} "
                    f"{self.selected_save.path} (score={self.selected_save.score})"
                )
            else:
                print("[launcher] no valid save candidates found")

        dest = os.environ.get("MEGABONK_SAVE_DEST")
        if self.selected_save and dest:
            try:
                ensure_symlink(self.selected_save.path, Path(dest).expanduser())
            except Exception:
                pass

        # Optional process launch.
        cmd = os.environ.get("MEGABONK_CMD") or os.environ.get("MEGABONK_COMMAND")
        if not cmd:
            tmpl = os.environ.get("MEGABONK_CMD_TEMPLATE") or os.environ.get("MEGABONK_COMMAND_TEMPLATE")
            if tmpl:
                try:
                    cmd = str(tmpl).format(
                        instance_id=self.instance_id,
                        sidecar_port=self.sidecar_port if self.sidecar_port is not None else "",
                        display=self.display or "",
                        worker_port=os.environ.get("WORKER_PORT") or os.environ.get("MEGABONK_WORKER_PORT") or "",
                        bonklink_host=os.environ.get("METABONK_BONKLINK_HOST") or "",
                        bonklink_port=os.environ.get("METABONK_BONKLINK_PORT") or "",
                    )
                except Exception:
                    cmd = str(tmpl)
        if not cmd:
            return

        try:
            args = shlex.split(cmd)
        except Exception:
            args = [cmd]

        env = os.environ.copy()
        if self.display:
            env["DISPLAY"] = self.display
        if self.sidecar_port is not None:
            env.setdefault("MEGABONK_SIDECAR_PORT", str(self.sidecar_port))

        # Optional logging for discovery (e.g., parsing Gamescope PipeWire node IDs).
        log_dir = env.get("MEGABONK_LOG_DIR") or ""
        out_target = subprocess.DEVNULL
        err_target = subprocess.DEVNULL
        if log_dir:
            try:
                p = Path(log_dir).expanduser()
                p.mkdir(parents=True, exist_ok=True)
                self.log_path = p / f"{self.instance_id}.log"
                # Open in append mode so restarts don't lose the last pipewire line.
                f = open(self.log_path, "ab", buffering=0)
                out_target = f
                err_target = f
                env["MEGABONK_LOG_PATH"] = str(self.log_path)
            except Exception:
                self.log_path = None

        # If a "job" process group is specified, bind the spawned game process to it
        # so a single killpg() from the top-level launcher can reliably stop *everything*
        # (even if the worker is SIGKILL'd).
        job_pgid = None
        try:
            jp = env.get("METABONK_JOB_PGID")
            if jp:
                job_pgid = int(jp)
        except Exception:
            job_pgid = None

        preexec_fn = None
        if os.name == "posix" and job_pgid:
            def _bind_to_job_pgid():  # type: ignore[no-redef]
                try:
                    os.setpgid(0, int(job_pgid))
                except Exception:
                    pass

            preexec_fn = _bind_to_job_pgid

        # Avoid spawning multiple copies.
        if self.proc and getattr(self.proc, "poll", lambda: None)() is None:
            return

        try:
            self.proc = subprocess.Popen(
                args,
                env=env,
                stdout=out_target,
                stderr=err_target,
                preexec_fn=preexec_fn,
            )
        except Exception:
            self.proc = None

    def discover_pipewire_node(self, timeout_s: float = 4.0, poll_s: float = 0.2) -> Optional[str]:
        """Best-effort: discover a PipeWire video source for the launched process (Gamescope).

        On modern PipeWire, gamescope exposes a `Video/Source` node and a set of
        output ports. The most reliable selector for GStreamer's `pipewiresrc` is
        the *port serial* (unique per instance), not the node id or a shared name.

        Returns a string suitable for:
          - GStreamer: `pipewiresrc target-object=<value>`
        """
        if not self.proc or getattr(self.proc, "poll", lambda: 0)() is not None:
            return None

        # Preferred path: use PipeWire metadata to find the node/port created by this gamescope PID.
        try:
            pid = int(getattr(self.proc, "pid", 0) or 0)
        except Exception:
            pid = 0
        if pid > 1:
            try:
                target = self._resolve_pipewire_target_object_for_pid(pid, timeout_s=timeout_s, poll_s=poll_s)
            except Exception:
                target = None
            if target:
                return target

        # Gamescope often logs the PipeWire node id; prefer parsing that (it is reliable
        # even when PipeWire node props do not include application PID).
        log_path = self.log_path
        if log_path and log_path.exists():
            deadline = time.time() + max(0.1, float(timeout_s))
            pat = re.compile(rb"stream available on node ID:\\s*([0-9]+)")
            last_id: Optional[str] = None
            while time.time() < deadline:
                try:
                    # Read a generous tail; the gamescope node line can be far above recent logs.
                    tail_bytes = int(os.environ.get("METABONK_PIPEWIRE_LOG_TAIL_BYTES", "2000000"))
                    data = log_path.read_bytes()
                    if tail_bytes > 0 and len(data) > tail_bytes:
                        data = data[-tail_bytes:]
                    m = pat.findall(data)
                    if m:
                        last_id = m[-1].decode("ascii", "ignore")
                        if last_id:
                            try:
                                node_id = int(last_id)
                            except Exception:
                                node_id = None
                            if node_id is not None:
                                # Give PipeWire a moment to publish node/port metadata after the log line.
                                target = self._resolve_pipewire_target_object(node_id, timeout_s=min(1.0, float(timeout_s)))
                                if target:
                                    return target
                                # Fall back to the node id itself; many `pipewiresrc` builds accept ids even
                                # when serial/path lookups are incomplete.
                                return str(int(node_id))
                except Exception:
                    pass
                time.sleep(max(0.05, float(poll_s)))

        # Last resort: no log path available (or port serial lookup failed).
        fallback = _find_gamescope_capture_target()
        if fallback:
            return fallback
        return None

    @staticmethod
    def pipewire_target_exists(target_object: str, timeout_s: float = 0.0) -> bool:
        """Return True if a PipeWire object matching `target_object` exists.

        `target_object` is intended to be passed to `pipewiresrc target-object=...`.
        In our setup, this is usually a numeric `object.serial` (node serial preferred).
        """
        s = str(target_object or "").strip()
        if not s:
            return False
        deadline = time.time() + max(0.0, float(timeout_s))
        want_serial = None
        try:
            if s.isdigit():
                want_serial = int(s)
        except Exception:
            want_serial = None
        while True:
            data = _pipewire_objects()
            try:
                for obj in data:
                    if not isinstance(obj, dict):
                        continue
                    typ = str(obj.get("type") or "")
                    if typ not in ("PipeWire:Interface:Port", "PipeWire:Interface:Node"):
                        continue
                    props = (obj.get("info") or {}).get("props") or {}
                    if not isinstance(props, dict):
                        continue
                    if want_serial is not None:
                        try:
                            if int(props.get("object.serial")) == want_serial:
                                return True
                        except Exception:
                            pass
                        # Some callers might pass the numeric object id instead of serial.
                        try:
                            if int(obj.get("id")) == want_serial:
                                return True
                        except Exception:
                            pass
                    op = props.get("object.path")
                    if isinstance(op, str) and op == s:
                        return True
                    name = props.get("node.name")
                    if isinstance(name, str) and name == s:
                        return True
            except Exception:
                pass
            if time.time() >= deadline:
                break
            time.sleep(0.05)
        return False

    @staticmethod
    def _resolve_pipewire_target_object(node_id: int, timeout_s: float = 2.0, poll_s: float = 0.1) -> Optional[str]:
        """Resolve gamescope's PipeWire node id -> unique `pipewiresrc target-object`.

        gamescope nodes typically share `node.name="gamescope"` and do not expose
        `object.path` at the node level. `pipewiresrc target-object` accepts an
        object name/id/serial. By default we prefer the node id (most widely supported).
        """
        if node_id <= 0:
            return None
        mode = str(os.environ.get("METABONK_PIPEWIRE_TARGET_MODE", "") or "").strip().lower()
        if not mode:
            # Prefer stable serials when available; fall back to node id if needed.
            # `pipewiresrc target-object` is documented as accepting name/serial.
            mode = "node-serial"
        allow_node_serial = mode in ("serial", "node-serial")
        allow_port_serial = mode in ("serial", "port-serial", "node-serial")
        fallback_id = str(int(node_id))
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            data = _pipewire_objects()
            # Prefer the node entry if present.
            try:
                for obj in data:
                    if not isinstance(obj, dict):
                        continue
                    if str(obj.get("type") or "") != "PipeWire:Interface:Node":
                        continue
                    try:
                        oid = int(obj.get("id"))
                    except Exception:
                        continue
                    if oid != int(node_id):
                        continue
                    props = (obj.get("info") or {}).get("props") or {}
                    if not isinstance(props, dict):
                        continue
                    if mode in ("id", "node", "node-id"):
                        return str(int(node_id))
                    if mode in ("path", "object.path"):
                        op = props.get("object.path")
                        if isinstance(op, str) and op:
                            return op
                    ser = props.get("object.serial")
                    if ser is not None and (allow_node_serial or mode in ("object.serial",)):
                        try:
                            return str(int(ser))
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                for obj in data:
                    if not isinstance(obj, dict):
                        continue
                    if str(obj.get("type") or "") != "PipeWire:Interface:Port":
                        continue
                    props = (obj.get("info") or {}).get("props") or {}
                    if not isinstance(props, dict):
                        continue
                    try:
                        nid = int(props.get("node.id"))
                    except Exception:
                        continue
                    if nid != int(node_id):
                        continue
                    if str(props.get("port.direction") or "") != "out":
                        continue
                    if allow_port_serial or mode in ("object.serial",):
                        ser = props.get("object.serial")
                        if ser is not None:
                            try:
                                return str(int(ser))
                            except Exception:
                                pass
                    if mode in ("path", "object.path"):
                        op = props.get("object.path")
                        if isinstance(op, str) and op:
                            return op
                    if mode in ("id", "node", "node-id"):
                        return str(int(node_id))
            except Exception:
                pass
            if time.time() >= deadline:
                break
            time.sleep(max(0.05, float(poll_s)))
        # If PipeWire introspection tools are unavailable, fall back to raw node id.
        return fallback_id

    @staticmethod
    def _resolve_pipewire_target_object_for_pid(
        pid: int, timeout_s: float = 2.0, poll_s: float = 0.1
    ) -> Optional[str]:
        """Resolve a gamescope process PID -> unique `pipewiresrc target-object`.

        This is more robust than relying on log parsing because the "stream available"
        line may be pushed out of the log tail, and because PipeWire can recreate ports
        after restarts.

        Strategy:
          - Find PipeWire nodes where `application.process.id == pid` and media is Video.
          - Return the node/port target based on METABONK_PIPEWIRE_TARGET_MODE.
        """
        if pid <= 1:
            return None
        deadline = time.time() + max(0.0, float(timeout_s))
        mode = str(os.environ.get("METABONK_PIPEWIRE_TARGET_MODE", "") or "").strip().lower()
        if not mode:
            mode = "node-serial"
        allow_node_serial = mode in ("serial", "node-serial")
        allow_port_serial = mode in ("serial", "port-serial", "node-serial")
        while True:
            data = _pipewire_objects()

            node_ids: list[int] = []
            try:
                for obj in data:
                    if not isinstance(obj, dict):
                        continue
                    if str(obj.get("type") or "") != "PipeWire:Interface:Node":
                        continue
                    props = (obj.get("info") or {}).get("props") or {}
                    if not isinstance(props, dict):
                        continue
                    try:
                        apid = int(props.get("application.process.id"))
                    except Exception:
                        continue
                    if apid != int(pid):
                        continue
                    media = str(props.get("media.class") or "")
                    if "Video" not in media:
                        continue
                    try:
                        node_id = int(obj.get("id"))
                    except Exception:
                        continue
                    if mode in ("id", "node", "node-id") and node_id > 0:
                        return str(node_id)
                    if mode in ("path", "object.path"):
                        op = props.get("object.path")
                        if isinstance(op, str) and op:
                            return op
                    if allow_node_serial or mode in ("object.serial",):
                        ser = props.get("object.serial")
                        if ser is not None:
                            try:
                                return str(int(ser))
                            except Exception:
                                pass
                    if node_id > 0:
                        node_ids.append(node_id)
            except Exception:
                node_ids = []

            if node_ids:
                # Find an output port for any of the matched nodes.
                try:
                    for obj in data:
                        if not isinstance(obj, dict):
                            continue
                        if str(obj.get("type") or "") != "PipeWire:Interface:Port":
                            continue
                        props = (obj.get("info") or {}).get("props") or {}
                        if not isinstance(props, dict):
                            continue
                        try:
                            nid = int(props.get("node.id"))
                        except Exception:
                            continue
                        if nid not in node_ids:
                            continue
                        if str(props.get("port.direction") or "") != "out":
                            continue
                        if allow_port_serial or mode in ("object.serial",):
                            ser = props.get("object.serial")
                            if ser is not None:
                                try:
                                    return str(int(ser))
                                except Exception:
                                    pass
                        if mode in ("path", "object.path"):
                            op = props.get("object.path")
                            if isinstance(op, str) and op:
                                return op
                        if mode in ("id", "node", "node-id"):
                            return str(int(nid))
                except Exception:
                    pass

            if time.time() >= deadline:
                break
            time.sleep(max(0.05, float(poll_s)))
        # Best-effort fallback: if we at least found a node id, use it directly.
        if node_ids:
            try:
                return str(int(node_ids[0]))
            except Exception:
                pass
        return None

    @staticmethod
    def list_pipewire_video_nodes() -> list[tuple[int, dict]]:
        """Return (node_id, props) for video-ish PipeWire nodes (best-effort)."""
        data = _pipewire_objects()
        if not isinstance(data, list) or not data:
            return []

        nodes: list[tuple[int, dict]] = []
        for obj in data:
            try:
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("type") or "") != "PipeWire:Interface:Node":
                    continue
                props = (obj.get("info") or {}).get("props") or {}
                if not isinstance(props, dict):
                    continue
                media_class = str(props.get("media.class") or "")
                if "Video" not in media_class:
                    continue
                node_id = int(obj.get("id"))
                nodes.append((node_id, props))
            except Exception:
                continue

        nodes.sort(key=lambda x: x[0])
        return nodes

    def shutdown(self) -> None:
        # Best-effort graceful shutdown for spawned process (if any).
        proc = self.proc
        self.proc = None
        if not proc:
            return
        try:
            if getattr(proc, "poll", lambda: 0)() is None:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    if os.name == "posix":
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    else:
                        proc.kill()
                    try:
                        proc.wait(timeout=2.0)
                    except Exception:
                        pass
        except Exception:
            return
