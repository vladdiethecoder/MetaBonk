"""Spectator Mode Dashboard (stream-facing layouts).

Two modes are supported:
  - `ingame` (default): ruthless minimalism for the main stream.
      Always-visible elements:
        1) Agent mosaic grid
        2) Compact leaderboard strip (top)
        3) Tiny status/ticker footer (bottom)
  - `full`: previous one-page view with charts for break/nerd scenes.

Video tiles are intentionally left blank until you provide a real source (e.g.
OBS browser sources or an NVDEC pipeline). The dashboard does not synthesize
video.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import dearpygui.dearpygui as dpg
    HAS_DPG = True
except ImportError:  # pragma: no cover
    HAS_DPG = False

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from src.common.schemas import Heartbeat


# Colorblind-safe-ish palette (RGB 0-255).
PALETTE: List[Tuple[int, int, int]] = [
    (0, 200, 255),   # cyan
    (255, 128, 0),   # orange
    (180, 0, 255),   # purple
    (0, 220, 120),   # green
    (255, 64, 128),  # pink
    (255, 215, 0),   # gold
    (128, 180, 255), # light blue
    (255, 80, 80),   # red
]


@dataclass
class AgentView:
    instance_id: str
    name: str
    color: Tuple[int, int, int]
    policy_name: str = ""
    last_reward: float = 0.0
    ema_reward: float = 0.0
    best_reward: float = 0.0
    step: int = 0
    last_ts: float = field(default_factory=time.time)
    history: List[float] = field(default_factory=list)

    def update_from_hb(self, hb: Heartbeat, ema_beta: float = 0.05) -> bool:
        """Update metrics. Returns True if a new personal best was set."""
        r = float(hb.reward or hb.steam_score or 0.0)
        self.last_reward = r
        self.ema_reward = (1 - ema_beta) * self.ema_reward + ema_beta * r
        new_best = r > self.best_reward + 1e-6
        self.best_reward = max(self.best_reward, r)
        self.policy_name = hb.policy_name or self.policy_name
        self.step = hb.step
        self.last_ts = hb.ts
        self.history.append(r)
        if len(self.history) > 300:
            self.history = self.history[-300:]
        return new_best


class SpectatorDashboard:
    def __init__(self, orch_url: str, refresh_hz: float = 2.0, mode: str = "ingame"):
        self.orch_url = orch_url.rstrip("/")
        self.refresh_hz = refresh_hz
        self.mode = mode
        self.agents: Dict[str, AgentView] = {}
        self.events: List[str] = []
        self._prev_ranks: Dict[str, int] = {}
        self._flash_until: Dict[str, float] = {}

        # Global histories
        self.mean_hist: List[float] = []
        self.best_hist: List[float] = []
        self.time_hist: List[float] = []

        self.initialized = False
        self.start_time = time.time()
        self.top_clip: Optional[dict] = None
        self._seen_clip_event: Optional[str] = None

    # ---------------- UI setup ----------------

    def setup_ui(self):
        if not HAS_DPG:
            print("Dear PyGui not installed - spectator dashboard disabled")
            return
        dpg.create_context()
        dpg.create_viewport(title="MetaBonk Spectator Mode", width=1920, height=1080)

        with dpg.window(tag="root", no_title_bar=True, width=1920, height=1080):
            self._build_top_bar()
            dpg.add_separator()
            if self.mode == "full":
                self._build_full_body()
            else:
                self._build_ingame_body()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.initialized = True

    def _build_top_bar(self):
        with dpg.group(horizontal=True):
            dpg.add_text("MetaBonk Cortex – Live Training", color=(255, 255, 255))
            dpg.add_spacer(width=20)
            dpg.add_text("Run:", color=(160, 160, 160))
            dpg.add_text("-", tag="run_label")
            dpg.add_spacer(width=20)
            dpg.add_text("Steps:", color=(160, 160, 160))
            dpg.add_text("0", tag="steps_label")
            dpg.add_spacer(width=20)
            dpg.add_text("Agents:", color=(160, 160, 160))
            dpg.add_text("0", tag="agents_label")
            dpg.add_spacer(width=20)
            dpg.add_text("Avg Reward:", color=(160, 160, 160))
            dpg.add_text("0.0", tag="avg_reward_label")

    def _build_full_body(self):
        # Previous one-page layout (used for break/nerd scenes).
        with dpg.group(horizontal=True):
            # Left column (feeds)
            with dpg.child_window(width=1400, height=680):
                with dpg.group(horizontal=True):
                    # Featured feed (optional)
                    with dpg.child_window(width=920, height=640, tag="featured_feed"):
                        dpg.add_text("Featured Agent", tag="featured_title")
                        dpg.add_spacer(height=10)
                        dpg.add_text("no video source configured", color=(120, 120, 120))
                        dpg.add_text("Reward: 0.0", tag="featured_reward")
                        dpg.add_text("Policy: -", tag="featured_policy")
                    # Thumbnails column
                    with dpg.child_window(width=440, height=640):
                        dpg.add_text("Other Agents")
                        for i in range(5):
                            with dpg.child_window(width=-1, height=110, tag=f"thumb_{i}"):
                                dpg.add_text("—", tag=f"thumb_name_{i}")
                                dpg.add_text("Reward: 0.0", tag=f"thumb_reward_{i}")

            # Right column (leaderboard)
            with dpg.child_window(width=480, height=680):
                dpg.add_text("Leaderboard", color=(255, 255, 255))
                with dpg.table(header_row=True, tag="leader_table", resizable=True):
                    dpg.add_table_column(label="#")
                    dpg.add_table_column(label="Agent")
                    dpg.add_table_column(label="AvgR")
                    dpg.add_table_column(label="BestR")
                    dpg.add_table_column(label="Policy")

        # Bottom band (global curves + ticker)
        with dpg.group(horizontal=True):
            with dpg.child_window(width=1440, height=340):
                dpg.add_text("Learning Progress", color=(255, 255, 255))
                with dpg.plot(label="Reward Over Time", height=260, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="reward_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Reward", tag="reward_y"):
                        dpg.add_line_series([], [], label="Mean", tag="mean_series")
                        dpg.add_line_series([], [], label="Best", tag="best_series")
            with dpg.child_window(width=440, height=340):
                dpg.add_text("Events", color=(255, 255, 255))
                dpg.add_input_text(
                    tag="event_log",
                    multiline=True,
                    readonly=True,
                    height=280,
                    width=-1,
                    default_value="Waiting for hype moments...",
                )

    def _build_ingame_body(self):
        """Minimal in-game overlay per stream spec."""
        # Speedrun-style composition: left 2x2 grid, right tall sidebar.
        main_h = 980
        left_w = 1400
        right_w = 480

        with dpg.group(horizontal=True):
            # Left: 2x2 agent grid.
            with dpg.child_window(width=left_w, height=main_h, tag="grid_left"):
                cols = 2
                rows = 2
                pad = 8
                tile_w = int((left_w - (cols + 1) * pad) / cols)
                tile_h = int((main_h - (rows + 1) * pad) / rows)
                for r in range(rows):
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=pad)
                        for c in range(cols):
                            i = r * cols + c
                            with dpg.child_window(width=tile_w, height=tile_h, tag=f"grid_tile_{i}"):
                                dpg.add_text("—", tag=f"grid_name_{i}")
                                dpg.add_text("Policy: —", tag=f"grid_policy_{i}", color=(150, 150, 150))
                                dpg.add_text("R: —", tag=f"grid_reward_{i}", color=(200, 200, 200))
                                dpg.add_progress_bar(default_value=0.0, tag=f"grid_bar_{i}", width=-1)
                            dpg.add_spacer(width=pad)

            # Right: leaderboard + events column.
            with dpg.child_window(width=right_w, height=main_h, tag="sidebar"):
                # Top run replay (optional).
                with dpg.child_window(width=-1, height=220, tag="top_replay_box"):
                    dpg.add_text("TOP RUN REPLAY", color=(246, 191, 60))
                    with dpg.child_window(width=-1, height=160, tag="top_replay_video"):
                        dpg.add_text("no replay source configured", color=(120, 120, 120))
                    dpg.add_text("—", tag="top_replay_meta", color=(200, 200, 200))

                # Leaderboard device (top).
                with dpg.child_window(width=-1, height=540, tag="leader_device"):
                    dpg.add_text("MEGABONK LADDER", color=(246, 191, 60))
                    for i in range(8):
                        with dpg.child_window(width=-1, height=70, tag=f"leader_row_{i}"):
                            with dpg.group(horizontal=True):
                                dpg.add_text("#", tag=f"leader_rank_{i}", color=(246, 191, 60))
                                dpg.add_spacer(width=6)
                                dpg.add_text("—", tag=f"leader_name_{i}")
                                dpg.add_spacer(width=6)
                                dpg.add_text("—", tag=f"leader_policy_{i}", color=(140, 140, 140))
                                dpg.add_spacer(width=6)
                                dpg.add_text("Best: —", tag=f"leader_best_{i}", color=(246, 191, 60))
                                dpg.add_spacer(width=4)
                                dpg.add_text("", tag=f"leader_delta_{i}", color=(200, 200, 200))

                # Events + tiny status footer (bottom).
                with dpg.child_window(width=-1, height=220, tag="events_box"):
                    dpg.add_text("Events", color=(200, 200, 200))
                    dpg.add_input_text(
                        tag="event_log_ingame",
                        multiline=True,
                        readonly=True,
                        height=140,
                        width=-1,
                        default_value="Waiting for hype moments…",
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_text("FPS: —", tag="footer_fps", color=(180, 180, 180))
                        dpg.add_spacer(width=10)
                        dpg.add_text("Step: 0", tag="footer_step", color=(180, 180, 180))
                        dpg.add_spacer(width=10)
                        dpg.add_text("Runtime: 00:00:00", tag="footer_runtime", color=(180, 180, 180))
    # ---------------- Data collection ----------------

    def _poll_workers(self) -> Dict[str, Heartbeat]:
        if requests is None:
            return {}
        try:
            r = requests.get(f"{self.orch_url}/workers", timeout=1.0)
            if not r.ok:
                return {}
            data = r.json() or {}
            out: Dict[str, Heartbeat] = {}
            for wid, hb_raw in data.items():
                try:
                    out[wid] = Heartbeat(**hb_raw)
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    def _poll_top_clip(self) -> Optional[dict]:
        if requests is None:
            return None
        try:
            r = requests.get(f"{self.orch_url}/events", params={"limit": 100}, timeout=1.0)
            if not r.ok:
                return None
            data = r.json() or []
            clips = [e for e in data if e.get("event_type") == "NewBestRunClip"]
            if not clips:
                return None
            latest = clips[-1]
            if latest.get("event_id") == self._seen_clip_event:
                return self.top_clip
            self._seen_clip_event = latest.get("event_id")
            return latest.get("payload") or {}
        except Exception:
            return None

    def _ensure_agent(self, hb: Heartbeat):
        if hb.instance_id in self.agents:
            return
        idx = len(self.agents) % len(PALETTE)
        color = PALETTE[idx]
        name = hb.instance_id
        self.agents[hb.instance_id] = AgentView(
            instance_id=hb.instance_id,
            name=name,
            color=color,
            policy_name=hb.policy_name or "",
        )

    def _update_global_histories(self):
        if not self.agents:
            return
        mean_r = float(np.mean([a.ema_reward for a in self.agents.values()]))
        best_r = float(max(a.best_reward for a in self.agents.values()))
        self.mean_hist.append(mean_r)
        self.best_hist.append(best_r)
        self.time_hist.append(time.time())
        if len(self.mean_hist) > 600:
            self.mean_hist = self.mean_hist[-600:]
            self.best_hist = self.best_hist[-600:]
            self.time_hist = self.time_hist[-600:]

    # ---------------- UI update loop ----------------

    def update_once(self):
        hbs = self._poll_workers()
        if self.mode != "full":
            clip = self._poll_top_clip()
            if clip:
                self.top_clip = clip
        for hb in hbs.values():
            self._ensure_agent(hb)
            new_best = self.agents[hb.instance_id].update_from_hb(hb)
            if new_best:
                self.events.append(
                    f"{self.agents[hb.instance_id].name} set PB {self.agents[hb.instance_id].best_reward:.1f}"
                )
        if len(self.events) > 200:
            self.events = self.events[-200:]

        self._update_global_histories()

        # Leaderboard: sort by EMA reward.
        ranked = sorted(self.agents.values(), key=lambda a: a.ema_reward, reverse=True)
        featured = ranked[0] if ranked else None

        # UI labels
        if HAS_DPG:
            dpg.set_value("agents_label", str(len(ranked)))
            if featured:
                dpg.set_value("avg_reward_label", f"{featured.ema_reward:.2f}")
                dpg.set_value("steps_label", str(featured.step))
            if self.mode == "full":
                dpg.set_value("featured_title", f"{featured.name}" if featured else "—")
                dpg.set_value(
                    "featured_reward",
                    f"Reward: {featured.last_reward:.2f} (EMA {featured.ema_reward:.2f})" if featured else "—",
                )
                dpg.set_value("featured_policy", f"Policy: {featured.policy_name}" if featured else "—")

                # Thumbnails
                for i in range(5):
                    if i + 1 < len(ranked):
                        a = ranked[i + 1]
                        dpg.set_value(f"thumb_name_{i}", a.name)
                        dpg.set_value(f"thumb_reward_{i}", f"Reward: {a.last_reward:.2f} (EMA {a.ema_reward:.2f})")
                    else:
                        dpg.set_value(f"thumb_name_{i}", "—")
                        dpg.set_value(f"thumb_reward_{i}", "Reward: —")

                # Rebuild leaderboard table rows.
                if dpg.does_item_exist("leader_table"):
                    dpg.delete_item("leader_table", children_only=True)
                    for i, a in enumerate(ranked, start=1):
                        with dpg.table_row(parent="leader_table"):
                            dpg.add_text(str(i), color=a.color)
                            dpg.add_text(a.name, color=a.color)
                            dpg.add_text(f"{a.ema_reward:.1f}")
                            dpg.add_text(f"{a.best_reward:.1f}")
                            dpg.add_text(a.policy_name)

                # Reward plot
                xs = list(range(len(self.mean_hist)))
                dpg.set_value("mean_series", [xs, self.mean_hist])
                dpg.set_value("best_series", [xs, self.best_hist])

                if self.events:
                    dpg.set_value("event_log", "\n".join(self.events[-20:]))
            else:
                # Rank deltas for climb/drop highlighting.
                rank_map = {a.instance_id: i + 1 for i, a in enumerate(ranked)}
                now_ts = time.time()
                prev_snapshot = dict(self._prev_ranks)
                rank_deltas: Dict[str, int] = {}
                for aid, rnk in rank_map.items():
                    prev = prev_snapshot.get(aid, rnk)
                    delta = prev - rnk
                    rank_deltas[aid] = delta
                    if delta > 0:
                        self._flash_until[aid] = now_ts + 0.9
                        self.events.append(f"{aid} climbed to #{rnk}")
                    elif delta < 0:
                        self._flash_until[aid] = now_ts + 0.6
                    self._prev_ranks[aid] = rnk

                # Update left 2x2 grid with top 4 agents.
                grid_agents = ranked[:4]
                global_best = max([a.best_reward for a in ranked], default=1.0)
                for i in range(4):
                    if i < len(grid_agents):
                        a = grid_agents[i]
                        dpg.set_value(f"grid_name_{i}", a.name)
                        dpg.configure_item(f"grid_name_{i}", color=a.color)
                        dpg.set_value(f"grid_policy_{i}", f"Policy: {a.policy_name}")
                        dpg.set_value(
                            f"grid_reward_{i}",
                            f"R: {a.last_reward:.1f} (avg {a.ema_reward:.1f})",
                        )
                        bar_v = max(0.0, min(1.0, a.ema_reward / (global_best + 1e-6)))
                        dpg.set_value(f"grid_bar_{i}", bar_v)
                    else:
                        dpg.set_value(f"grid_name_{i}", "—")
                        dpg.set_value(f"grid_policy_{i}", "Policy: —")
                        dpg.set_value(f"grid_reward_{i}", "R: —")
                        dpg.set_value(f"grid_bar_{i}", 0.0)

                # Update leaderboard rows (up to 8).
                for i in range(8):
                    if i < len(ranked):
                        a = ranked[i]
                        rnk = i + 1
                        dpg.set_value(f"leader_rank_{i}", str(rnk))
                        dpg.set_value(f"leader_name_{i}", a.name)
                        dpg.configure_item(f"leader_name_{i}", color=a.color)
                        dpg.set_value(f"leader_policy_{i}", a.policy_name)
                        dpg.set_value(f"leader_best_{i}", f"{a.best_reward:.1f}")
                        delta = rank_deltas.get(a.instance_id, 0)
                        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "")
                        dpg.set_value(f"leader_delta_{i}", arrow)
                        # Flash climbed rows gold.
                        if now_ts < self._flash_until.get(a.instance_id, 0.0):
                            dpg.configure_item(f"leader_row_{i}", border=True)
                            dpg.configure_item(f"leader_name_{i}", color=(246, 191, 60))
                        else:
                            dpg.configure_item(f"leader_row_{i}", border=False)
                    else:
                        dpg.set_value(f"leader_rank_{i}", "#")
                        dpg.set_value(f"leader_name_{i}", "—")
                        dpg.set_value(f"leader_policy_{i}", "—")
                        dpg.set_value(f"leader_best_{i}", "—")
                        dpg.set_value(f"leader_delta_{i}", "")

                # Events box.
                if self.events:
                    dpg.set_value("event_log_ingame", "\n".join(self.events[-12:]))

                # Footer runtime/step.
                runtime_s = int(time.time() - self.start_time)
                hh = runtime_s // 3600
                mm = (runtime_s % 3600) // 60
                ss = runtime_s % 60
                dpg.set_value("footer_runtime", f"Runtime: {hh:02d}:{mm:02d}:{ss:02d}")
                dpg.set_value("footer_step", f"Step: {featured.step if featured else 0}")

                # Top replay box metadata (video texture pending).
                if self.top_clip and dpg.does_item_exist("top_replay_meta"):
                    name = self.top_clip.get("agent_name") or self.top_clip.get("instance_id") or "—"
                    score = self.top_clip.get("score", 0.0)
                    speed = self.top_clip.get("speed", 3.0)
                    dpg.set_value("top_replay_meta", f"{name} — PB {score:.1f} — x{speed} replay")

    def run(self):
        if not self.initialized:
            return
        last = 0.0
        period = 1.0 / max(self.refresh_hz, 0.1)
        while dpg.is_dearpygui_running():
            now = time.time()
            if now - last >= period:
                self.update_once()
                last = now
            dpg.render_dearpygui_frame()

    def cleanup(self):
        if HAS_DPG:
            dpg.destroy_context()
