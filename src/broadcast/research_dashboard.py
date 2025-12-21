"""Research Command Center (RCC) - The "Nerd Tab".

Real-time scientific observability dashboard using Dear PyGui:
- Panel A: Research Worth (Epistemic Value / Surprise)
- Panel B: SOTA Evaluation (Baseline Comparison)
- Panel C: Skill Vector Library (Phylogenetic Tree)
- Panel D: System Health (GPU Stats, Training Metrics)

References:
- Nerd Tab Protocol specification
- Active Inference epistemic value
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import dearpygui.dearpygui as dpg
    HAS_DPG = True
except ImportError:
    HAS_DPG = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    import rerun as rr  # type: ignore
    HAS_RERUN = True
except Exception:
    HAS_RERUN = False

try:
    import pynvml  # type: ignore
    HAS_NVML = True
except Exception:
    HAS_NVML = False


@dataclass
class ResearchMetrics:
    """Metrics for the research dashboard."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Epistemic value (surprise/information gain)
    epistemic_value: float = 0.0
    world_model_surprise: float = 0.0
    kl_divergence: float = 0.0
    
    # Performance vs baselines
    agent_score: float = 0.0
    random_baseline: float = float("nan")
    scripted_baseline: float = float("nan")
    human_world_record: float = float("nan")
    previous_best: float = float("nan")
    
    # Win probability
    win_probability: float = 0.0
    
    # Skill vectors
    active_skills: List[str] = field(default_factory=list)
    skill_weights: Dict[str, float] = field(default_factory=dict)
    
    # Training dynamics
    gradient_norm: float = 0.0
    policy_entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    
    # System health
    vram_usage_gb: float = 0.0
    vram_total_gb: float = 32.0
    gpu_utilization: float = 0.0
    tensor_core_utilization: float = 0.0
    fp8_saturation: float = 0.0
    samples_per_second: float = 0.0
    
    # Goals
    ultimate_goal: str = ""
    ultimate_progress: float = 0.0
    strategic_goal: str = ""
    strategic_progress: float = 0.0
    tactical_goal: str = ""
    tactical_progress: float = 0.0
    
    # Internal monologue
    reasoning_trace: str = ""


@dataclass  
class EurekaEvent:
    """A "eureka" discovery event."""
    
    timestamp: float
    hypothesis: str
    evidence: str
    confidence: float


class ResearchDashboard:
    """The Research Command Center UI.
    
    Provides real-time scientific observability for the AI system.
    """
    
    def __init__(self, history_length: int = 1000):
        self.history_len = history_length
        
        # Data histories
        self.entropy_history = np.zeros(history_length)
        self.surprise_history = np.zeros(history_length)
        self.agent_score_history = np.zeros(history_length)
        self.baseline_history = np.zeros(history_length)
        self.human_wr_history = np.zeros(history_length)
        self.gradient_history = np.zeros(history_length)
        self.tps_history = np.zeros(history_length)
        
        # Eureka log
        self.eureka_events: List[EurekaEvent] = []
        
        # Current metrics
        self.current_metrics = ResearchMetrics()
        
        # UI tags
        self.initialized = False
        self._skill_button_tags: List[str] = []
        self._skill_composition_tags: List[str] = []
    
    def setup_ui(self):
        """Set up the Dear PyGui UI."""
        if not HAS_DPG:
            print("Dear PyGui not installed - dashboard disabled")
            return
        
        dpg.create_context()
        
        # Create viewport
        dpg.create_viewport(title="🔬 Metabonk Research Command Center", width=800, height=1000)
        
        with dpg.window(label="Research Telemetry", tag="main_window", width=780, height=980):
            
            # ═══════════════════════════════════════════════════════════
            # PANEL A: RESEARCH WORTH
            # ═══════════════════════════════════════════════════════════
            with dpg.collapsing_header(label="🧠 Research Worth (Epistemic Value)", default_open=True):
                dpg.add_text("Quantifying Discovery: Is the agent learning something new?")
                
                with dpg.plot(label="Information Gain", height=180, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps", tag="surprise_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Bits", tag="surprise_y"):
                        dpg.add_line_series(
                            list(range(self.history_len)),
                            list(self.surprise_history),
                            label="World Model Surprise",
                            tag="surprise_series"
                        )
                        dpg.add_line_series(
                            list(range(self.history_len)),
                            list(self.entropy_history),
                            label="Policy Entropy",
                            tag="entropy_series"
                        )
                
                # Eureka indicator
                with dpg.group(horizontal=True):
                    dpg.add_text("💡 Eureka Status:", color=(255, 255, 0))
                    dpg.add_text("No recent discoveries", tag="eureka_text", color=(150, 150, 150))
                
                # Discovery log
                dpg.add_text("Recent Discoveries:", color=(200, 200, 255))
                dpg.add_input_text(
                    tag="eureka_log",
                    multiline=True,
                    readonly=True,
                    height=80,
                    width=-1,
                    default_value="Waiting for discoveries..."
                )
            
            dpg.add_separator()
            
            # ═══════════════════════════════════════════════════════════
            # PANEL B: SOTA EVALUATION
            # ═══════════════════════════════════════════════════════════
            with dpg.collapsing_header(label="📊 SOTA Evaluation", default_open=True):
                dpg.add_text("Real-time comparison against baselines")
                
                with dpg.plot(label="Performance vs Baselines", height=180, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="sota_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Normalized Score", tag="sota_y"):
                        dpg.add_line_series([], [], label="Current Agent", tag="agent_score_series")
                        dpg.add_line_series([], [], label="Scripted Bot", tag="baseline_series")
                        dpg.add_line_series([], [], label="Human WR", tag="human_wr_series")
                
                # Win probability
                with dpg.group(horizontal=True):
                    dpg.add_text("🏆 Win Probability:", color=(0, 255, 128))
                    dpg.add_text("—", tag="win_prob_text", color=(255, 255, 255))
                
                # Baseline comparison table
                with dpg.table(header_row=True, tag="baseline_table"):
                    dpg.add_table_column(label="Baseline")
                    dpg.add_table_column(label="Score")
                    dpg.add_table_column(label="Gap")
                    
                    with dpg.table_row():
                        dpg.add_text("Random Agent")
                        dpg.add_text("—", tag="random_score")
                        dpg.add_text("—", tag="random_gap")
                    
                    with dpg.table_row():
                        dpg.add_text("Scripted Bot")
                        dpg.add_text("—", tag="scripted_score")
                        dpg.add_text("—", tag="scripted_gap")
                    
                    with dpg.table_row():
                        dpg.add_text("Human World Record")
                        dpg.add_text("—", tag="human_score")
                        dpg.add_text("—", tag="human_gap")
            
            dpg.add_separator()
            
            # ═══════════════════════════════════════════════════════════
            # PANEL C: SKILL VECTORS
            # ═══════════════════════════════════════════════════════════
            with dpg.collapsing_header(label="🧬 Skill Vector Library", default_open=True):
                dpg.add_text("Active Task Arithmetic Skills")
                
                # Active skills
                with dpg.group(horizontal=True, tag="skill_buttons"):
                    dpg.add_text("no active skills yet", tag="skills_empty", color=(150, 150, 150))
                
                # Skill composition
                dpg.add_text("Agent Composition:", color=(200, 200, 255))
                with dpg.group(tag="skill_composition"):
                    dpg.add_text("no composition yet", tag="skill_comp_empty", color=(150, 150, 150))
            
            dpg.add_separator()
            
            # ═══════════════════════════════════════════════════════════
            # PANEL D: SYSTEM HEALTH
            # ═══════════════════════════════════════════════════════════
            with dpg.collapsing_header(label="💻 System Health", default_open=True):
                dpg.add_text("RTX 5090 Status & Training Dynamics")
                
                # GPU metrics
                with dpg.table(header_row=False):
                    dpg.add_table_column(width=150)
                    dpg.add_table_column(width=-1)
                    dpg.add_table_column(width=60)
                    
                    with dpg.table_row():
                        dpg.add_text("VRAM Usage:")
                        dpg.add_progress_bar(default_value=0.0, tag="vram_bar", width=-1)
                        dpg.add_text("0/32 GB", tag="vram_text")
                    
                    with dpg.table_row():
                        dpg.add_text("GPU Utilization:")
                        dpg.add_progress_bar(default_value=0.0, tag="gpu_bar", width=-1)
                        dpg.add_text("0%", tag="gpu_text")
                    
                    with dpg.table_row():
                        dpg.add_text("Tensor Cores:")
                        dpg.add_progress_bar(default_value=0.0, tag="tensor_bar", width=-1)
                        dpg.add_text("0%", tag="tensor_text")
                    
                    with dpg.table_row():
                        dpg.add_text("FP8 Saturation:")
                        dpg.add_progress_bar(default_value=0.0, tag="fp8_bar", width=-1)
                        dpg.add_text("0%", tag="fp8_text")
                
                # Training metrics
                dpg.add_text("Training Dynamics:", color=(200, 200, 255))
                with dpg.group(horizontal=True):
                    dpg.add_text("Throughput:")
                    dpg.add_text("0 steps/sec", tag="tps_text", color=(0, 255, 128))
                    dpg.add_spacer(width=20)
                    dpg.add_text("Gradient Norm:")
                    dpg.add_text("0.0", tag="grad_norm_text")
                
                # Gradient plot
                with dpg.plot(label="Training Stability", height=120, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Steps", tag="grad_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Gradient Norm", tag="grad_y"):
                        dpg.add_line_series([], [], label="∇θ", tag="gradient_series")
            
            dpg.add_separator()
            
            # ═══════════════════════════════════════════════════════════
            # PANEL E: GOALS & AMBITIONS
            # ═══════════════════════════════════════════════════════════
            with dpg.collapsing_header(label="🎯 Goals & Ambitions", default_open=True):
                dpg.add_text("Current AI Objectives")
                
                # Goal hierarchy
                with dpg.group():
                    dpg.add_text("🏆 Ultimate Goal:", color=(255, 215, 0))
                    with dpg.group(horizontal=True):
                        dpg.add_text("—", tag="ultimate_goal")
                        dpg.add_progress_bar(default_value=0.0, tag="ultimate_bar", width=200)
                        dpg.add_text("—", tag="ultimate_pct")
                    
                    dpg.add_spacer(height=5)
                    dpg.add_text("📋 Strategic Goal:", color=(100, 200, 255))
                    with dpg.group(horizontal=True):
                        dpg.add_text("—", tag="strategic_goal")
                        dpg.add_progress_bar(default_value=0.0, tag="strategic_bar", width=200)
                        dpg.add_text("—", tag="strategic_pct")
                    
                    dpg.add_spacer(height=5)
                    dpg.add_text("⚡ Tactical Goal:", color=(255, 150, 100))
                    with dpg.group(horizontal=True):
                        dpg.add_text("—", tag="tactical_goal")
                        dpg.add_progress_bar(default_value=0.0, tag="tactical_bar", width=200)
                        dpg.add_text("—", tag="tactical_pct")
                
                # Internal monologue
                dpg.add_spacer(height=10)
                dpg.add_text("💭 Internal Reasoning:", color=(200, 200, 255))
                dpg.add_input_text(
                    tag="reasoning_log",
                    multiline=True,
                    readonly=True,
                    height=60,
                    width=-1,
                    default_value=""
                )
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        self.initialized = True
    
    def update(self, metrics: ResearchMetrics):
        """Update dashboard with new metrics."""
        if not self.initialized or not HAS_DPG:
            return
        
        self.current_metrics = metrics
        
        # Update histories
        self.surprise_history = np.roll(self.surprise_history, -1)
        self.surprise_history[-1] = metrics.world_model_surprise
        
        self.entropy_history = np.roll(self.entropy_history, -1)
        self.entropy_history[-1] = metrics.policy_entropy
        
        self.agent_score_history = np.roll(self.agent_score_history, -1)
        self.agent_score_history[-1] = metrics.agent_score
        
        self.baseline_history = np.roll(self.baseline_history, -1)
        self.baseline_history[-1] = metrics.scripted_baseline
        
        self.human_wr_history = np.roll(self.human_wr_history, -1)
        self.human_wr_history[-1] = metrics.human_world_record
        
        self.gradient_history = np.roll(self.gradient_history, -1)
        self.gradient_history[-1] = metrics.gradient_norm
        
        self.tps_history = np.roll(self.tps_history, -1)
        self.tps_history[-1] = metrics.samples_per_second
        
        # Update plots
        x = list(range(self.history_len))
        dpg.set_value("surprise_series", [x, list(self.surprise_history)])
        dpg.set_value("entropy_series", [x, list(self.entropy_history)])
        dpg.set_value("agent_score_series", [x, list(self.agent_score_history)])
        dpg.set_value("baseline_series", [x, list(self.baseline_history)])
        dpg.set_value("human_wr_series", [x, list(self.human_wr_history)])
        dpg.set_value("gradient_series", [x, list(self.gradient_history)])
        
        # Update win probability
        dpg.set_value("win_prob_text", f"{metrics.win_probability * 100:.1f}%")
        
        # Update baseline table
        def _fmt(v: float) -> str:
            return "—" if not math.isfinite(float(v)) else f"{float(v):.1f}"

        dpg.set_value("random_score", _fmt(metrics.random_baseline))
        dpg.set_value("scripted_score", _fmt(metrics.scripted_baseline))
        dpg.set_value("human_score", _fmt(metrics.human_world_record))

        def _gap(agent: float, base: float) -> str:
            if not (math.isfinite(float(agent)) and math.isfinite(float(base))):
                return "—"
            g = float(agent) - float(base)
            return f"+{g:.1f}" if g >= 0 else f"{g:.1f}"

        dpg.set_value("random_gap", _gap(metrics.agent_score, metrics.random_baseline))
        dpg.set_value("scripted_gap", _gap(metrics.agent_score, metrics.scripted_baseline))
        dpg.set_value("human_gap", _gap(metrics.agent_score, metrics.human_world_record))

        self._update_skill_panel(metrics)
        
        # Update system health
        vram_pct = metrics.vram_usage_gb / metrics.vram_total_gb
        dpg.set_value("vram_bar", vram_pct)
        dpg.set_value("vram_text", f"{metrics.vram_usage_gb:.1f}/{metrics.vram_total_gb:.0f} GB")
        
        dpg.set_value("gpu_bar", metrics.gpu_utilization / 100)
        dpg.set_value("gpu_text", f"{metrics.gpu_utilization:.0f}%")
        
        dpg.set_value("tensor_bar", metrics.tensor_core_utilization / 100)
        dpg.set_value("tensor_text", f"{metrics.tensor_core_utilization:.0f}%")
        
        dpg.set_value("fp8_bar", metrics.fp8_saturation / 100)
        dpg.set_value("fp8_text", f"{metrics.fp8_saturation:.0f}%")
        
        dpg.set_value("tps_text", f"{metrics.samples_per_second:,.0f} steps/sec")
        dpg.set_value("grad_norm_text", f"{metrics.gradient_norm:.4f}")
        
        # Update goals
        dpg.set_value("ultimate_goal", metrics.ultimate_goal or "—")
        dpg.set_value("ultimate_bar", metrics.ultimate_progress if metrics.ultimate_goal else 0.0)
        dpg.set_value("ultimate_pct", f"{metrics.ultimate_progress * 100:.0f}%" if metrics.ultimate_goal else "—")

        dpg.set_value("strategic_goal", metrics.strategic_goal or "—")
        dpg.set_value("strategic_bar", metrics.strategic_progress if metrics.strategic_goal else 0.0)
        dpg.set_value("strategic_pct", f"{metrics.strategic_progress * 100:.0f}%" if metrics.strategic_goal else "—")

        dpg.set_value("tactical_goal", metrics.tactical_goal or "—")
        dpg.set_value("tactical_bar", metrics.tactical_progress if metrics.tactical_goal else 0.0)
        dpg.set_value("tactical_pct", f"{metrics.tactical_progress * 100:.0f}%" if metrics.tactical_goal else "—")
        
        # Update reasoning log
        if metrics.reasoning_trace:
            dpg.set_value("reasoning_log", metrics.reasoning_trace)

    def _update_skill_panel(self, metrics: ResearchMetrics) -> None:
        if not HAS_DPG:
            return

        # --- Active skills (chips/buttons) ---
        if dpg.does_item_exist("skills_empty"):
            dpg.delete_item("skills_empty")
        for tag in self._skill_button_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._skill_button_tags = []

        if not metrics.active_skills:
            dpg.add_text("no active skills yet", parent="skill_buttons", tag="skills_empty", color=(150, 150, 150))
        else:
            for i, s in enumerate(metrics.active_skills[:16]):  # cap to keep UI responsive
                tag = f"skill_btn_{i}"
                dpg.add_button(label=str(s), parent="skill_buttons", tag=tag)
                self._skill_button_tags.append(tag)

        # --- Skill composition (weights) ---
        if dpg.does_item_exist("skill_comp_empty"):
            dpg.delete_item("skill_comp_empty")
        for tag in self._skill_composition_tags:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._skill_composition_tags = []

        if not metrics.skill_weights:
            dpg.add_text("no composition yet", parent="skill_composition", tag="skill_comp_empty", color=(150, 150, 150))
            return

        items = sorted(metrics.skill_weights.items(), key=lambda kv: float(kv[1]), reverse=True)[:8]
        for i, (name, w) in enumerate(items):
            row_tag = f"skill_comp_row_{i}"
            with dpg.group(horizontal=True, parent="skill_composition", tag=row_tag):
                dpg.add_text(str(name))
                dpg.add_progress_bar(default_value=float(w), width=200)
                dpg.add_text(f"{float(w) * 100:.0f}%")
            self._skill_composition_tags.append(row_tag)
    
    def add_eureka(self, event: EurekaEvent):
        """Add a eureka discovery event."""
        self.eureka_events.append(event)
        
        if not self.initialized or not HAS_DPG:
            return
        
        # Update eureka indicator
        dpg.set_value("eureka_text", f"NEW: {event.hypothesis}")
        
        # Update log
        log_text = "\n".join([
            f"[{e.confidence:.0%}] {e.hypothesis}"
            for e in self.eureka_events[-5:]
        ])
        dpg.set_value("eureka_log", log_text)
    
    def run_frame(self) -> bool:
        """Run one frame of the dashboard. Returns False if window closed."""
        if not self.initialized or not HAS_DPG:
            return True
        
        dpg.render_dearpygui_frame()
        return dpg.is_dearpygui_running()
    
    def cleanup(self):
        """Clean up Dear PyGui context."""
        if HAS_DPG:
            dpg.destroy_context()


class MetricsCollector:
    """Collects metrics from various system components."""

    def __init__(
        self,
        orchestrator_url: Optional[str] = None,
        learner_url: Optional[str] = None,
        policy_name: Optional[str] = None,
        baselines_path: Optional[str] = None,
        fps: float = 60.0,
        rerun_enabled: bool = False,
        request_timeout_s: float = 0.4,
    ):
        self.start_time = time.time()
        self.step_count = 0
        self.fps = fps
        self.policy_name = policy_name
        self.orch_url = orchestrator_url.rstrip("/") if orchestrator_url else None
        self.learner_url = learner_url.rstrip("/") if learner_url else None
        self.timeout_s = request_timeout_s
        self.rerun_enabled = rerun_enabled and HAS_RERUN
        self._rr_started = False

        self._instance_start: Dict[str, float] = {}
        self._last_events_ts: float = 0.0
        self._pending_eureka: List[EurekaEvent] = []

        self.baselines: Dict[str, float] = {
            "random": float("nan"),
            "scripted": float("nan"),
            "human_wr": float("nan"),
            "previous_best": float("nan"),
        }
        if baselines_path:
            self._load_baselines(baselines_path)

        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None
        else:
            self._nvml_handle = None

    # --------------------------- Public API ---------------------------

    def collect(
        self,
    ) -> ResearchMetrics:
        """Collect and compute all research metrics.

        This dashboard does not synthesize demo telemetry. If live endpoints are
        not configured or not reachable, metrics will remain at defaults and
        the UI should reflect a disconnected state.
        """
        if not (self.orch_url or self.learner_url):
            raise RuntimeError("MetricsCollector requires orchestrator_url and/or learner_url (demo mode removed).")

        metrics = self._collect_live()

        self._maybe_log_rerun(metrics)
        return metrics

    def pop_eureka_events(self) -> List[EurekaEvent]:
        out = self._pending_eureka
        self._pending_eureka = []
        return out

    # --------------------------- Live mode ---------------------------

    def _collect_live(self) -> ResearchMetrics:
        import torch

        self.step_count += 1
        now = time.time()
        elapsed = now - self.start_time

        learner_metrics = self._poll_learner()
        orch_workers = self._poll_workers()
        pbt_state = self._poll_pbt()
        self._poll_events()

        # Select agent score / survival stats from workers.
        scores, healths, survival_times = [], [], []
        for wid, hb in orch_workers.items():
            pn = (hb.get("policy_name") or "").lower()
            if self.policy_name and pn != self.policy_name.lower():
                continue
            score = hb.get("steam_score")
            if score is None:
                score = hb.get("reward", 0.0)
            scores.append(float(score or 0.0))
            hval = hb.get("health")
            if hval is not None:
                healths.append(float(hval))
            st = self._instance_start.setdefault(wid, now)
            survival_times.append(max(0.0, now - st))

        agent_score = max(scores) if scores else 0.0
        health = min(healths) if healths else 100.0
        survival_time = max(survival_times) if survival_times else 0.0

        # Learner-side metrics.
        wm_kl = float(learner_metrics.get("wm_kl", 0.0) or 0.0)
        wm_recon = float(learner_metrics.get("wm_recon", 0.0) or 0.0)
        entropy = float(learner_metrics.get("entropy", learner_metrics.get("policy_entropy", 0.0)) or 0.0)
        grad_norm = float(learner_metrics.get("grad_norm", learner_metrics.get("gradient_norm", 0.0)) or 0.0)
        policy_loss = float(learner_metrics.get("policy_loss", 0.0) or 0.0)
        value_loss = float(learner_metrics.get("value_loss", 0.0) or 0.0)
        tps = float(learner_metrics.get("learner_tps", 0.0) or 0.0)

        # GPU stats (best-effort).
        vram_usage_gb, vram_total_gb = 0.0, 32.0
        gpu_util = tensor_util = fp8_sat = 0.0
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            vram_total_gb = total_b / 1e9
            vram_usage_gb = (total_b - free_b) / 1e9
        if self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = float(util.gpu)
                tensor_util = gpu_util  # no direct TC metric in NVML; proxy with GPU util
            except Exception:
                pass

        # Baselines are only meaningful when configured (do not synthesize).
        random_baseline = float(self.baselines.get("random", float("nan")))
        scripted_baseline = float(self.baselines.get("scripted", float("nan")))
        human_wr = float(self.baselines.get("human_wr", float("nan")))
        previous_best = float(self.baselines.get("previous_best", float("nan")))

        # Skill composition / lineage.
        active_skills: List[str] = []
        skill_weights: Dict[str, float] = {}
        if self.policy_name and pbt_state.get(self.policy_name):
            st = pbt_state[self.policy_name]
            gen = st.get("generation", 0)
            active_skills.append(f"{self.policy_name}_Gen{gen}")
            parents = st.get("parents") or []
            for p in parents:
                active_skills.append(str(p))
            skill_weights = st.get("mix_weights") or {self.policy_name: 1.0}
        else:
            # Fallback to best policy in population.
            try:
                best = max(pbt_state.values(), key=lambda x: float(x.get("steam_score", 0.0)))
                pname = best.get("policy_name", "policy")
                active_skills.append(f"{pname}_Gen{best.get('generation', 0)}")
                skill_weights = best.get("mix_weights") or {pname: 1.0}
            except Exception:
                pass

        metrics = ResearchMetrics(
            timestamp=now,
            epistemic_value=wm_kl,
            world_model_surprise=wm_recon,
            kl_divergence=wm_kl,
            agent_score=agent_score,
            random_baseline=random_baseline,
            scripted_baseline=scripted_baseline,
            human_world_record=human_wr,
            previous_best=previous_best,
            win_probability=min(survival_time / 1200.0, 1.0),
            active_skills=active_skills,
            skill_weights=skill_weights,
            gradient_norm=grad_norm,
            policy_entropy=entropy,
            value_loss=value_loss,
            policy_loss=policy_loss,
            vram_usage_gb=vram_usage_gb,
            vram_total_gb=vram_total_gb or 32.0,
            gpu_utilization=gpu_util,
            tensor_core_utilization=tensor_util,
            fp8_saturation=fp8_sat,
            samples_per_second=tps if tps > 0 else (self.step_count / elapsed if elapsed > 0 else 0.0),
            ultimate_goal="",
            ultimate_progress=0.0,
            tactical_goal="",
            tactical_progress=0.0,
            reasoning_trace=self._latest_reasoning(orch_workers),
        )
        return metrics

    def _poll_learner(self) -> Dict[str, Any]:
        if not (self.learner_url and HAS_REQUESTS):
            return {}
        params = {"policy_name": self.policy_name} if self.policy_name else None
        data = self._get_json(f"{self.learner_url}/metrics", params=params)
        if not isinstance(data, dict):
            return {}
        # If not filtered, choose best available.
        if self.policy_name and self.policy_name in data:
            return data[self.policy_name] if isinstance(data[self.policy_name], dict) else {}
        if "policy_loss" in data or "wm_kl" in data:
            return data
        try:
            # dict of policies -> pick one with latest ts
            best = max(
                (v for v in data.values() if isinstance(v, dict)),
                key=lambda x: float(x.get("ts", 0.0)),
            )
            return best
        except Exception:
            return {}

    def _poll_workers(self) -> Dict[str, Dict[str, Any]]:
        if not (self.orch_url and HAS_REQUESTS):
            return {}
        data = self._get_json(f"{self.orch_url}/workers")
        return data if isinstance(data, dict) else {}

    def _poll_pbt(self) -> Dict[str, Dict[str, Any]]:
        if not (self.orch_url and HAS_REQUESTS):
            return {}
        data = self._get_json(f"{self.orch_url}/pbt")
        return data if isinstance(data, dict) else {}

    def _poll_events(self) -> None:
        if not (self.orch_url and HAS_REQUESTS):
            return
        evs = self._get_json(f"{self.orch_url}/events", params={"limit": 200})
        if not isinstance(evs, list):
            return
        for ev in evs:
            try:
                ts = float(ev.get("ts", 0.0))
                if ts <= self._last_events_ts:
                    continue
                etype = str(ev.get("event_type", ""))
                msg = str(ev.get("message", ""))
                if any(k in etype.lower() for k in ("eureka", "glitch", "newbest", "record")):
                    self._pending_eureka.append(
                        EurekaEvent(
                            timestamp=ts,
                            hypothesis=msg or etype,
                            evidence=str(ev.get("payload", {})),
                            confidence=0.7,
                        )
                    )
                self._last_events_ts = max(self._last_events_ts, ts)
            except Exception:
                continue

    def _latest_reasoning(self, workers: Dict[str, Dict[str, Any]]) -> str:
        # Prefer most recent reasoning/trace field if workers provide it.
        best = ""
        best_ts = 0.0
        for hb in workers.values():
            try:
                ts = float(hb.get("ts", 0.0))
                trace = hb.get("reasoning_trace") or hb.get("reasoning") or hb.get("trace")
                if trace and ts >= best_ts:
                    best = str(trace)
                    best_ts = ts
            except Exception:
                pass
        return best

    def _get_json(self, url: str, params: Optional[dict] = None):
        try:
            resp = requests.get(url, params=params, timeout=self.timeout_s)
            if resp.status_code != 200:
                return {}
            return resp.json()
        except Exception:
            return {}

    # --------------------------- Baselines / Rerun ---------------------------

    def _load_baselines(self, path: str) -> None:
        try:
            import json as _json
            p = path
            with open(p, "r", encoding="utf-8") as f:
                if p.endswith((".yml", ".yaml")):
                    try:
                        import yaml  # type: ignore
                        data = yaml.safe_load(f)
                    except Exception:
                        data = {}
                else:
                    data = _json.load(f)
            if isinstance(data, dict):
                for k in self.baselines:
                    if k in data:
                        self.baselines[k] = float(data[k])
        except Exception:
            return

    def _maybe_log_rerun(self, metrics: ResearchMetrics) -> None:
        if not self.rerun_enabled:
            return
        try:
            if not self._rr_started:
                rr.init("metabonk_rcc", spawn=False)
                self._rr_started = True
            rr.log("rcc/epistemic_value", rr.Scalar(metrics.epistemic_value))
            rr.log("rcc/agent_score", rr.Scalar(metrics.agent_score))
            rr.log("rcc/policy_entropy", rr.Scalar(metrics.policy_entropy))
            rr.log("rcc/grad_norm", rr.Scalar(metrics.gradient_norm))
            rr.log("rcc/vram_usage_gb", rr.Scalar(metrics.vram_usage_gb))
            rr.log("rcc/gpu_utilization", rr.Scalar(metrics.gpu_utilization))
            rr.log("rcc/samples_per_second", rr.Scalar(metrics.samples_per_second))
        except Exception:
            return
