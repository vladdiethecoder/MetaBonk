"""Production metrics with optional Prometheus export."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False

try:
    import GPUtil  # type: ignore

    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, start_http_server  # type: ignore

    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False


class MetricsCollector:
    """Collect and export a small set of metrics.

    If prometheus_client isn't installed, this becomes a no-op collector.
    """

    def __init__(self, *, port: int = 9090) -> None:
        self.enabled = bool(_PROM_AVAILABLE)
        if not self.enabled:
            logger.warning("prometheus_client not available; metrics disabled")
            return

        self.actions_discovered = Gauge("metabonk_actions_discovered_total", "Total actions discovered")

        self.training_steps = Counter("metabonk_training_steps_total", "Total training steps", ["agent_id"])
        self.reward_mean = Gauge("metabonk_reward_mean", "Mean reward", ["agent_id"])

        self.gpu_utilization = Gauge("metabonk_gpu_utilization_percent", "GPU utilization", ["gpu_id"])
        self.cpu_percent = Gauge("metabonk_cpu_percent", "CPU utilization")

        try:
            start_http_server(int(port))
            logger.info("✓ Metrics server on port %s", int(port))
        except Exception as e:
            logger.error("Failed to start metrics server: %s", e)

    def record_actions_discovered(self, count: int) -> None:
        if not self.enabled:
            return
        try:
            self.actions_discovered.set(int(count))
        except Exception:
            pass

    def record_training_step(self, *, agent_id: str, reward: float) -> None:
        if not self.enabled:
            return
        try:
            self.training_steps.labels(agent_id=str(agent_id)).inc()
            self.reward_mean.labels(agent_id=str(agent_id)).set(float(reward))
        except Exception:
            pass

    def update_system_metrics(self) -> None:
        if not self.enabled:
            return
        try:
            if _GPU_AVAILABLE:
                for i, gpu in enumerate(GPUtil.getGPUs()):
                    self.gpu_utilization.labels(gpu_id=str(i)).set(float(gpu.load) * 100.0)
            if _PSUTIL_AVAILABLE:
                self.cpu_percent.set(float(psutil.cpu_percent()))
        except Exception as e:
            logger.error("Error updating metrics: %s", e)


__all__ = ["MetricsCollector"]

