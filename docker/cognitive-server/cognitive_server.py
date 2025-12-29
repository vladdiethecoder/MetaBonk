#!/usr/bin/env python3
"""
MetaBonk Centralized Cognitive Server.

Runs SGLang with Phi-3-Vision for multi-instance System 2 reasoning.
Communication:
- Receives JSON requests over ZeroMQ ROUTER from multiple game instances (DEALER).
- Returns JSON strategic directives.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
import sglang as sgl
from sglang import assistant, gen, image, system, user

try:
    # sglang 0.2.x
    from sglang.backend.runtime_endpoint import RuntimeEndpoint  # type: ignore
except Exception:  # pragma: no cover
    RuntimeEndpoint = None  # type: ignore

from rl_integration import RLLogger
from temporal_processor import TemporalFrameProcessor
from zmq_bridge import ZeroMQBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("metabonk.cognitive_server")


METABONK_SYSTEM_PROMPT = """You are a strategic AI agent playing MetaBonk, a fast-paced roguelike game.

**Your Role**: Provide high-level strategic directives based on visual analysis.

**Visual Input**: You will see 9 frames:
- 5 past frames (showing recent history)
- 1 current frame (now)
- 3 predicted future frames (what will likely happen)

**Output Format** (strict JSON):
```json
{
  "reasoning": "Brief analysis of current situation",
  "goal": "immediate_objective",
  "strategy": "high_level_plan",
  "directive": {
    "action": "move|attack|defend|retreat|collect",
    "target": [x, y],
    "duration_seconds": 3.0,
    "priority": "critical|high|medium|low"
  },
  "confidence": 0.85
}
```

**Game Rules**:
- Survive as long as possible
- Collect experience orbs (blue circles)
- Avoid damage from enemies (red entities)
- Use terrain strategically (corners, bottlenecks)
- Prioritize staying alive over offensive actions

**Strategic Principles**:
1. If health < 30%: RETREAT to safe zone
2. If enemies > 10: Find BOTTLENECK or CORNER
3. If clear path: COLLECT nearby orbs
4. If boss visible: KITE (attack while retreating)
5. Always maintain ESCAPE ROUTE

Analyze the temporal frames to understand motion and predict threats.
"""


@sgl.function
def metabonk_strategist(s, temporal_frames: List[str], agent_state: Dict[str, Any]):
    # System prompt (cached via RadixAttention)
    s += system(METABONK_SYSTEM_PROMPT)

    # User query with temporal context
    s += user(
        [
            f"**Agent State**: {json.dumps(agent_state)}",
            "",
            "**Temporal Frames** (past → present → future):",
            *[image(frame_b64) for frame_b64 in temporal_frames],
            "",
            "**Task**: Analyze the temporal sequence and provide strategic directive.",
        ]
    )

    # Generate response with a loose JSON guard (SGLang can also do stricter schema decoding).
    s += assistant(
        gen(
            "response",
            max_tokens=512,
            temperature=0.7,
            regex=r"\\{.*\\}",
        )
    )


class CognitiveServer:
    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        quantization: str,
        kv_cache_dtype: str,
        zmq_port: int = 5555,
        max_running_requests: int = 64,
        rl_log_dir: Optional[str] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.zmq_port = int(zmq_port)

        if RuntimeEndpoint is None:
            raise RuntimeError("sglang RuntimeEndpoint import failed; ensure sglang is installed in the container")

        logger.info("Initializing SGLang runtime...")
        self.runtime = RuntimeEndpoint(
            model_path=self.model_path,
            tp_size=int(tp_size),
            quantization=str(quantization),
            kv_cache_dtype=str(kv_cache_dtype),
            max_running_requests=int(max_running_requests),
            enable_flashinfer=True,
            disable_radix_cache=False,
        )

        self.temporal_processor = TemporalFrameProcessor()
        self.zmq_bridge = ZeroMQBridge(port=self.zmq_port)

        self._sema = asyncio.Semaphore(int(max_running_requests))
        self._tasks: set[asyncio.Task] = set()

        self.rl_logger: Optional[RLLogger] = RLLogger(rl_log_dir) if rl_log_dir else None

        self.request_count = 0
        self.total_latency_s = 0.0
        self.active_agents: set[str] = set()

        logger.info("✅ Cognitive Server initialized (model=%s, zmq_port=%d)", self.model_path, self.zmq_port)

    def _normalize_frames(self, frames: List[str]) -> List[str]:
        # Ensure exactly 9 frames.
        frames = list(frames or [])
        if not frames:
            return frames
        if len(frames) < 9:
            frames = frames + [frames[-1]] * (9 - len(frames))
        elif len(frames) > 9:
            frames = frames[-9:]
        return frames

    def _run_inference(self, *, frames: List[str], agent_state: Dict[str, Any]) -> str:
        result = metabonk_strategist.run(
            temporal_frames=frames,
            agent_state=agent_state,
            backend=self.runtime,
        )
        return str(result.get("response") or "")

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        agent_id = str(request_data.get("agent_id") or "unknown")
        self.active_agents.add(agent_id)

        try:
            frames = self._normalize_frames(list(request_data.get("frames") or []))
            agent_state = dict(request_data.get("state") or {})

            # Optional temporal preprocessing hook (currently unused by prompt).
            try:
                _ = self.temporal_processor.process(frames)
            except Exception:
                pass

            # Run SGLang inference in a worker thread so the asyncio loop can keep polling ZMQ.
            response_text = await asyncio.to_thread(self._run_inference, frames=frames, agent_state=agent_state)

            # Extract JSON object from response.
            try:
                strategy: Dict[str, Any] = json.loads(response_text)
            except json.JSONDecodeError:
                m = re.search(r"\\{.*\\}", response_text, re.DOTALL)
                if not m:
                    raise ValueError(f"Could not parse JSON from response: {response_text[:2000]}")
                strategy = json.loads(m.group(0))

            strategy["agent_id"] = agent_id
            strategy["inference_time_ms"] = (time.time() - start) * 1000.0
            strategy["timestamp"] = time.time()

            self.request_count += 1
            self.total_latency_s += (time.time() - start)

            if self.rl_logger is not None:
                try:
                    self.rl_logger.log_decision(agent_id=agent_id, request_data=request_data, response_data=strategy)
                except Exception:
                    pass

            logger.info(
                "%s: directive generated (%.1fms) goal=%s",
                agent_id,
                float(strategy.get("inference_time_ms", 0.0)),
                strategy.get("goal", "unknown"),
            )

            return strategy

        except Exception as e:
            logger.error("%s: error processing request: %s", agent_id, e)
            return {
                "agent_id": agent_id,
                "reasoning": f"Error: {e}",
                "goal": "survive",
                "strategy": "defensive",
                "directive": {
                    "action": "retreat",
                    "target": [0, 0],
                    "duration_seconds": 2.0,
                    "priority": "high",
                },
                "confidence": 0.0,
                "error": True,
                "inference_time_ms": (time.time() - start) * 1000.0,
                "timestamp": time.time(),
            }

    async def _handle_one(self, identity: bytes, request_data: Dict[str, Any]) -> None:
        async with self._sema:
            response = await self.process_request(request_data)
            await self.zmq_bridge.send(identity, response)

    async def run(self) -> None:
        logger.info("🚀 Cognitive Server starting (ZMQ port %d)...", self.zmq_port)
        await self.zmq_bridge.start()

        try:
            while True:
                req = await self.zmq_bridge.receive(timeout_ms=100)
                if req is None:
                    await asyncio.sleep(0.001)
                    continue

                identity, request_data = req
                task = asyncio.create_task(self._handle_one(identity, request_data))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

                if self.request_count and (self.request_count % 10 == 0):
                    avg_ms = (self.total_latency_s / max(1, self.request_count)) * 1000.0
                    logger.info(
                        "📊 Metrics: %d requests, %d agents, avg latency %.1fms",
                        self.request_count,
                        len(self.active_agents),
                        avg_ms,
                    )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Drain pending tasks.
            if self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await self.zmq_bridge.stop()
            logger.info("✅ Cognitive Server stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaBonk Cognitive Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--quantization", type=str, default="awq")
    parser.add_argument("--kv-cache-dtype", type=str, default="fp8")
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--rl-log-dir", type=str, default=os.environ.get("METABONK_RL_LOG_DIR"))

    args = parser.parse_args()

    server = CognitiveServer(
        model_path=args.model_path,
        tp_size=args.tp_size,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        zmq_port=args.zmq_port,
        max_running_requests=args.max_running_requests,
        rl_log_dir=args.rl_log_dir,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
