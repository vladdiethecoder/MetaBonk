"""
ZeroMQ Bridge for Cognitive Server.

Implements a ROUTER socket for receiving requests from multiple game clients.

Security:
- Uses JSON serialization (NOT pickle) to avoid deserialization vulnerabilities.
- Binds to localhost by default.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)


class ZeroMQBridge:
    """Async ROUTER socket wrapper."""

    def __init__(self, *, port: int = 5555, bind_address: str = "tcp://127.0.0.1") -> None:
        self.port = int(port)
        self.bind_address = str(bind_address)

        self.context = zmq.asyncio.Context()
        self.socket: Optional[zmq.asyncio.Socket] = None

        self.messages_received = 0
        self.messages_sent = 0

    async def start(self) -> None:
        """Initialize and bind ROUTER socket."""
        self.socket = self.context.socket(zmq.ROUTER)

        # Socket options
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.SNDHWM, 1000)

        bind_url = f"{self.bind_address}:{self.port}"
        self.socket.bind(bind_url)
        logger.info("✅ ZMQ ROUTER bound to %s", bind_url)

    async def receive(self, *, timeout_ms: int = 100) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """
        Receive request from a game client (non-blocking with poll).

        For ROUTER <-> DEALER:
          - Expect multipart: [identity, data]
        For ROUTER <-> REQ:
          - Expect multipart: [identity, empty, data]
        """
        if self.socket is None:
            raise RuntimeError("ZMQ bridge not started")

        try:
            if not await self.socket.poll(timeout=timeout_ms):
                return None

            frames = await self.socket.recv_multipart()
            if len(frames) < 2:
                return None

            identity = frames[0]
            data_bytes = frames[-1]

            request_data = json.loads(data_bytes.decode("utf-8"))
            if not isinstance(request_data, dict):
                return None

            self.messages_received += 1
            return identity, request_data

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error receiving message: %s", e)
            return None

    async def send(self, identity: bytes, response_data: Dict[str, Any]) -> None:
        """Send response to a specific client identity."""
        if self.socket is None:
            raise RuntimeError("ZMQ bridge not started")

        try:
            response_bytes = json.dumps(response_data).encode("utf-8")
            # For DEALER clients, send exactly one payload frame (no empty delimiter).
            await self.socket.send_multipart([identity, response_bytes])
            self.messages_sent += 1
        except Exception as e:
            logger.error("Error sending message: %s", e)

    async def stop(self) -> None:
        """Cleanup."""
        try:
            if self.socket is not None:
                self.socket.close()
        finally:
            self.context.term()
        logger.info(
            "✅ ZMQ Bridge stopped (%d received, %d sent)", self.messages_received, self.messages_sent
        )

