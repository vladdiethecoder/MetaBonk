"""Synthetic Eye frame handoff ABI (v1).

Python reference implementation for docs/synthetic_eye_abi.md.
"""

from __future__ import annotations

import os
import socket
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

MAGIC = b"MBEYEABI"
VERSION = 1

MSG_HELLO = 1
MSG_HELLO_ACK = 2
MSG_FRAME = 3
MSG_RESET = 4
MSG_PING = 5
MSG_PONG = 6

_HDR_STRUCT = struct.Struct("<8sHHII")
_MAX_FDS_PER_MSG = 16


@dataclass(frozen=True)
class PlaneV1:
    fd_index: int
    stride: int
    offset: int
    size_bytes: int = 0


@dataclass
class FrameV1:
    frame_id: int
    width: int
    height: int
    drm_fourcc: int
    modifier: int
    dmabuf_fds: List[int]
    planes: List[PlaneV1]
    acquire_fence_fd: int
    release_fence_fd: int

    def close(self) -> None:
        for fd in list(self.dmabuf_fds) + [self.acquire_fence_fd, self.release_fence_fd]:
            try:
                os.close(int(fd))
            except Exception:
                pass


@dataclass(frozen=True)
class ResetV1:
    reason: int


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("socket closed")
        out += chunk
    return bytes(out)

def _recv_header_with_fds(sock: socket.socket) -> Tuple[bytes, List[int]]:
    """Receive exactly one HeaderV1 and capture any SCM_RIGHTS FDs attached to it.

    For SOCK_STREAM, SCM_RIGHTS is delivered with the *first* recvmsg() that reads
    bytes from the corresponding sendmsg(). If you read the header with recv(),
    you will lose the attached FDs.
    """
    cmsg_space = socket.CMSG_SPACE(_MAX_FDS_PER_MSG * struct.calcsize("i"))
    data = bytearray()
    fds: List[int] = []
    while len(data) < _HDR_STRUCT.size:
        need = _HDR_STRUCT.size - len(data)
        chunk, ancdata, _msg_flags, _addr = sock.recvmsg(need, cmsg_space)
        if not chunk:
            raise EOFError("socket closed")
        data += chunk
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                got = list(struct.unpack(f"<{len(cmsg_data)//4}i", cmsg_data))
                fds.extend(int(x) for x in got)
    return bytes(data), fds


def _recv_msg_with_fds(sock: socket.socket, data_len: int, fd_count: int) -> Tuple[bytes, List[int]]:
    """Receive (payload_bytes, fds) for a message using recvmsg + SCM_RIGHTS."""
    max_fds = max(0, int(fd_count))
    cmsg_space = socket.CMSG_SPACE(max_fds * struct.calcsize("i"))
    payload = bytearray()
    fds: List[int] = []
    while len(payload) < data_len:
        need = data_len - len(payload)
        chunk, ancdata, _msg_flags, _addr = sock.recvmsg(need, cmsg_space)
        if not chunk:
            raise EOFError("socket closed while receiving payload")
        payload += chunk
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                got = list(struct.unpack(f"<{len(cmsg_data)//4}i", cmsg_data))
                fds.extend(int(x) for x in got)
    if len(fds) < max_fds:
        raise RuntimeError(f"expected {max_fds} fds, got {len(fds)}")
    return bytes(payload), fds[:max_fds]


def _recv_payload_with_optional_fds(sock: socket.socket, data_len: int) -> Tuple[bytes, List[int]]:
    """Receive payload bytes and collect any SCM_RIGHTS FDs delivered with them.

    Some implementations may deliver SCM_RIGHTS with the first recvmsg() that
    reads bytes from the corresponding sendmsg(). While we expect the header
    recvmsg() to collect all FDs, be robust and collect any that arrive with
    the payload chunks as well.
    """
    cmsg_space = socket.CMSG_SPACE(_MAX_FDS_PER_MSG * struct.calcsize("i"))
    payload = bytearray()
    fds: List[int] = []
    while len(payload) < data_len:
        need = data_len - len(payload)
        chunk, ancdata, _msg_flags, _addr = sock.recvmsg(need, cmsg_space)
        if not chunk:
            raise EOFError("socket closed while receiving payload")
        payload += chunk
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                got = list(struct.unpack(f"<{len(cmsg_data)//4}i", cmsg_data))
                fds.extend(int(x) for x in got)
    return bytes(payload), fds


def _parse_frame_v1(payload: bytes, fds: List[int]) -> FrameV1:
    fixed = struct.Struct("<QIIIQBBH")
    if len(payload) < fixed.size:
        raise ValueError("frame payload too short")
    (
        frame_id,
        width,
        height,
        drm_fourcc,
        modifier,
        dmabuf_fd_count,
        plane_count,
        _reserved0,
    ) = fixed.unpack_from(payload, 0)
    dmabuf_fd_count = int(dmabuf_fd_count)
    plane_count = int(plane_count)
    min_fds = dmabuf_fd_count
    max_fds = dmabuf_fd_count + 2
    if len(fds) < min_fds or len(fds) > max_fds:
        raise ValueError(f"frame fd_count mismatch: expected {min_fds}..{max_fds}, got {len(fds)}")
    dmabuf_fds = [int(x) for x in fds[:dmabuf_fd_count]]
    acquire_fence_fd = -1
    release_fence_fd = -1
    if len(fds) >= dmabuf_fd_count + 1:
        acquire_fence_fd = int(fds[dmabuf_fd_count])
    if len(fds) >= dmabuf_fd_count + 2:
        release_fence_fd = int(fds[dmabuf_fd_count + 1])

    planes: List[PlaneV1] = []
    plane_struct = struct.Struct("<BBHIIII")
    off = fixed.size
    for _ in range(plane_count):
        if off + plane_struct.size > len(payload):
            raise ValueError("frame plane payload too short")
        fd_index, _r0, _r1, stride, offset, size_bytes, _r2 = plane_struct.unpack_from(payload, off)
        planes.append(
            PlaneV1(
                fd_index=int(fd_index),
                stride=int(stride),
                offset=int(offset),
                size_bytes=int(size_bytes),
            )
        )
        off += plane_struct.size
    return FrameV1(
        frame_id=int(frame_id),
        width=int(width),
        height=int(height),
        drm_fourcc=int(drm_fourcc),
        modifier=int(modifier),
        dmabuf_fds=dmabuf_fds,
        planes=planes,
        acquire_fence_fd=acquire_fence_fd,
        release_fence_fd=release_fence_fd,
    )


def _parse_reset_v1(payload: bytes) -> ResetV1:
    st = struct.Struct("<II")
    if len(payload) < st.size:
        raise ValueError("reset payload too short")
    reason, _r0 = st.unpack_from(payload, 0)
    return ResetV1(reason=int(reason))


class FrameSocketClient:
    def __init__(self, path: str, *, connect_timeout_s: float = 5.0) -> None:
        self.path = str(path)
        self.connect_timeout_s = float(connect_timeout_s)
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout_s)
        s.connect(self.path)
        s.settimeout(None)
        self.sock = s
        self.send_hello()

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None

    def send_hello(self) -> None:
        if self.sock is None:
            return
        hdr = _HDR_STRUCT.pack(MAGIC, VERSION, MSG_HELLO, 0, 0)
        self.sock.sendall(hdr)

    def recv(self) -> Tuple[str, object]:
        if self.sock is None:
            raise RuntimeError("not connected")
        hdr, fds = _recv_header_with_fds(self.sock)
        magic, ver, msg_type, payload_len, fd_count = _HDR_STRUCT.unpack(hdr)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if int(ver) != VERSION:
            raise ValueError(f"unsupported abi version: {ver}")
        payload, extra_fds = _recv_payload_with_optional_fds(self.sock, int(payload_len))
        want = int(fd_count)
        if extra_fds:
            fds.extend(extra_fds)
        if want > len(fds):
            raise RuntimeError(f"expected {want} fds, got {len(fds)}")
        fds = fds[:want]
        if int(msg_type) == MSG_FRAME:
            return "frame", _parse_frame_v1(payload, fds)
        if int(msg_type) == MSG_RESET:
            return "reset", _parse_reset_v1(payload)
        if int(msg_type) == MSG_PONG:
            return "pong", None
        if int(msg_type) == MSG_HELLO_ACK:
            return "hello_ack", None
        if int(msg_type) == MSG_PING:
            return "ping", None
        return "unknown", (int(msg_type), payload, fds)
