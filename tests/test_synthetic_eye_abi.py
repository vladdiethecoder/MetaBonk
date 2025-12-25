import os
import socket
import struct

import pytest


def _send_with_fds(sock: socket.socket, data: bytes, fds: list[int]) -> None:
    anc = []
    if fds:
        packed = struct.pack(f"<{len(fds)}i", *fds)
        anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, packed)]
    sock.sendmsg([data], anc)


def test_frame_abi_roundtrip_frame_header_and_fds():
    from src.worker import frame_abi

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Dummy fds to pass (pipe ends).
        r0, w0 = os.pipe()
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()
        try:
            dmabuf_fd = r0
            acquire_fd = r1
            release_fd = r2

            # Frame payload: 1 dmabuf fd, 1 plane.
            payload = struct.pack(
                "<QIIIQBBH",
                123,  # frame_id
                1280,  # w
                720,  # h
                875713112,  # drm_fourcc (dummy)
                0,  # modifier
                1,  # dmabuf_fd_count
                1,  # plane_count
                0,  # reserved
            )
            payload += struct.pack(
                "<BBHIIII",
                0,  # fd_index
                0,
                0,
                5120,  # stride
                0,  # offset
                4096,  # size_bytes
                0,
            )
            hdr = frame_abi._HDR_STRUCT.pack(  # type: ignore[attr-defined]
                frame_abi.MAGIC,
                frame_abi.VERSION,
                frame_abi.MSG_FRAME,
                len(payload),
                3,  # 1 dmabuf + acquire + release
            )
            _send_with_fds(b, hdr + payload, [dmabuf_fd, acquire_fd, release_fd])

            got_hdr, fds0 = frame_abi._recv_header_with_fds(a)  # type: ignore[attr-defined]
            magic, ver, msg_type, payload_len, fd_count = frame_abi._HDR_STRUCT.unpack(got_hdr)  # type: ignore[attr-defined]
            assert magic == frame_abi.MAGIC
            assert ver == frame_abi.VERSION
            assert msg_type == frame_abi.MSG_FRAME
            payload2 = frame_abi._recv_exact(a, int(payload_len))  # type: ignore[attr-defined]
            assert len(fds0) >= int(fd_count)
            fr = frame_abi._parse_frame_v1(payload2, fds0[: int(fd_count)])  # type: ignore[attr-defined]
            assert fr.frame_id == 123
            assert fr.width == 1280
            assert fr.height == 720
            assert fr.dmabuf_fds and len(fr.dmabuf_fds) == 1
            assert fr.acquire_fence_fd >= 0
            assert fr.release_fence_fd >= 0
        finally:
            for fd in (r0, w0, r1, w1, r2, w2):
                try:
                    os.close(fd)
                except Exception:
                    pass
    finally:
        a.close()
        b.close()


def test_frame_abi_reset_parses():
    from src.worker import frame_abi

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        payload = struct.pack("<II", 1, 0)
        hdr = frame_abi._HDR_STRUCT.pack(  # type: ignore[attr-defined]
            frame_abi.MAGIC,
            frame_abi.VERSION,
            frame_abi.MSG_RESET,
            len(payload),
            0,
        )
        b.sendall(hdr + payload)

        got_hdr, fds0 = frame_abi._recv_header_with_fds(a)  # type: ignore[attr-defined]
        _magic, _ver, _typ, payload_len, fd_count = frame_abi._HDR_STRUCT.unpack(got_hdr)  # type: ignore[attr-defined]
        payload2 = frame_abi._recv_exact(a, int(payload_len))  # type: ignore[attr-defined]
        assert int(fd_count) == 0
        assert not fds0
        rst = frame_abi._parse_reset_v1(payload2)  # type: ignore[attr-defined]
        assert rst.reason == 1
    finally:
        a.close()
        b.close()
