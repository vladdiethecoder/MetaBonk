from __future__ import annotations

import socket
import struct
import threading
from typing import Optional, Tuple

import pytest


def _encode_7bit_int(value: int) -> bytes:
    v = int(value)
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _encode_dotnet_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return _encode_7bit_int(len(raw)) + raw


def _build_state_bytes(*, menu: str = "None") -> bytes:
    # Matches src/bridge/bonklink_client.py:_parse_state field order.
    header = struct.pack(
        "<9f",
        1.0,
        2.0,
        3.0,  # pos
        0.1,
        0.2,
        0.3,  # vel
        50.0,
        100.0,  # health
        12.5,  # game_time
    )
    flags = struct.pack("<2?", True, False)  # playing, paused
    enemies = struct.pack("<i", 0)
    menu_bytes = _encode_dotnet_string(menu)
    opts = struct.pack("<i", 2) + _encode_dotnet_string("opt_a") + _encode_dotnet_string("opt_b")

    # Optional trailer: 'HINP' + version + payload.
    trailer = struct.pack(
        "<Ii4f4?2f",
        0x504E4948,  # 'HINP' little-endian
        1,
        0.9,
        -0.9,  # input_move
        0.25,
        -0.25,  # input_look
        True,
        False,
        True,
        True,  # fire, ability, interact, ui_click
        0.1,
        0.2,  # click_norm
    )

    return header + flags + enemies + menu_bytes + opts + trailer


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("unexpected EOF")
        buf.extend(chunk)
    return bytes(buf)


def test_testing_spec_bonklink_client_roundtrip_smoke() -> None:
    torch = pytest.importorskip("torch")
    del torch  # unused; align with spec which assumes torch env.

    from src.bridge.bonklink_client import BonkLinkAction, BonkLinkClient

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    received_action: dict[str, Optional[bytes]] = {"payload": None}

    def server() -> None:
        conn: Optional[socket.socket] = None
        try:
            conn, _addr = listener.accept()
            state_bytes = _build_state_bytes(menu="MainMenu")
            frame_bytes = b""
            pkt = (
                struct.pack("<i", len(state_bytes))
                + state_bytes
                + struct.pack("<i", len(frame_bytes))
                + frame_bytes
            )
            conn.sendall(pkt)

            # Read action message: int32 size + payload.
            size = struct.unpack("<i", _read_exact(conn, 4))[0]
            received_action["payload"] = _read_exact(conn, size)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            try:
                listener.close()
            except Exception:
                pass

    t = threading.Thread(target=server, daemon=True)
    t.start()

    client = BonkLinkClient(host=str(host), port=int(port))
    assert client.connect(timeout_s=2.0)
    try:
        pkt: Optional[Tuple[object, Optional[bytes]]] = None
        for _ in range(50):
            pkt = client.read_state_frame(timeout_ms=50)
            if pkt is not None:
                break
        assert pkt is not None
        state, frame = pkt
        assert frame is None
        assert state.current_menu == "MainMenu"
        assert state.level_up_options == ["opt_a", "opt_b"]
        assert state.is_playing is True
        assert state.is_paused is False
        assert state.input_move == pytest.approx((0.9, -0.9))
        assert state.input_ui_click is True
        assert state.input_click_norm == pytest.approx((0.1, 0.2))

        action = BonkLinkAction(move_x=1.0, move_y=-1.0, btn0=True)
        assert client.send_action(action)
    finally:
        client.close()

    t.join(timeout=2.0)
    payload = received_action["payload"]
    assert payload is not None

    # Verify wire payload matches BonkLinkAction wire layout.
    move_x, move_y, look_x, look_y, btn0, btn1, btn2, ui_click = struct.unpack("<4f4?", payload)
    assert move_x == pytest.approx(1.0)
    assert move_y == pytest.approx(-1.0)
    assert look_x == pytest.approx(0.0)
    assert look_y == pytest.approx(0.0)
    assert btn0 is True
    assert btn1 is False
    assert btn2 is False
    assert ui_click is False

