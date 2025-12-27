from __future__ import annotations

import pytest


def test_universal_game_encoder_basic_forward() -> None:
    torch = pytest.importorskip("torch", reason="torch required")

    from src.agent.generalization.universal_encoder import UniversalGameEncoder

    enc = UniversalGameEncoder(output_dim=64)
    frames = torch.zeros((2, 3, 64, 64), dtype=torch.uint8)
    z1 = enc(frames, game_id="megabonk")
    z2 = enc(frames, game_id="factorio")

    assert z1.shape == (2, 64)
    assert z2.shape == (2, 64)
    assert torch.isfinite(z1).all()
    assert torch.isfinite(z2).all()


def test_universal_game_encoder_creates_adapter_for_new_game() -> None:
    torch = pytest.importorskip("torch", reason="torch required")

    from src.agent.generalization.universal_encoder import UniversalGameEncoder

    enc = UniversalGameEncoder(output_dim=32)
    frames = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    z = enc(frames, game_id="new_game")
    assert z.shape == (1, 32)

