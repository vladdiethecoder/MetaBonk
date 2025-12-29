from __future__ import annotations


def test_instance_config_allows_spectator_size_fields() -> None:
    from src.common.schemas import InstanceConfig

    cfg = InstanceConfig(
        instance_id="omega-0",
        policy_name="Greed",
        hparams={},
        spectator_width=1920,
        spectator_height=1080,
    )
    assert cfg.spectator_width == 1920
    assert cfg.spectator_height == 1080

