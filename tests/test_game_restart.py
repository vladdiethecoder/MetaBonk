from __future__ import annotations

import os

from src.worker.main import _game_restart_possible


def test_game_restart_possible_env():
    prev = dict(os.environ)
    try:
        for k in (
            "MEGABONK_CMD",
            "MEGABONK_COMMAND",
            "MEGABONK_CMD_TEMPLATE",
            "MEGABONK_COMMAND_TEMPLATE",
        ):
            os.environ.pop(k, None)
        assert _game_restart_possible() is False

        os.environ["MEGABONK_CMD"] = "echo test"
        assert _game_restart_possible() is True
    finally:
        os.environ.clear()
        os.environ.update(prev)
