import numpy as np

import pytest


torch = pytest.importorskip("torch")


def test_metabonk2_controller_step_shapes():
    from src.metabonk2.controller import MetaBonk2Controller, MetaBonk2ControllerConfig

    specs = [
        {"kind": "key", "name": "W"},
        {"kind": "key", "name": "A"},
        {"kind": "key", "name": "S"},
        {"kind": "key", "name": "D"},
        {"kind": "mouse", "button": "LEFT"},
    ]
    ctrl = MetaBonk2Controller(button_specs=specs, cfg=MetaBonk2ControllerConfig(log_reasoning=False))

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    a_cont, a_disc = ctrl.step(frame, {}, time_budget_ms=10.0)

    assert isinstance(a_cont, list)
    assert isinstance(a_disc, list)
    assert len(a_cont) == 2
    assert len(a_disc) == len(specs)
    assert all(isinstance(x, (int, float)) for x in a_cont)
    assert all(int(x) in (0, 1) for x in a_disc)

    st = ctrl.get_status()
    assert isinstance(st, dict)
    assert st.get("enabled") is True

