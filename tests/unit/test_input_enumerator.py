from __future__ import annotations

from src.discovery import InputEnumerator


def test_input_enumerator_schema() -> None:
    spec = InputEnumerator().get_input_space_spec()
    assert "keyboard" in spec
    assert "mouse" in spec
    assert "discovered_at" in spec
    assert "source" in spec
    assert "warnings" in spec
    assert isinstance(spec["keyboard"]["available_keys"], list)
    assert isinstance(spec["keyboard"]["total_keys"], int)
    assert spec["keyboard"]["total_keys"] == len(spec["keyboard"]["available_keys"])


def test_input_enumerator_is_best_effort_and_repeatable() -> None:
    e1 = InputEnumerator().get_input_space_spec()
    e2 = InputEnumerator().get_input_space_spec()
    # In some CI environments enumeration may be empty; we only assert stability of schema.
    assert set(e1.keys()) == set(e2.keys())
    if e1["keyboard"]["total_keys"] > 0:
        assert set(e1["keyboard"]["available_keys"]) == set(e2["keyboard"]["available_keys"])

