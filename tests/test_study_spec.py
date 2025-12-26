from __future__ import annotations

from pathlib import Path

import pytest

from src.common.study import load_study, slugify


def test_slugify_basic():
    assert slugify("Baseline") == "baseline"
    assert slugify("lr=1e-4") == "lr-1e-4"
    assert slugify("  ") == "variant"


def test_load_study_parses_yaml(tmp_path: Path):
    p = tmp_path / "study.yaml"
    p.write_text(
        "\n".join(
            [
                "study_id: gold-smoke",
                "title: Gold Smoke",
                "experiment_id: exp-omega",
                "base_env:",
                "  METABONK_SYNTHETIC_EYE: '1'",
                "base_args: ['--mode', 'train', '--workers', '1']",
                "variants:",
                "  - name: baseline",
                "    duration_s: 5",
            ]
        )
    )
    s = load_study(p)
    assert s.study_id == "gold-smoke"
    assert s.experiment_id == "exp-omega"
    assert s.variants and s.variants[0].name == "baseline"
    assert s.variants[0].duration_s == 5


def test_load_study_rejects_duplicate_variant_slugs(tmp_path: Path):
    p = tmp_path / "study.yaml"
    p.write_text(
        "\n".join(
            [
                "study_id: dup",
                "variants:",
                "  - name: A/B",
                "  - name: a b",
            ]
        )
    )
    with pytest.raises(ValueError):
        _ = load_study(p)

