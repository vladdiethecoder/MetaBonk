from __future__ import annotations

from typing import Any, Dict

from src.meta import ArchitectureOptimizer


def test_architecture_search_beats_first_trial_baseline() -> None:
    opt = ArchitectureOptimizer(
        seed=0,
        search_space={
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 3, 4],
        },
    )

    def eval_cfg(cfg: Dict[str, Any]) -> float:
        # Prefer larger hidden, fewer layers (toy proxy).
        return float(cfg["hidden_dim"] - 50 * cfg["num_layers"])

    res = opt.search(eval_cfg, budget_trials=12)
    first = float(res.history[0]["score"])
    assert res.best_score >= first

