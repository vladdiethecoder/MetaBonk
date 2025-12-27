from __future__ import annotations

from typing import Any, Dict

from src.meta import ArchitectureEvolution, ArchitectureOptimizer
from src.meta.performance_predictor import PerformancePredictor


def test_architecture_optimizer_finds_best() -> None:
    opt = ArchitectureOptimizer(seed=0, search_space={"a": [1, 2, 3], "b": [0, 10]})

    def eval_cfg(cfg: Dict[str, Any]) -> float:
        return float(cfg["a"] - cfg["b"])

    res = opt.search(eval_cfg, budget_trials=10)
    assert res.trials == 10
    assert res.best_config["a"] == 3
    assert res.best_config["b"] == 0


def test_architecture_evolution_keeps_population_size() -> None:
    evo = ArchitectureEvolution(seed=0, population_size=6)
    evo.seed_population([{"a": 1, "b": 2}, {"a": 2, "b": 1}])

    def fitness(cfg: Dict[str, Any]) -> float:
        return float(cfg.get("a", 0) + cfg.get("b", 0))

    res = evo.evolve_step(fitness)
    assert len(res.population) == 6
    assert res.best_fitness == max(res.fitness_scores)


def test_performance_predictor_fit_predict() -> None:
    samples = [
        {"config": {"hidden_dim": 128, "num_layers": 2}, "score": 1.0},
        {"config": {"hidden_dim": 256, "num_layers": 2}, "score": 2.0},
        {"config": {"hidden_dim": 128, "num_layers": 4}, "score": 0.5},
    ]
    pred = PerformancePredictor(l2=1e-3)
    fit = pred.fit(samples, feature_keys=["hidden_dim", "num_layers"])
    assert len(fit.weights) == 2
    s = pred.predict({"hidden_dim": 256, "num_layers": 2})
    assert isinstance(s, float)

