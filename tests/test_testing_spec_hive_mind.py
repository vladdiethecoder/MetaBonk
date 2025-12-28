from __future__ import annotations

import pytest


def test_testing_spec_hive_mind_swarm_init_and_ties_merge_cpu() -> None:
    torch = pytest.importorskip("torch")

    from src.orchestrator.hive_mind import SwarmConfig, VirtualSwarm

    cfg = SwarmConfig(
        n_scouts=1,
        n_speedrunners=1,
        n_killers=0,
        n_tanks=0,
        n_builders=0,
        state_dim=4,
        action_dim=2,
        hidden_dim=8,
        device="cpu",
    )
    swarm = VirtualSwarm(cfg)

    assert cfg.total_agents == 2
    stats = swarm.get_swarm_stats()
    assert stats["total_agents"] == 2

    # Make two agents disagree on the same parameter entry to exercise sign election.
    param_name = next(iter(dict(swarm.base_model.named_parameters()).keys()))
    base_param = dict(swarm.base_model.named_parameters())[param_name].detach().clone()

    (_, scout_model) = swarm.get_agent(0)
    (_, speed_model) = swarm.get_agent(1)

    scout_param = dict(scout_model.named_parameters())[param_name]
    speed_param = dict(speed_model.named_parameters())[param_name]

    scout_param.data[0, 0] += 1.0
    speed_param.data[0, 0] -= 1.0

    god = swarm.merge_god_agent(method="ties")
    merged_param = dict(god.named_parameters())[param_name].detach()

    assert torch.isclose(merged_param[0, 0], base_param[0, 0] + 1.0)


def test_testing_spec_ties_merger_smoke() -> None:
    torch = pytest.importorskip("torch")

    from src.learner.task_vectors import TIESMerger, TaskVector, TaskVectorMetadata

    v1 = TaskVector({"w": torch.tensor([1.0, -1.0, 0.5])}, TaskVectorMetadata(name="a"))
    v2 = TaskVector({"w": torch.tensor([2.0, -0.5, -0.2])}, TaskVectorMetadata(name="b"))

    merged = TIESMerger(topk=1.0).merge([v1, v2])

    w = merged.vector["w"]
    assert tuple(w.shape) == (3,)
    assert float(w[0]) == pytest.approx(1.5)
    assert float(w[1]) == pytest.approx(-0.75)
    assert float(w[2]) == pytest.approx(0.0)
    assert merged.metadata.name.startswith("TIES(")

