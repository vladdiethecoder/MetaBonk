from __future__ import annotations

from typing import Dict

import pytest


def test_federated_merge_ties_smoke() -> None:
    pytest.importorskip("torch", reason="torch required")
    import copy

    import torch
    import torch.nn as nn

    from src.hive.federated_merge import FederatedMerge

    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    specialists: Dict[str, nn.Module] = {}
    for i, role in enumerate(["Scout", "Speedrunner", "Killer", "Tank"]):
        m = copy.deepcopy(base)
        # Small parameter noise to simulate specialization.
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        specialists[role] = m

    merger = FederatedMerge(method="ties")
    merged = merger.merge(specialists)
    assert isinstance(merged, nn.Module)

    x = torch.randn(4, 16)
    out = merged(x)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_federated_merge_avg_similarity() -> None:
    pytest.importorskip("torch", reason="torch required")
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from src.hive.federated_merge import FederatedMerge

    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    specialists: Dict[str, nn.Module] = {}
    for role in ["Scout", "Speedrunner", "Killer", "Tank"]:
        m = copy.deepcopy(base)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.02 * torch.randn_like(p))
        specialists[role] = m

    merged = FederatedMerge(method="avg").merge(specialists)
    x = torch.randn(1, 16)
    y0 = specialists["Scout"](x)
    y = merged(x)
    sim = F.cosine_similarity(y0, y, dim=-1).item()
    assert sim > 0.5
