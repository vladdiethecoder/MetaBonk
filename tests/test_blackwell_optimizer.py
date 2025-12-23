from __future__ import annotations


def test_blackwell_optimizer_no_crash_cpu():
    from src.neuro_genie.blackwell_optimizer import BlackwellOptimConfig, apply_blackwell_defaults, maybe_compile

    cfg = BlackwellOptimConfig(enable_tf32=True, cudnn_benchmark=True, matmul_precision="high", alloc_conf="")
    applied = apply_blackwell_defaults(cfg)
    assert isinstance(applied, dict)

    # maybe_compile should be best-effort and not crash on CPU.
    import torch

    m = torch.nn.Linear(4, 4)
    m2 = maybe_compile(m, enabled=False)
    assert m2 is m

