from __future__ import annotations

from pathlib import Path


def test_async_checkpointer_writes(tmp_path: Path):
    import torch

    from src.training.async_checkpoint import AsyncCheckpointer

    ckpt = AsyncCheckpointer(str(tmp_path))
    ckpt.save_async({"x": torch.ones((2, 3))}, "a.pt")
    ckpt.wait_all()
    ckpt.stop()
    assert (tmp_path / "a.pt").exists()


def test_batch_size_tuner_cpu_no_crash():
    import torch

    from src.training.batch_size_tuner import BatchSizeTuner

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    x = torch.zeros((1, 4))
    y = torch.zeros((1, 4))
    tuner = BatchSizeTuner(model, x, y, min_batch=1, max_batch=8, target_vram_gb=1.0)
    res = tuner.find()
    assert res.batch_size >= 1


def test_apply_gradient_checkpointing_skips_when_no_layers_attr():
    import torch

    from src.training.gradient_checkpoint import apply_gradient_checkpointing

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert apply_gradient_checkpointing(model, layer_attr="layers") is False

