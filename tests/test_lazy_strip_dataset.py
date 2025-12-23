import torch
from pathlib import Path

from src.training.lazy_strip_dataset import LazyStripDataset


def _write_traj(path: Path, length: int = 8):
    obs = torch.randint(0, 255, (length, 3, 16, 16), dtype=torch.uint8)
    actions = torch.randn(length, 4)
    rewards = torch.zeros(length)
    dones = torch.zeros(length, dtype=torch.bool)
    dones[-1] = True
    torch.save(
        {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        },
        path,
    )


def test_lazy_strip_dataset(tmp_path: Path):
    _write_traj(tmp_path / "a.pt", length=8)
    ds = LazyStripDataset(pt_dir=str(tmp_path), strip_length=4, max_strip_length=8, overlap=0, cache_size=2)
    sample = ds[0]
    assert sample["observations"].shape[0] == 4
    assert sample["observations"].shape[1] == 3
    assert sample["valid_mask"].shape[0] == 4
