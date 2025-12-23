import torch

from src.training.masked_video_loss import MaskedVideoLoss
from src.neuro_genie.masked_strip_model import StripEncoder


def test_masked_video_loss_runs():
    obs = torch.rand(2, 4, 3, 16, 16)
    encoder = StripEncoder(in_channels=3, latent_dim=256)
    loss_fn = MaskedVideoLoss(mask_ratio=0.5)
    loss, info = loss_fn(obs, encoder)
    assert torch.isfinite(loss).item()
    assert 0.0 <= info["masked_ratio"] <= 1.0
