import torch


def test_center_crop_aspect_chw_square_crops_width():
    from src.worker.frame_normalizer import center_crop_aspect_chw

    x = torch.zeros((3, 100, 200), dtype=torch.uint8)
    y = center_crop_aspect_chw(x, 1.0)
    assert tuple(y.shape) == (3, 100, 100)


def test_normalize_spectator_produces_16_9_shape():
    from src.worker.frame_normalizer import normalize_spectator_u8_chw

    x = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
    y = normalize_spectator_u8_chw(x, out_h=540, out_w=960)
    assert tuple(y.shape) == (3, 540, 960)
    assert y.dtype == torch.uint8
    assert y.is_contiguous()

