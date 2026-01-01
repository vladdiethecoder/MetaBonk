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


def test_normalize_obs_cutile_backend_smoke():
    from src.worker.frame_normalizer import normalize_obs_u8_chw
    from src.worker.gpu_preprocess import HAS_CUTILE

    x = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)

    if HAS_CUTILE and torch.cuda.is_available():
        y = normalize_obs_u8_chw(x.cuda(), out_h=128, out_w=128, backend="cutile")
        assert tuple(y.shape) == (3, 128, 128)
        assert y.dtype == torch.uint8
        assert y.is_cuda
        assert y.is_contiguous()
    else:
        # cuTile backend is GPU-only; in non-CUDA environments it must not silently fall back.
        try:
            _ = normalize_obs_u8_chw(x.cuda() if torch.cuda.is_available() else x, out_h=128, out_w=128, backend="cutile")
        except Exception:
            pass
        else:
            raise AssertionError("Expected cuTile backend to fail when unavailable")
