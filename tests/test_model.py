import pytest

from fixed_noise_diffusion.model import build_model


def _config(image_size: int):
    return {
        "data": {"channels": 3, "image_size": image_size},
        "model": {
            "base_channels": 8,
            "channel_mults": [1, 2, 2, 4],
            "time_emb_dim": 32,
            "dropout": 0.0,
        },
    }


def test_build_model_rejects_image_size_not_divisible_by_downsampling_factor():
    with pytest.raises(ValueError, match="downsampling factor"):
        build_model(_config(30))


def test_build_model_accepts_packaged_image_size():
    model = build_model(_config(32))

    assert model is not None
