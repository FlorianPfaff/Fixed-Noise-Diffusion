import pytest
import torch

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


@pytest.mark.parametrize("time_emb_dim", [1, 2, 3, 4])
def test_build_model_small_time_embedding_dimensions_forward(time_emb_dim):
    config = _config(32)
    config["model"]["base_channels"] = 4
    config["model"]["channel_mults"] = [1, 2]
    config["model"]["time_emb_dim"] = time_emb_dim
    model = build_model(config)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    output = model(images, timesteps)

    assert output.shape == images.shape


def test_build_model_rejects_nonpositive_time_embedding_dim():
    config = _config(32)
    config["model"]["time_emb_dim"] = 0

    with pytest.raises(ValueError, match="time_emb_dim"):
        build_model(config)
