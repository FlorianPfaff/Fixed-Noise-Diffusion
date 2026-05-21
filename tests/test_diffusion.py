import pytest
import torch

from fixed_noise_diffusion.diffusion import GaussianDiffusion


def _make_diffusion(**overrides):
    kwargs = {
        "num_timesteps": 10,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "device": torch.device("cpu"),
    }
    kwargs.update(overrides)
    return GaussianDiffusion(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_timesteps": 0}, "num_timesteps"),
        ({"num_timesteps": -1}, "num_timesteps"),
        ({"num_timesteps": 10.5}, "num_timesteps"),
        ({"num_timesteps": True}, "num_timesteps"),
        ({"beta_start": 0.0}, "beta_start"),
        ({"beta_start": -0.1}, "beta_start"),
        ({"beta_end": 1.0}, "beta_end"),
        ({"beta_end": float("inf")}, "beta_end"),
        ({"beta_start": 0.02, "beta_end": 0.01}, "less than"),
    ],
)
def test_gaussian_diffusion_rejects_invalid_schedule_config(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_diffusion(**overrides)


def test_gaussian_diffusion_accepts_valid_linear_schedule():
    diffusion = _make_diffusion(num_timesteps="10", beta_start="0.0001", beta_end="0.02")

    assert diffusion.num_timesteps == 10
    assert diffusion.betas.shape == (10,)


def test_gaussian_diffusion_from_config_rejects_fractional_timesteps():
    config = {
        "diffusion": {
            "num_timesteps": 10.5,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }
    }

    with pytest.raises(ValueError, match="num_timesteps"):
        GaussianDiffusion.from_config(config, torch.device("cpu"))
