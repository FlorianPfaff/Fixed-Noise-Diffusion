import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fixed_noise_diffusion.evaluate import denoising_loss_from_timesteps, sample_grid


class _IdentityDiffusion:
    num_timesteps = 4

    def q_sample(self, images, timesteps, noise):
        return images + 0 * noise + 0 * timesteps.reshape(-1, 1, 1, 1)

    def sample(self, *args, **kwargs):
        raise AssertionError("sample should not be called for invalid sample configuration")


class _ZeroModel(torch.nn.Module):
    def forward(self, x, timesteps):
        return torch.zeros_like(x)


def _loader():
    images = torch.zeros(2, 1, 2, 2)
    labels = torch.zeros(2, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=2)


@pytest.mark.parametrize("batches", [0, -1, 1.5, True])
def test_denoising_loss_rejects_invalid_batch_budget(batches):
    sampler = type(
        "Sampler",
        (),
        {"sample": lambda self, n: torch.zeros(n, 1, 2, 2)},
    )()

    with pytest.raises(ValueError, match="evaluation.denoising_batches"):
        denoising_loss_from_timesteps(
            model=_ZeroModel(),
            diffusion=_IdentityDiffusion(),
            loader=_loader(),
            sampler=sampler,
            device=torch.device("cpu"),
            batches=batches,
            make_timesteps=lambda batch_size: torch.zeros(batch_size, dtype=torch.long),
        )


@pytest.mark.parametrize("sample_count", [-1, 1.5, True])
def test_sample_grid_rejects_invalid_sample_count(tmp_path, sample_count):
    config = {
        "evaluation": {"sample_count": sample_count, "sample_steps": 1, "sampler": "ddim"},
        "data": {"channels": 1, "image_size": 2},
    }

    with pytest.raises(ValueError, match="evaluation.sample_count"):
        sample_grid(
            _ZeroModel(),
            _IdentityDiffusion(),
            config,
            torch.device("cpu"),
            tmp_path / "samples.png",
            seed=1,
        )


def test_sample_grid_allows_zero_sample_count(tmp_path):
    config = {
        "evaluation": {"sample_count": 0, "sample_steps": 1, "sampler": "ddim"},
        "data": {"channels": 1, "image_size": 2},
    }

    samples = sample_grid(
        _ZeroModel(),
        _IdentityDiffusion(),
        config,
        torch.device("cpu"),
        tmp_path / "samples.png",
        seed=1,
    )

    assert samples.numel() == 0
