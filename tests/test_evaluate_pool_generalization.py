import torch

from fixed_noise_diffusion.evaluate_pool_generalization import _train_pool_seed_or_blank
from fixed_noise_diffusion.noise import FixedPoolNoiseSampler, GaussianNoiseSampler


def test_train_pool_seed_is_blank_for_gaussian_sampler():
    sampler = GaussianNoiseSampler((3, 4, 4), torch.device("cpu"), seed=123)

    assert _train_pool_seed_or_blank(sampler) == ""


def test_train_pool_seed_uses_actual_fixed_pool_sampler_seed():
    sampler = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=8,
        pool_seed=321,
        index_seed=654,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )

    assert _train_pool_seed_or_blank(sampler) == 321
