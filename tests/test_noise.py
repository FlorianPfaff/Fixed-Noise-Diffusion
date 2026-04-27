import torch

from fixed_noise_diffusion.noise import FixedPoolNoiseSampler, GaussianNoiseSampler


def test_fixed_pool_reuses_existing_pool_on_fork():
    sampler = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=8,
        pool_seed=1,
        index_seed=2,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )
    fork = sampler.fork(3)
    assert fork.pool.data_ptr() == sampler.pool.data_ptr()
    assert sampler.sample(2).shape == (2, 3, 4, 4)
    assert fork.sample(2).shape == (2, 3, 4, 4)


def test_gaussian_sampler_shape():
    sampler = GaussianNoiseSampler((3, 8, 8), torch.device("cpu"), seed=1)
    noise = sampler.sample(5)
    assert noise.shape == (5, 3, 8, 8)
    assert noise.dtype == torch.float32

