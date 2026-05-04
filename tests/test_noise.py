import torch

from fixed_noise_diffusion.noise import FixedPoolNoiseSampler, GaussianNoiseSampler
from fixed_noise_diffusion.train import (
    make_evaluation_samplers,
    make_heldout_pool_sampler,
)


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


def test_fixed_pool_index_stream_is_seeded():
    sampler_a = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=16,
        pool_seed=10,
        index_seed=20,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )
    sampler_b = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=16,
        pool_seed=10,
        index_seed=20,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )
    sampler_c = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=16,
        pool_seed=10,
        index_seed=21,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )

    sample_a = sampler_a.sample(8)
    assert torch.equal(sample_a, sampler_b.sample(8))
    assert not torch.equal(sample_a, sampler_c.sample(8))


def test_heldout_gaussian_eval_sampler_does_not_reuse_training_pool():
    train_sampler = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=16,
        pool_seed=30,
        index_seed=40,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )

    train_eval, gaussian_eval = make_evaluation_samplers(
        train_sampler,
        seed=50,
        epoch=1,
        device=torch.device("cpu"),
    )

    assert isinstance(train_eval, FixedPoolNoiseSampler)
    assert train_eval.pool.data_ptr() == train_sampler.pool.data_ptr()
    assert isinstance(gaussian_eval, GaussianNoiseSampler)
    assert gaussian_eval.info.mode == "gaussian"
    assert gaussian_eval.info.pool_size is None
    assert not torch.equal(train_eval.sample(4), gaussian_eval.sample(4))


def test_heldout_pool_eval_sampler_uses_distinct_pool_seed():
    config = {
        "seed": 0,
        "data": {"channels": 3, "image_size": 4},
        "evaluation": {
            "enable_heldout_pool": True,
            "heldout_pool_seed": None,
            "heldout_pool_seed_offset": 17,
        },
        "noise": {
            "mode": "fixed_pool",
            "pool_size": 16,
            "pool_seed": 30,
            "pool_dtype": "float32",
            "pool_chunk_size": 4,
            "whiten": False,
        },
    }
    train_sampler = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=16,
        pool_seed=30,
        index_seed=40,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )

    heldout_sampler = make_heldout_pool_sampler(
        config, train_sampler, torch.device("cpu")
    )

    assert isinstance(heldout_sampler, FixedPoolNoiseSampler)
    assert heldout_sampler.pool_seed == 47
    assert heldout_sampler.pool.data_ptr() != train_sampler.pool.data_ptr()
    assert not torch.equal(heldout_sampler.pool, train_sampler.pool)


def test_heldout_pool_eval_sampler_skips_gaussian_noise():
    config = {
        "seed": 0,
        "evaluation": {"enable_heldout_pool": True},
        "noise": {"mode": "gaussian", "pool_seed": 30},
    }
    train_sampler = GaussianNoiseSampler((3, 4, 4), torch.device("cpu"), seed=1)

    assert make_heldout_pool_sampler(config, train_sampler, torch.device("cpu")) is None


def test_large_fixed_pool_stays_cpu_backed():
    sampler = FixedPoolNoiseSampler(
        image_shape=(1, 1, 1),
        device=torch.device("cpu"),
        pool_size=100_000,
        pool_seed=1,
        index_seed=2,
        dtype="float16",
        chunk_size=8192,
        whiten=False,
    )

    assert sampler.pool.device.type == "cpu"
    assert sampler.info.pool_size == 100_000
    assert sampler.info.pool_memory_mb < 1
    assert sampler.sample(4).device.type == "cpu"
