import pytest
import torch

from fixed_noise_diffusion.noise import FixedPoolNoiseSampler


def test_whitened_fixed_pool_requires_at_least_two_rows():
    with pytest.raises(ValueError, match="pool_size must be at least 2"):
        FixedPoolNoiseSampler(
            image_shape=(1, 1, 1),
            device=torch.device("cpu"),
            pool_size=1,
            pool_seed=1,
            index_seed=2,
            dtype="float32",
            chunk_size=1,
            whiten=True,
        )
