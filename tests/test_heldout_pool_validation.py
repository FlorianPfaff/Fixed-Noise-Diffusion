import pytest
import torch

from fixed_noise_diffusion.config import load_config
from fixed_noise_diffusion.noise import FixedPoolNoiseSampler
from fixed_noise_diffusion.train import make_heldout_pool_sampler


def test_heldout_pool_rejects_fractional_pool_chunk_size():
    config = load_config("smoke.yaml")
    config["evaluation"]["enable_heldout_pool"] = True
    config["noise"]["pool_chunk_size"] = 1.5
    train_noise_sampler = FixedPoolNoiseSampler(
        image_shape=(1, 2, 2),
        device=torch.device("cpu"),
        pool_size=4,
        pool_seed=1,
        index_seed=2,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )

    with pytest.raises(ValueError, match="pool_chunk_size"):
        make_heldout_pool_sampler(config, train_noise_sampler, torch.device("cpu"))
