from types import SimpleNamespace

import pytest
import torch

import fixed_noise_diffusion.train as train_module
from fixed_noise_diffusion.noise import FixedPoolNoiseSampler


def test_evaluate_checkpoint_does_not_precast_denoising_batches(monkeypatch, tmp_path):
    def fake_denoising_loss(
        model,
        diffusion,
        loader,
        sampler,
        device,
        batches,
        seed,
    ):
        assert batches is True
        raise ValueError("evaluation.denoising_batches must be a positive integer")

    monkeypatch.setattr(train_module, "denoising_loss", fake_denoising_loss)
    monkeypatch.setattr(
        train_module,
        "sample_grid",
        lambda *args, **kwargs: torch.empty(0),
    )

    train_sampler = FixedPoolNoiseSampler(
        image_shape=(1, 1, 1),
        device=torch.device("cpu"),
        pool_size=8,
        pool_seed=30,
        index_seed=40,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )

    with pytest.raises(ValueError, match="evaluation.denoising_batches"):
        train_module.evaluate_checkpoint(
            model=torch.nn.Identity(),
            diffusion=object(),
            loaders=SimpleNamespace(val=object()),
            train_noise_sampler=train_sampler,
            heldout_noise_sampler=None,
            config={
                "seed": 3,
                "evaluation": {"denoising_batches": True, "enable_metrics": False},
            },
            device=torch.device("cpu"),
            run_dir=tmp_path,
            logger=SimpleNamespace(log=lambda record: None),
            epoch=7,
            step=123,
            timer=SimpleNamespace(elapsed=lambda: 0.0),
        )
