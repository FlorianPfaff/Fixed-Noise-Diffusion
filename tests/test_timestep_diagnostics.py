from argparse import Namespace
from pathlib import Path

import fixed_noise_diffusion.evaluate_timestep_diagnostics as diagnostics

from fixed_noise_diffusion.evaluate_timestep_diagnostics import (
    parse_int_list,
    summarize_timestep_rows,
)


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("0, 25,100") == [0, 25, 100]


def test_summarize_timestep_rows_groups_by_condition_epoch_and_timestep():
    rows = [
        {
            "dataset": "cifar10",
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": 1000,
            "epoch": 100,
            "timestep": 50,
            "train_noise_loss": 0.1,
            "gaussian_noise_loss": 0.4,
            "timestep_gap": 0.3,
        },
        {
            "dataset": "stl10",
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": 1000,
            "epoch": 100,
            "timestep": 50,
            "train_noise_loss": 0.2,
            "gaussian_noise_loss": 0.6,
            "timestep_gap": 0.4,
        },
    ]

    summary = summarize_timestep_rows(rows)

    assert len(summary) == 2
    assert [row["dataset"] for row in summary] == ["cifar10", "stl10"]
    assert all(row["condition"] == "fixed_pool_1k" for row in summary)
    assert all(row["epoch"] == "100" for row in summary)
    assert all(row["timestep"] == "50" for row in summary)
    assert [row["n"] for row in summary] == ["1", "1"]
    assert [row["train_noise_loss_mean"] for row in summary] == ["0.1", "0.2"]
    assert [row["gaussian_noise_loss_mean"] for row in summary] == ["0.4", "0.6"]
    assert [row["timestep_gap_mean"] for row in summary] == ["0.3", "0.4"]


def test_evaluate_run_verifies_checkpoint_pool_fingerprint(monkeypatch):
    config = {
        "seed": 0,
        "data": {"dataset": "cifar10", "eval_batch_size": 2, "num_workers": 0},
        "noise": {"mode": "fixed_pool", "pool_size": 1000},
    }

    class DummyDiffusion:
        num_timesteps = 4

    class DummyInfo:
        mode = "fixed_pool"
        pool_size = 1000
        pool_memory_mb = 1.0

    class DummySampler:
        image_shape = (3, 32, 32)
        info = DummyInfo()

        def fork(self, seed):
            return self

    class DummyGaussianSampler(DummySampler):
        def __init__(self, image_shape, device, seed):
            self.image_shape = image_shape
            self.device = device
            self.seed = seed

    def fake_load_checkpoint_model(run_dir, epoch, device):
        return object(), DummyDiffusion(), config, 7

    monkeypatch.setattr(diagnostics, "load_checkpoint_model", fake_load_checkpoint_model)
    monkeypatch.setattr(
        diagnostics,
        "make_dataloaders",
        lambda config: type("Loaders", (), {"val": object()})(),
    )
    train_sampler = DummySampler()
    monkeypatch.setattr(
        diagnostics,
        "make_noise_sampler",
        lambda config, device, purpose_seed_offset=0: train_sampler,
    )
    monkeypatch.setattr(diagnostics, "GaussianNoiseSampler", DummyGaussianSampler)
    monkeypatch.setattr(diagnostics, "run_identity_from_config", lambda run_dir, config: ("fixed_pool_1k", 0))

    calls = []

    def fake_verify_train_pool_fingerprint(run_dir, epoch, sampler, device):
        calls.append((run_dir, epoch, sampler, device))
        return "abc123"

    monkeypatch.setattr(
        diagnostics,
        "verify_train_pool_fingerprint",
        fake_verify_train_pool_fingerprint,
    )

    losses = iter([(0.1, 2), (0.2, 2)])
    monkeypatch.setattr(
        diagnostics,
        "fixed_timestep_denoising_loss",
        lambda **kwargs: next(losses),
    )
    args = Namespace(
        device="cpu",
        batch_size=2,
        batches=1,
        data_dir=None,
        num_workers=0,
        download_data=False,
        seed=0,
    )

    rows = diagnostics.evaluate_run(Path("run"), [1], [0], args)

    assert len(calls) == 1
    assert calls[0][0] == Path("run")
    assert calls[0][1] == 1
    assert calls[0][2] is train_sampler
    assert str(calls[0][3]) == "cpu"
    assert rows[0]["train_pool_fingerprint_sha256"] == "abc123"
