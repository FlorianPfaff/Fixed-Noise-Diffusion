import pytest
import torch

from fixed_noise_diffusion.evaluate_pool_generalization import (
    heldout_pool_config,
    prepare_eval_config,
    summarize_rows,
    verify_train_pool_fingerprint,
)
from fixed_noise_diffusion.noise import FixedPoolNoiseSampler


def test_heldout_pool_config_changes_pool_seed_without_mutating_original():
    config = {
        "noise": {
            "mode": "fixed_pool",
            "pool_size": 1000,
            "pool_seed": 4242,
        }
    }

    heldout = heldout_pool_config(config, pool_seed_offset=17)

    assert config["noise"]["pool_seed"] == 4242
    assert heldout["noise"]["pool_seed"] == 4259


def _write_checkpoint_fingerprint(run_dir, fingerprint):
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(
        {"train_noise_pool_fingerprint": fingerprint},
        checkpoint_dir / "epoch_0001.pt",
    )


def test_verify_train_pool_fingerprint_accepts_matching_checkpoint(tmp_path):
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
    fingerprint = sampler.pool_fingerprint()
    _write_checkpoint_fingerprint(tmp_path, fingerprint)

    assert (
        verify_train_pool_fingerprint(tmp_path, 1, sampler, torch.device("cpu"))
        == fingerprint["sha256"]
    )


def test_verify_train_pool_fingerprint_rejects_mismatch(tmp_path):
    checkpoint_sampler = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=8,
        pool_seed=1,
        index_seed=2,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )
    reconstructed_sampler = FixedPoolNoiseSampler(
        image_shape=(3, 4, 4),
        device=torch.device("cpu"),
        pool_size=8,
        pool_seed=9,
        index_seed=2,
        dtype="float32",
        chunk_size=4,
        whiten=False,
    )
    _write_checkpoint_fingerprint(tmp_path, checkpoint_sampler.pool_fingerprint())

    with pytest.raises(ValueError, match="does not match"):
        verify_train_pool_fingerprint(
            tmp_path,
            1,
            reconstructed_sampler,
            torch.device("cpu"),
        )


def test_verify_train_pool_fingerprint_allows_legacy_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({}, checkpoint_dir / "epoch_0001.pt")
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

    assert verify_train_pool_fingerprint(tmp_path, 1, sampler, torch.device("cpu")) == ""


def test_prepare_eval_config_expands_eval_subset():
    config = {
        "data": {
            "download": False,
            "eval_batch_size": 8,
            "num_workers": 4,
            "eval_subset_size": 16,
        }
    }

    prepared = prepare_eval_config(
        config,
        batch_size=32,
        batches=3,
        data_dir="alt-data",
        num_workers=0,
    )

    assert config["data"]["eval_subset_size"] == 16
    assert prepared["data"]["download"] is True
    assert prepared["data"]["data_dir"] == "alt-data"
    assert prepared["data"]["eval_subset_size"] == 96


def test_summarize_rows_aggregates_heldout_gaps():
    rows = [
        {
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "epoch": "100",
            "train_noise_loss": "0.1",
            "heldout_pool_loss": "0.3",
            "fresh_gaussian_loss": "0.5",
            "heldout_pool_gap": "0.2",
            "fresh_gaussian_gap": "0.4",
            "gaussian_minus_heldout_gap": "0.2",
        },
        {
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "epoch": "100",
            "train_noise_loss": "0.2",
            "heldout_pool_loss": "0.6",
            "fresh_gaussian_loss": "0.8",
            "heldout_pool_gap": "0.4",
            "fresh_gaussian_gap": "0.6",
            "gaussian_minus_heldout_gap": "0.2",
        },
    ]

    summary = summarize_rows(rows)

    assert len(summary) == 1
    assert summary[0]["train_noise_loss_mean"] == "0.15"
    assert summary[0]["heldout_pool_gap_mean"] == "0.3"
    assert summary[0]["fresh_gaussian_gap_mean"] == "0.5"
