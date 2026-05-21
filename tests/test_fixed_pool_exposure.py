import pytest
import torch

from fixed_noise_diffusion.noise import FixedPoolNoiseSampler, make_noise_sampler


def _tracked_sampler(track_exposure: bool = True) -> FixedPoolNoiseSampler:
    return FixedPoolNoiseSampler(
        image_shape=(1, 1, 1),
        device=torch.device("cpu"),
        pool_size=4,
        pool_seed=1,
        index_seed=2,
        dtype="float32",
        chunk_size=4,
        whiten=False,
        track_exposure=track_exposure,
    )


def test_fixed_pool_exposure_summary_tracks_actual_draws_and_resets():
    sampler = _tracked_sampler(track_exposure=True)

    initial = sampler.exposure_summary()
    assert initial["pool_exposure_tracked"] is True
    assert initial["pool_draws"] == 0
    assert initial["pool_unique_entries"] == 0
    assert initial["pool_unique_fraction"] == 0.0

    sampler.sample(3)
    summary = sampler.exposure_summary()
    unique_entries = summary["pool_unique_entries"]

    assert summary["pool_draws"] == 3
    assert 1 <= unique_entries <= 3
    assert summary["pool_unseen_entries"] == 4 - unique_entries
    assert summary["pool_unique_fraction"] == pytest.approx(unique_entries / 4)
    assert summary["pool_duplicate_draw_fraction"] == pytest.approx(1 - unique_entries / 3)
    assert summary["pool_expected_unique_entries"] == pytest.approx(4 * (1 - (3 / 4) ** 3))
    assert summary["pool_expected_unique_fraction"] == pytest.approx(1 - (3 / 4) ** 3)
    assert summary["pool_expected_duplicate_draw_fraction"] == pytest.approx(1 - (4 * (1 - (3 / 4) ** 3)) / 3)
    assert summary["pool_max_entry_draws"] >= 1

    sampler.reset_exposure()
    reset = sampler.exposure_summary()
    assert reset["pool_draws"] == 0
    assert reset["pool_unique_entries"] == 0


def test_untracked_fixed_pool_reports_tracking_disabled():
    sampler = _tracked_sampler(track_exposure=False)
    sampler.sample(3)

    assert sampler.exposure_summary() == {"pool_exposure_tracked": False}


def test_fixed_pool_forks_do_not_share_exposure_counters_by_default():
    sampler = _tracked_sampler(track_exposure=True)
    sampler.sample(2)

    untracked_fork = sampler.fork(100)
    untracked_fork.sample(3)
    assert untracked_fork.exposure_summary() == {"pool_exposure_tracked": False}
    assert sampler.exposure_summary()["pool_draws"] == 2

    tracked_fork = sampler.fork(101, track_exposure=True)
    tracked_fork.sample(3)
    assert tracked_fork.exposure_summary()["pool_draws"] == 3
    assert sampler.exposure_summary()["pool_draws"] == 2


def test_make_noise_sampler_can_enable_fixed_pool_exposure_tracking():
    config = {
        "seed": 0,
        "data": {"channels": 1, "image_size": 1},
        "noise": {
            "mode": "fixed_pool",
            "pool_size": 4,
            "pool_seed": 1,
            "pool_dtype": "float32",
            "pool_chunk_size": 4,
            "whiten": False,
        },
    }

    sampler = make_noise_sampler(config, torch.device("cpu"), track_exposure=True)
    assert isinstance(sampler, FixedPoolNoiseSampler)

    sampler.sample(4)
    assert sampler.exposure_summary()["pool_draws"] == 4
