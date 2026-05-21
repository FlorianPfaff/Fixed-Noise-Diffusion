import pytest

from fixed_noise_diffusion.evaluate_sample_quality import (
    _metric_population_count,
    _nonnegative_cli_int,
    _positive_cli_int,
    _evaluation_seed,
    _parse_run_metadata,
    _requested_real_count,
)
from fixed_noise_diffusion.utils import seed_everything


def test_non_wp2_run_name_uses_safe_seed_component():
    condition, run_seed = _parse_run_metadata("custom_experiment")

    assert condition == "custom_experiment"
    assert run_seed is None
    assert _evaluation_seed(0, run_seed, epoch=1) == 1
    assert _evaluation_seed(0, run_seed, epoch=1, offset=50_000) == 50_001
    seed_everything(_evaluation_seed(0, run_seed, epoch=1))


def test_wp2_run_name_preserves_seed_component():
    condition, run_seed = _parse_run_metadata("wp2_50ep_fixed_pool_seed3")

    assert condition == "fixed_pool"
    assert run_seed == 3
    assert _evaluation_seed(7, run_seed, epoch=5) == 3012


def test_requested_real_count_defaults_to_sample_count() -> None:
    assert _requested_real_count(real_count=None, sample_count=64) == 64


def test_requested_real_count_preserves_explicit_positive_count() -> None:
    assert _requested_real_count(real_count=32, sample_count=64) == 32


@pytest.mark.parametrize(
    ("real_count", "sample_count", "match"),
    [
        (0, 64, "--real-count"),
        (-1, 64, "--real-count"),
        (1, 64, "--real-count"),
        (None, 0, "--sample-count"),
        (None, -1, "--sample-count"),
        (None, 1, "--sample-count"),
    ],
)
def test_requested_real_count_rejects_nonpositive_counts(
    real_count,
    sample_count,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        _requested_real_count(real_count=real_count, sample_count=sample_count)


def test_metric_population_count_requires_two_images_for_metrics() -> None:
    assert _metric_population_count("--sample-count", 2) == 2
    with pytest.raises(ValueError, match="--sample-count"):
        _metric_population_count("--sample-count", 1)


def test_positive_cli_int_rejects_zero_sample_batch_size() -> None:
    assert _positive_cli_int("--sample-batch-size", 1) == 1
    with pytest.raises(ValueError, match="--sample-batch-size"):
        _positive_cli_int("--sample-batch-size", 0)


def test_nonnegative_cli_int_allows_zero_grid_count_only() -> None:
    assert _nonnegative_cli_int("--grid-count", 0) == 0
    assert _nonnegative_cli_int("--grid-count", 1) == 1
    with pytest.raises(ValueError, match="--grid-count"):
        _nonnegative_cli_int("--grid-count", -1)
