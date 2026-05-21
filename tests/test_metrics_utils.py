import pytest

from fixed_noise_diffusion.metrics_utils import effective_kid_subset_size


def test_effective_kid_subset_size_respects_real_count() -> None:
    assert effective_kid_subset_size(kid_subset_size=100, sample_count=2048, real_count=32) == 32


def test_effective_kid_subset_size_respects_sample_count() -> None:
    assert effective_kid_subset_size(kid_subset_size=100, sample_count=64, real_count=2048) == 64


def test_effective_kid_subset_size_defaults_real_count_to_sample_count() -> None:
    assert effective_kid_subset_size(kid_subset_size=100, sample_count=64, real_count=None) == 64


def test_effective_kid_subset_size_rejects_zero_real_count() -> None:
    with pytest.raises(ValueError, match="real_count"):
        effective_kid_subset_size(kid_subset_size=100, sample_count=64, real_count=0)


def test_effective_kid_subset_size_rejects_zero_sample_count() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        effective_kid_subset_size(kid_subset_size=100, sample_count=0, real_count=None)


def test_effective_kid_subset_size_rejects_too_small_kid_subset_size() -> None:
    with pytest.raises(ValueError, match="kid_subset_size"):
        effective_kid_subset_size(kid_subset_size=1, sample_count=64, real_count=64)
