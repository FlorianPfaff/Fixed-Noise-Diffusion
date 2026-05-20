from fixed_noise_diffusion.metrics_utils import effective_kid_subset_size


def test_effective_kid_subset_size_respects_real_count() -> None:
    assert effective_kid_subset_size(kid_subset_size=100, sample_count=2048, real_count=32) == 32


def test_effective_kid_subset_size_respects_sample_count() -> None:
    assert effective_kid_subset_size(kid_subset_size=100, sample_count=64, real_count=2048) == 64


def test_effective_kid_subset_size_defaults_real_count_to_sample_count() -> None:
    assert effective_kid_subset_size(kid_subset_size=100, sample_count=64, real_count=None) == 64
