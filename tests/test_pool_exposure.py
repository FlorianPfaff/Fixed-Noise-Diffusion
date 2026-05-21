import pytest

from fixed_noise_diffusion.pool_exposure import expected_fixed_pool_exposure


def test_expected_fixed_pool_exposure_zero_draws():
    stats = expected_fixed_pool_exposure(pool_size=10, draws=0)

    assert stats["pool_size"] == 10
    assert stats["draws"] == 0
    assert stats["expected_unique_pool_entries"] == 0.0
    assert stats["expected_unique_pool_fraction"] == 0.0
    assert stats["expected_duplicate_draw_fraction"] == 0.0


def test_expected_fixed_pool_exposure_one_draw():
    stats = expected_fixed_pool_exposure(pool_size=10, draws=1)

    assert stats["expected_unique_pool_entries"] == pytest.approx(1.0)
    assert stats["expected_unique_pool_fraction"] == pytest.approx(0.1)
    assert stats["expected_duplicate_draw_fraction"] == pytest.approx(0.0)


def test_expected_fixed_pool_exposure_many_draws():
    stats = expected_fixed_pool_exposure(pool_size=10, draws=10)

    assert stats["draws_per_pool_entry"] == pytest.approx(1.0)
    assert stats["expected_unique_pool_entries"] == pytest.approx(10 * (1 - 0.9**10))
    assert 0.0 < stats["expected_duplicate_draw_fraction"] < 1.0


@pytest.mark.parametrize(("pool_size", "draws"), [(0, 1), (10, -1)])
def test_expected_fixed_pool_exposure_rejects_invalid_inputs(pool_size, draws):
    with pytest.raises(ValueError):
        expected_fixed_pool_exposure(pool_size=pool_size, draws=draws)
