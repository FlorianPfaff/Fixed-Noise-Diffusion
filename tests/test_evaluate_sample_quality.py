from fixed_noise_diffusion.evaluate_sample_quality import (
    _evaluation_seed,
    _parse_run_metadata,
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
