from pathlib import Path

import pytest

from fixed_noise_diffusion.evaluate_sample_quality import _resolve_run_metadata


def test_resolve_run_metadata_prefers_checkpoint_seed_over_run_directory():
    condition, seed = _resolve_run_metadata(
        Path("wp2_50ep_fixed_pool_1k_seed3"),
        {"seed": 17},
    )

    assert condition == "fixed_pool_1k"
    assert seed == 17


def test_resolve_run_metadata_falls_back_to_run_directory_seed():
    condition, seed = _resolve_run_metadata(
        Path("wp2_50ep_gaussian_seed4"),
        {},
    )

    assert condition == "gaussian"
    assert seed == 4


def test_resolve_run_metadata_uses_safe_default_for_unmatched_run_directory():
    condition, seed = _resolve_run_metadata(Path("renamed_manual_export"), {})

    assert condition == "renamed_manual_export"
    assert seed == 0


def test_resolve_run_metadata_rejects_invalid_checkpoint_seed():
    with pytest.raises(ValueError, match="seed"):
        _resolve_run_metadata(
            Path("wp2_50ep_fixed_pool_1k_seed3"),
            {"seed": "not-an-int"},
        )
