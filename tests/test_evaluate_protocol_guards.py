from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fixed_noise_diffusion.evaluate import optional_fid_kid, sample_grid


def test_sample_grid_rejects_sample_steps_above_num_timesteps(tmp_path):
    config = {
        "evaluation": {"sample_count": 1, "sample_steps": 11, "sampler": "ddim"},
        "data": {"channels": 1, "image_size": 4},
    }
    diffusion = SimpleNamespace(num_timesteps=10)

    with pytest.raises(ValueError, match="sample_steps must not exceed"):
        sample_grid(
            model=torch.nn.Identity(),
            diffusion=diffusion,
            config=config,
            device=torch.device("cpu"),
            output_path=Path(tmp_path) / "samples.png",
            seed=0,
        )


def test_optional_fid_kid_reports_configured_feature_on_empty_inputs():
    metrics = optional_fid_kid(
        torch.empty(0),
        torch.empty(0),
        device=torch.device("cpu"),
        feature=128,
    )

    assert metrics["fid_feature"] == 128
    assert metrics["fid"] is None
    assert "non-empty" in str(metrics["metrics_error"])


def test_optional_fid_kid_rejects_invalid_feature():
    with pytest.raises(ValueError, match="fid_feature"):
        optional_fid_kid(
            torch.empty(0),
            torch.empty(0),
            device=torch.device("cpu"),
            feature=0,
        )
