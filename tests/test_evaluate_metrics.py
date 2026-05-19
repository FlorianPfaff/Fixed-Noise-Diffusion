import pytest
import torch

from fixed_noise_diffusion.evaluate import empty_fid_kid_metrics, optional_fid_kid


def test_empty_fid_kid_metrics_uses_feature_specific_fid_key() -> None:
    metrics = empty_fid_kid_metrics(2048)

    assert "fid" not in metrics
    assert metrics["fid2048"] is None
    assert metrics["fid_feature"] == 2048
    assert metrics["fid_metric_name"] == "fid2048"
    assert metrics["kid_mean"] is None
    assert metrics["kid_std"] is None


def test_optional_fid_kid_rejects_unsupported_fid_feature() -> None:
    images = torch.empty(0, 3, 32, 32)

    with pytest.raises(ValueError, match="fid_feature"):
        optional_fid_kid(
            images, images, torch.device("cpu"), fid_feature=123
        )
