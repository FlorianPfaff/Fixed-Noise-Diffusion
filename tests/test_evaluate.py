import sys
from types import ModuleType

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fixed_noise_diffusion.evaluate import first_real_batch, optional_fid_kid


def _install_fake_torchmetrics(monkeypatch, fid_cls, kid_cls):
    torchmetrics_module = ModuleType("torchmetrics")
    torchmetrics_module.__path__ = []
    image_module = ModuleType("torchmetrics.image")
    image_module.__path__ = []
    fid_module = ModuleType("torchmetrics.image.fid")
    kid_module = ModuleType("torchmetrics.image.kid")
    fid_module.FrechetInceptionDistance = fid_cls
    kid_module.KernelInceptionDistance = kid_cls
    torchmetrics_module.image = image_module
    image_module.fid = fid_module
    image_module.kid = kid_module
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", image_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", fid_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", kid_module)


class _UnexpectedMetric:
    def __init__(self, *args, **kwargs):
        raise AssertionError("image metrics should be skipped for fewer than two images")


def test_first_real_batch_collects_requested_images_across_batches():
    images = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    labels = torch.zeros(images.shape[0], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    real = first_real_batch(loader, torch.device("cpu"), count=4)

    assert torch.equal(real, images[:4])


@pytest.mark.parametrize("real_count, fake_count", [(1, 1), (1, 4), (4, 1)])
def test_optional_fid_kid_skips_too_small_batches(monkeypatch, real_count, fake_count):
    _install_fake_torchmetrics(monkeypatch, _UnexpectedMetric, _UnexpectedMetric)

    metrics = optional_fid_kid(
        torch.zeros(real_count, 3, 4, 4),
        torch.zeros(fake_count, 3, 4, 4),
        torch.device("cpu"),
    )

    assert metrics == {"fid": None, "kid_mean": None, "kid_std": None}


def test_optional_fid_kid_caps_kid_subset_size_by_smaller_batch(monkeypatch):
    subset_sizes = []

    class FakeFid:
        def __init__(self, feature, normalize):
            assert feature == 64
            assert normalize is False

        def to(self, device):
            return self

        def update(self, images, real):
            assert images.dtype == torch.uint8
            assert isinstance(real, bool)

        def compute(self):
            return torch.tensor(12.0)

    class FakeKid:
        def __init__(self, subset_size, normalize):
            subset_sizes.append(subset_size)
            assert normalize is False

        def to(self, device):
            return self

        def update(self, images, real):
            assert images.dtype == torch.uint8
            assert isinstance(real, bool)

        def compute(self):
            return torch.tensor(0.5), torch.tensor(0.25)

    _install_fake_torchmetrics(monkeypatch, FakeFid, FakeKid)

    metrics = optional_fid_kid(
        torch.zeros(3, 3, 4, 4),
        torch.zeros(7, 3, 4, 4),
        torch.device("cpu"),
    )

    assert subset_sizes == [3]
    assert metrics == {"fid": 12.0, "kid_mean": 0.5, "kid_std": 0.25}
