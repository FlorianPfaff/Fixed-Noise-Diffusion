import sys
import types

import torch
from torch.utils.data import DataLoader, TensorDataset

from fixed_noise_diffusion.evaluate import first_real_batch, optional_fid_kid


def test_first_real_batch_collects_requested_images_across_batches():
    images = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    labels = torch.zeros(images.shape[0], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    real = first_real_batch(loader, torch.device("cpu"), count=4)

    assert torch.equal(real, images[:4])


def test_optional_fid_kid_clamps_kid_subset_to_real_and_fake_counts(monkeypatch):
    class FakeFrechetInceptionDistance:
        def __init__(self, *, feature, normalize):
            self.feature = feature
            self.normalize = normalize

        def to(self, device):
            return self

        def update(self, images, *, real):
            pass

        def compute(self):
            return torch.tensor(0.0)

    class FakeKernelInceptionDistance:
        subset_size = None

        def __init__(self, *, subset_size, normalize):
            type(self).subset_size = subset_size
            self.normalize = normalize

        def to(self, device):
            return self

        def update(self, images, *, real):
            pass

        def compute(self):
            return torch.tensor(0.0), torch.tensor(0.0)

    torchmetrics_module = types.ModuleType("torchmetrics")
    image_module = types.ModuleType("torchmetrics.image")
    fid_module = types.ModuleType("torchmetrics.image.fid")
    kid_module = types.ModuleType("torchmetrics.image.kid")
    fid_module.FrechetInceptionDistance = FakeFrechetInceptionDistance
    kid_module.KernelInceptionDistance = FakeKernelInceptionDistance
    image_module.fid = fid_module
    image_module.kid = kid_module
    torchmetrics_module.image = image_module
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", image_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", fid_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", kid_module)

    metrics = optional_fid_kid(
        torch.zeros(3, 3, 4, 4),
        torch.zeros(5, 3, 4, 4),
        torch.device("cpu"),
    )

    assert FakeKernelInceptionDistance.subset_size == 3
    assert metrics == {"fid": 0.0, "kid_mean": 0.0, "kid_std": 0.0}
