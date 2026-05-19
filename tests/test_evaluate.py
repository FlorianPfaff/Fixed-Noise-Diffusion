import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fixed_noise_diffusion.evaluate import first_real_batch


def test_first_real_batch_collects_requested_count_across_batches():
    images = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    labels = torch.arange(5)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    real = first_real_batch(loader, torch.device("cpu"), count=5)

    assert torch.equal(real, images)
    assert real.shape[0] == 5


def test_first_real_batch_truncates_final_batch_to_requested_count():
    images = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    labels = torch.arange(5)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    real = first_real_batch(loader, torch.device("cpu"), count=3)

    assert torch.equal(real, images[:3])
    assert real.shape[0] == 3


def test_first_real_batch_raises_when_count_exceeds_loader_size():
    images = torch.zeros(3, 3, 2, 2)
    labels = torch.arange(3)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="Requested 4 real images"):
        first_real_batch(loader, torch.device("cpu"), count=4)
