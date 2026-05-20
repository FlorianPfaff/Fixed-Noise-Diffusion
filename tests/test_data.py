import pytest
from PIL import Image

from fixed_noise_diffusion.data import RandomImageDataset, _image_transform, _subset


def test_center_crop_resize_transform_returns_square_normalized_tensor():
    transform = _image_transform(
        {"image_size": 64, "resize": True, "center_crop_size": 178},
        native_size=178,
    )

    image = Image.new("RGB", (178, 218), color=(128, 128, 128))
    tensor = transform(image)

    assert tensor.shape == (3, 64, 64)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0


def test_center_crop_resizes_to_configured_size_without_resize_flag():
    transform = _image_transform(
        {"image_size": 64, "resize": False, "center_crop_size": 178},
        native_size=218,
    )

    image = Image.new("RGB", (178, 218), color=(128, 128, 128))
    tensor = transform(image)

    assert tensor.shape == (3, 64, 64)


def test_image_transform_respects_grayscale_channel_config():
    transform = _image_transform(
        {"image_size": 32, "channels": 1},
        native_size=32,
    )

    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    tensor = transform(image)

    assert tensor.shape == (1, 32, 32)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0


def test_image_transform_rejects_unsupported_torchvision_channel_count():
    with pytest.raises(ValueError, match="data.channels=1 or 3"):
        _image_transform({"image_size": 32, "channels": 2}, native_size=32)


def test_subset_rejects_negative_size():
    dataset = RandomImageDataset(length=4, channels=1, image_size=2, seed=0)

    with pytest.raises(ValueError, match="subset_size"):
        _subset(dataset, -1, seed=0, name="subset_size")


def test_eval_subset_rejects_negative_size():
    dataset = RandomImageDataset(length=4, channels=1, image_size=2, seed=0)

    with pytest.raises(ValueError, match="eval_subset_size"):
        _subset(dataset, -2, seed=0, name="eval_subset_size")
