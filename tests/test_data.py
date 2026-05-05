from fixed_noise_diffusion.data import _image_transform
from PIL import Image


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
