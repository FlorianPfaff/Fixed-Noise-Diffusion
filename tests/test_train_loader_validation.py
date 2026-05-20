import pytest

from fixed_noise_diffusion.train import _require_train_batches


def test_require_train_batches_rejects_zero_batches() -> None:
    with pytest.raises(ValueError, match="no batches"):
        _require_train_batches(0)


def test_require_train_batches_accepts_nonempty_loader() -> None:
    _require_train_batches(1)
