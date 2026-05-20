import pytest

from fixed_noise_diffusion.config import load_config
from fixed_noise_diffusion.train import (
    _accumulation_group_size,
    _mean_accumulated_loss,
    _positive_training_int,
    _should_finish_accumulation,
    _require_train_batches,
    train,
)


def test_gradient_accumulation_finishes_incomplete_tail_group():
    total_batches = 5
    grad_accum_steps = 2

    assert [
        _accumulation_group_size(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [2, 2, 2, 2, 1]
    assert [
        _should_finish_accumulation(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [False, True, False, True, True]


def test_gradient_accumulation_uses_actual_tail_group_size():
    total_batches = 3
    grad_accum_steps = 8

    assert [
        _accumulation_group_size(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [3, 3, 3]
    assert [
        _should_finish_accumulation(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [False, False, True]


def test_gradient_accumulation_reports_mean_loss_over_optimizer_step():
    microbatch_losses = [0.25, 0.75, 1.25]

    assert _mean_accumulated_loss(
        sum(microbatch_losses), len(microbatch_losses)
    ) == pytest.approx(0.75)


def test_gradient_accumulation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="grad_accum_steps"):
        _accumulation_group_size(batch_index=1, total_batches=1, grad_accum_steps=0)
    with pytest.raises(ValueError, match="total_batches"):
        _accumulation_group_size(batch_index=1, total_batches=0, grad_accum_steps=1)
    with pytest.raises(ValueError, match="batch_index"):
        _accumulation_group_size(batch_index=2, total_batches=1, grad_accum_steps=1)
    with pytest.raises(ValueError, match="accumulation_steps"):
        _mean_accumulated_loss(
            loss_sum=1.0,
            accumulation_steps=0,
        )


def test_training_rejects_empty_train_loader():
    _require_train_batches(1)

    with pytest.raises(ValueError, match="Training loader produced no batches"):
        _require_train_batches(0)


def test_training_positive_integer_config_rejects_invalid_log_interval():
    config = {"training": {"log_interval_steps": 0}}

    with pytest.raises(ValueError, match="training.log_interval_steps"):
        _positive_training_int(config, "log_interval_steps", 100)


def test_training_positive_integer_config_accepts_defaults_and_strings():
    assert _positive_training_int({"training": {}}, "log_interval_steps", 100) == 100
    assert _positive_training_int({"training": {"log_interval_steps": "7"}}, "log_interval_steps", 100) == 7


def test_train_rejects_zero_training_batches(tmp_path):
    config = load_config("smoke.yaml")
    config["output_dir"] = str(tmp_path)
    config["run_name"] = "zero_batches"
    config["device"] = "cpu"
    config["data"]["fake_train_size"] = 2
    config["data"]["batch_size"] = 4
    config["training"]["max_train_steps"] = 1

    with pytest.raises(ValueError, match="Training loader produced no batches"):
        train(config)
