from fixed_noise_diffusion.evaluate_timestep_diagnostics import (
    parse_int_list,
    summarize_timestep_rows,
)


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("0, 25,100") == [0, 25, 100]


def test_summarize_timestep_rows_groups_by_condition_epoch_and_timestep():
    rows = [
        {
            "dataset": "cifar10",
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": 1000,
            "epoch": 100,
            "timestep": 50,
            "train_noise_loss": 0.1,
            "gaussian_noise_loss": 0.4,
            "timestep_gap": 0.3,
        },
        {
            "dataset": "stl10",
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": 1000,
            "epoch": 100,
            "timestep": 50,
            "train_noise_loss": 0.2,
            "gaussian_noise_loss": 0.6,
            "timestep_gap": 0.4,
        },
    ]

    summary = summarize_timestep_rows(rows)

    assert len(summary) == 2
    assert [row["dataset"] for row in summary] == ["cifar10", "stl10"]
    assert all(row["condition"] == "fixed_pool_1k" for row in summary)
    assert all(row["epoch"] == "100" for row in summary)
    assert all(row["timestep"] == "50" for row in summary)
    assert [row["n"] for row in summary] == ["1", "1"]
    assert [row["train_noise_loss_mean"] for row in summary] == ["0.1", "0.2"]
    assert [row["gaussian_noise_loss_mean"] for row in summary] == ["0.4", "0.6"]
    assert [row["timestep_gap_mean"] for row in summary] == ["0.3", "0.4"]
