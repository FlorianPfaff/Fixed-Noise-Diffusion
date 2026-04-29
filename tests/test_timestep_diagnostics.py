from fixed_noise_diffusion.evaluate_timestep_diagnostics import (
    parse_int_list,
    summarize_timestep_rows,
)


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("0, 25,100") == [0, 25, 100]


def test_summarize_timestep_rows_groups_by_condition_epoch_and_timestep():
    rows = [
        {
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

    assert len(summary) == 1
    assert summary[0]["condition"] == "fixed_pool_1k"
    assert summary[0]["epoch"] == "100"
    assert summary[0]["timestep"] == "50"
    assert summary[0]["n"] == "2"
    assert summary[0]["train_noise_loss_mean"] == "0.15"
    assert summary[0]["gaussian_noise_loss_mean"] == "0.5"
    assert summary[0]["timestep_gap_mean"] == "0.35"
