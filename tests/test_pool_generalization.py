from fixed_noise_diffusion.evaluate_pool_generalization import (
    heldout_pool_config,
    prepare_eval_config,
    summarize_rows,
)


def test_heldout_pool_config_changes_pool_seed_without_mutating_original():
    config = {
        "noise": {
            "mode": "fixed_pool",
            "pool_size": 1000,
            "pool_seed": 4242,
        }
    }

    heldout = heldout_pool_config(config, pool_seed_offset=17)

    assert config["noise"]["pool_seed"] == 4242
    assert heldout["noise"]["pool_seed"] == 4259


def test_prepare_eval_config_expands_eval_subset():
    config = {
        "data": {
            "download": False,
            "eval_batch_size": 8,
            "num_workers": 4,
            "eval_subset_size": 16,
        }
    }

    prepared = prepare_eval_config(
        config,
        batch_size=32,
        batches=3,
        data_dir="alt-data",
        num_workers=0,
    )

    assert config["data"]["eval_subset_size"] == 16
    assert prepared["data"]["download"] is True
    assert prepared["data"]["data_dir"] == "alt-data"
    assert prepared["data"]["eval_subset_size"] == 96


def test_summarize_rows_aggregates_heldout_gaps():
    rows = [
        {
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "epoch": "100",
            "train_noise_loss": "0.1",
            "heldout_pool_loss": "0.3",
            "fresh_gaussian_loss": "0.5",
            "heldout_pool_gap": "0.2",
            "fresh_gaussian_gap": "0.4",
            "gaussian_minus_heldout_gap": "0.2",
        },
        {
            "kind": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "epoch": "100",
            "train_noise_loss": "0.2",
            "heldout_pool_loss": "0.6",
            "fresh_gaussian_loss": "0.8",
            "heldout_pool_gap": "0.4",
            "fresh_gaussian_gap": "0.6",
            "gaussian_minus_heldout_gap": "0.2",
        },
    ]

    summary = summarize_rows(rows)

    assert len(summary) == 1
    assert summary[0]["train_noise_loss_mean"] == "0.15"
    assert summary[0]["heldout_pool_gap_mean"] == "0.3"
    assert summary[0]["fresh_gaussian_gap_mean"] == "0.5"
