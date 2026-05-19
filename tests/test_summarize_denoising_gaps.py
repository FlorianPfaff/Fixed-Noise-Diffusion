from fixed_noise_diffusion.summarize_denoising_gaps import summarize_rows


def test_summarize_rows_does_not_require_seed_column_in_summary_rows():
    rows = [
        {
            "run_name": "wp2_100ep_fixed_pool_10k_seed1",
            "dataset": "cifar10",
            "experiment": "standard",
            "family": "fixed pool",
            "noise_mode": "fixed_pool",
            "condition": "fixed_pool_10k",
            "pool_size": "10000",
            "seed": "1",
            "epoch": "100",
            "step": "1000",
            "train_den_loss": "0.8",
            "gaussian_den_loss": "1.0",
            "denoising_gap": "0.2",
            "source_run_dir": "runs/wp2_100ep_fixed_pool_10k_seed1",
        },
        {
            "run_name": "wp2_100ep_fixed_pool_10k_seed2",
            "dataset": "cifar10",
            "experiment": "standard",
            "family": "fixed pool",
            "noise_mode": "fixed_pool",
            "condition": "fixed_pool_10k",
            "pool_size": "10000",
            "seed": "2",
            "epoch": "100",
            "step": "1000",
            "train_den_loss": "0.7",
            "gaussian_den_loss": "1.1",
            "denoising_gap": "0.4",
            "source_run_dir": "runs/wp2_100ep_fixed_pool_10k_seed2",
        },
    ]

    summary = summarize_rows(rows)

    assert summary == [
        {
            "dataset": "cifar10",
            "experiment": "standard",
            "family": "fixed pool",
            "noise_mode": "fixed_pool",
            "condition": "fixed_pool_10k",
            "pool_size": "10000",
            "epoch": "100",
            "n": "2",
            "denoising_gap_mean": "0.3",
            "denoising_gap_std": "0.1414213562373095",
            "denoising_gap_sem": "0.09999999999999999",
        }
    ]
