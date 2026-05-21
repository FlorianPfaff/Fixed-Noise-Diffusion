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
            "beta_schedule": "",
            "num_timesteps": "",
            "image_size": "",
            "channels": "",
            "base_channels": "",
            "channel_mults": "",
            "time_emb_dim": "",
            "epoch": "100",
            "n": "2",
            "denoising_gap_mean": "0.3",
            "denoising_gap_std": "0.1414213562373095",
            "denoising_gap_sem": "0.09999999999999999",
        }
    ]


def test_summarize_rows_sorts_without_seed_key():
    rows = [
        {
            "run_name": "run0",
            "dataset": "cifar10",
            "experiment": "standard",
            "family": "fixed pool",
            "noise_mode": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "epoch": "100",
            "step": "10",
            "train_den_loss": "0.5",
            "gaussian_den_loss": "0.6",
            "denoising_gap": "0.1",
            "source_run_dir": "runs/run0",
        },
        {
            "run_name": "run1",
            "dataset": "cifar10",
            "experiment": "standard",
            "family": "fixed pool",
            "noise_mode": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "epoch": "100",
            "step": "12",
            "train_den_loss": "0.5",
            "gaussian_den_loss": "0.8",
            "denoising_gap": "0.3",
            "source_run_dir": "runs/run1",
        },
    ]

    summary = summarize_rows(rows)

    assert len(summary) == 1
    assert summary[0]["n"] == "2"
    assert summary[0]["denoising_gap_mean"] == "0.2"


def test_summarize_rows_keeps_protocols_separate():
    def row(beta_schedule: str, denoising_gap: str) -> dict[str, str]:
        return {
            "run_name": f"run_{beta_schedule}",
            "dataset": "cifar10",
            "experiment": "standard",
            "family": "fixed pool",
            "noise_mode": "fixed_pool",
            "condition": "fixed_pool_1k",
            "pool_size": "1000",
            "beta_schedule": beta_schedule,
            "num_timesteps": "1000",
            "image_size": "32",
            "channels": "3",
            "base_channels": "64",
            "channel_mults": "1,2,2,4",
            "time_emb_dim": "256",
            "seed": "0",
            "epoch": "100",
            "step": "1000",
            "train_den_loss": "0.8",
            "gaussian_den_loss": str(0.8 + float(denoising_gap)),
            "denoising_gap": denoising_gap,
            "source_run_dir": f"runs/run_{beta_schedule}",
        }

    summary = summarize_rows(
        [
            row("cosine", "0.2"),
            row("linear", "0.4"),
        ]
    )

    assert len(summary) == 2
    by_schedule = {item["beta_schedule"]: item for item in summary}
    assert by_schedule["cosine"]["n"] == "1"
    assert by_schedule["cosine"]["denoising_gap_mean"] == "0.2"
    assert by_schedule["linear"]["n"] == "1"
    assert by_schedule["linear"]["denoising_gap_mean"] == "0.4"
