import csv

from fixed_noise_diffusion.summarize_sample_quality import (
    condition_kind,
    condition_pool_size,
    merge_gap_summary,
    read_gap_rows,
    read_quality_rows,
    summarize_quality,
)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_condition_parsing():
    assert condition_kind("gaussian") == "gaussian"
    assert condition_pool_size("gaussian") is None
    assert condition_kind("fixed_pool_10k") == "fixed_pool"
    assert condition_pool_size("fixed_pool_10k") == 10_000
    assert condition_kind("fixed_pool_whitened_100k") == "whitened"
    assert condition_pool_size("fixed_pool_whitened_100k") == 100_000


def test_summarize_quality_groups_by_condition_and_epoch(tmp_path):
    quality_path = tmp_path / "quality" / "sample_quality.csv"
    quality_path.parent.mkdir()
    _write_csv(
        quality_path,
        [
            {
                "run_name": "wp2_50ep_fixed_pool_1k_seed0",
                "condition": "fixed_pool_1k",
                "seed": "0",
                "epoch": "50",
                "fid": "10",
                "kid_mean": "0.1",
                "seconds": "3",
            },
            {
                "run_name": "wp2_50ep_fixed_pool_1k_seed1",
                "condition": "fixed_pool_1k",
                "seed": "1",
                "epoch": "50",
                "fid": "14",
                "kid_mean": "0.3",
                "seconds": "5",
            },
        ],
    )

    rows = read_quality_rows([quality_path.parent])
    summary = summarize_quality(rows)

    assert len(rows) == 2
    assert len(summary) == 1
    assert summary[0]["condition"] == "fixed_pool_1k"
    assert summary[0]["pool_size"] == "1000"
    assert summary[0]["n"] == "2"
    assert summary[0]["fid_mean"] == "12"
    assert summary[0]["kid_mean_mean"] == "0.2"


def test_gap_summary_column_variants_are_merged(tmp_path):
    gap_path = tmp_path / "gap.csv"
    _write_csv(
        gap_path,
        [
            {
                "condition": "fixed_pool_1k",
                "epoch": "1",
                "mean_denoising_gap": "0.01",
                "std_denoising_gap": "0.02",
            },
            {
                "condition": "fixed_pool_1k",
                "epoch": "50",
                "mean_denoising_gap": "0.12",
                "std_denoising_gap": "0.03",
            },
        ],
    )

    merged = merge_gap_summary(
        [
            {
                "kind": "fixed_pool",
                "condition": "fixed_pool_1k",
                "pool_size": "1000",
                "epoch": "50",
                "n": "2",
                "fid_mean": "12",
            }
        ],
        read_gap_rows([gap_path]),
    )

    assert merged[0]["denoising_gap_mean"] == "0.12"
    assert merged[0]["denoising_gap_std"] == "0.03"
