from fixed_noise_diffusion.summarize_phase_diagram import (
    normalize_summary_row,
    parse_input_spec,
    read_phase_rows,
)


def test_parse_input_spec_requires_label():
    label, path = parse_input_spec("linear=runs/summary.csv")

    assert label == "linear"
    assert str(path) == "runs\\summary.csv" or str(path) == "runs/summary.csv"


def test_normalize_summary_row_infers_schedule_model_and_pool_size(tmp_path):
    row = normalize_summary_row(
        {
            "condition": "strong96_cosine_fixed_pool_20k",
            "epoch": "50",
            "n": "4",
            "fid_mean": "2.2",
            "denoising_gap_mean": "0.01",
            "low_mid_mean_timestep_gap": "0.02",
        },
        "strong96_cosine",
        tmp_path / "summary.csv",
    )

    assert row["schedule"] == "cosine"
    assert row["model"] == "strong96"
    assert row["kind"] == "fixed_pool"
    assert row["pool_size"] == "20000"
    assert row["fid_mean"] == "2.2"


def test_read_phase_rows_sorts_gaussian_after_fixed_pools(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "\n".join(
            [
                "condition,epoch,n,fid_mean,denoising_gap_mean,"
                "low_mid_mean_timestep_gap",
                "cosine_gaussian,100,4,2.0,0,0",
                "cosine_fixed_pool_5k,100,4,4.0,0.1,0.2",
            ]
        ),
        encoding="utf-8",
    )

    rows = read_phase_rows([f"cosine={summary}"])

    assert [row["pool_size"] for row in rows] == ["5000", ""]
