from fixed_noise_diffusion.summarize_pool_dtype_control import summarize


def _row(*, dtype: str, mode: str, whiten: bool, gap: float) -> dict:
    return {
        "dataset": "cifar10",
        "pool_size": 1000,
        "pool_dtype": dtype,
        "noise_mode": mode,
        "whiten": whiten,
        "epoch": 100,
        "denoising_gap": gap,
    }


def test_dtype_control_summary_does_not_average_raw_and_whitened_protocols():
    rows = [
        _row(dtype="float16", mode="fixed_pool", whiten=False, gap=0.20),
        _row(dtype="float16", mode="fixed_pool", whiten=True, gap=0.80),
        _row(dtype="float32", mode="fixed_pool", whiten=False, gap=0.25),
        _row(dtype="float32", mode="fixed_pool", whiten=True, gap=0.90),
    ]

    summary = summarize(rows)
    by_protocol = {
        (row["pool_dtype"], row["noise_mode"], row["whiten"]): row
        for row in summary
    }

    assert len(summary) == 4
    assert by_protocol[("float16", "fixed_pool", False)]["denoising_gap_mean"] == 0.20
    assert by_protocol[("float16", "fixed_pool", True)]["denoising_gap_mean"] == 0.80
    assert by_protocol[("float32", "fixed_pool", False)]["float32_minus_float16_gap_mean"] == 0.05
    assert by_protocol[("float16", "fixed_pool", True)]["float32_minus_float16_gap_mean"] == 0.10


def test_dtype_control_summary_does_not_average_noise_modes():
    rows = [
        _row(dtype="float16", mode="fixed_pool", whiten=False, gap=0.20),
        _row(dtype="float16", mode="fixed_pool_whitened", whiten=True, gap=0.80),
    ]

    summary = summarize(rows)

    assert len(summary) == 2
    assert {row["noise_mode"] for row in summary} == {"fixed_pool", "fixed_pool_whitened"}
