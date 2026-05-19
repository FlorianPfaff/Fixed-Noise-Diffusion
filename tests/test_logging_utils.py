import csv
import pytest

from fixed_noise_diffusion.logging_utils import MetricLogger


def test_metric_logger_writes_denoising_eval_timestep_seed_to_csv(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log(
        {
            "type": "eval",
            "epoch": 7,
            "step": 123,
            "denoising_eval_timestep_seed": 30010,
        }
    )

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["denoising_eval_timestep_seed"] == "30010"


def test_metric_logger_refuses_existing_metric_files_by_default(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to append"):
        MetricLogger(tmp_path)

    assert metrics_path.read_text(encoding="utf-8") == "stale\n"


def test_metric_logger_append_is_explicit(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log({"type": "eval", "epoch": 1, "step": 1})

    resumed_logger = MetricLogger(tmp_path, append=True)
    resumed_logger.log({"type": "run_end", "epoch": 1, "step": 2})

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["type"] for row in rows] == ["eval", "run_end"]


def test_metric_logger_append_validates_existing_csv_schema(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "type,epoch\neval,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible"):
        MetricLogger(tmp_path, append=True)


def test_metric_logger_fresh_run_creates_empty_jsonl(tmp_path):
    MetricLogger(tmp_path)

    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == ""
