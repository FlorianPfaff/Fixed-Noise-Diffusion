import json
from pathlib import Path
from typing import Any

import torch

from fixed_noise_diffusion.config import load_config
from fixed_noise_diffusion.train import train


def _tiny_config(
    tmp_path: Path, run_name: str, save_checkpoint: bool = False
) -> dict[str, Any]:
    config = load_config("smoke.yaml")
    config["run_name"] = run_name
    config["output_dir"] = str(tmp_path)
    config["device"] = "cpu"
    config["data"]["fake_train_size"] = 8
    config["data"]["fake_val_size"] = 4
    config["data"]["batch_size"] = 4
    config["data"]["eval_batch_size"] = 4
    config["evaluation"]["sample_count"] = 0
    config["training"]["save_checkpoint"] = save_checkpoint
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _key_metrics(run_dir: Path) -> dict[str, float]:
    rows = _metrics(run_dir)
    train_row = next(row for row in rows if row["type"] == "train_step")
    eval_row = next(row for row in rows if row["type"] == "eval")
    return {
        "loss": train_row["loss"],
        "train_den_loss": eval_row["train_den_loss"],
        "gaussian_den_loss": eval_row["gaussian_den_loss"],
        "denoising_gap": eval_row["denoising_gap"],
    }


def _checkpoint_model(run_dir: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        run_dir / "checkpoints" / "epoch_0001.pt",
        map_location="cpu",
    )
    return checkpoint["model"]


def test_training_writes_integrity_artifacts(tmp_path):
    run_dir = train(_tiny_config(tmp_path, "integrity_smoke"))

    metadata = _read_json(run_dir / "run_metadata.json")
    summary = _read_json(run_dir / "run_summary.json")

    assert metadata["schema_version"] == 1
    assert metadata["run_name"] == "integrity_smoke"
    assert metadata["seed"] == 123
    assert metadata["noise"]["mode"] == "fixed_pool"
    assert metadata["torch"]["version"]
    assert summary["status"] == "completed"
    assert summary["config_hash"] == metadata["config_hash"]
    assert summary["final_step"] == 1
    assert summary["last_eval"]["denoising_gap"] is not None


def test_tiny_training_reproducibility_key_metrics(tmp_path):
    run_a = train(_tiny_config(tmp_path, "repro_a", save_checkpoint=True))
    run_b = train(_tiny_config(tmp_path, "repro_b", save_checkpoint=True))

    assert _key_metrics(run_a) == _key_metrics(run_b)
    model_a = _checkpoint_model(run_a)
    model_b = _checkpoint_model(run_b)
    assert model_a.keys() == model_b.keys()
    for key, tensor in model_a.items():
        assert torch.equal(tensor, model_b[key])
