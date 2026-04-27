from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "run",
    "output_dir": "runs",
    "seed": 0,
    "device": "cuda",
    "data": {
        "dataset": "cifar10",
        "data_dir": "data",
        "download": True,
        "image_size": 32,
        "channels": 3,
        "batch_size": 128,
        "eval_batch_size": 128,
        "num_workers": 4,
        "subset_size": None,
        "eval_subset_size": 2048,
        "fake_train_size": 1024,
        "fake_val_size": 256,
    },
    "model": {
        "base_channels": 64,
        "channel_mults": [1, 2, 2, 4],
        "time_emb_dim": 256,
        "dropout": 0.1,
    },
    "diffusion": {
        "num_timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
    },
    "training": {
        "epochs": 50,
        "max_train_steps": None,
        "lr": 0.0002,
        "weight_decay": 0.0,
        "grad_accum_steps": 1,
        "amp": True,
        "log_interval_steps": 100,
        "checkpoint_epochs": [1, 5, 10, 25, 50],
        "save_checkpoint": True,
    },
    "evaluation": {
        "denoising_batches": 16,
        "sample_count": 64,
        "sample_steps": 50,
        "sampler": "ddim",
        "enable_metrics": False,
    },
    "noise": {
        "mode": "gaussian",
        "pool_size": None,
        "pool_seed": 4242,
        "pool_dtype": "float16",
        "pool_chunk_size": 8192,
        "whiten": False,
    },
}

PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml_with_inherits(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent = data.pop("inherits", None)
    if parent is None:
        return data
    parent_path = (path.parent / parent).resolve()
    parent_data = _read_yaml_with_inherits(parent_path)
    return deep_update(parent_data, data)


def parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be KEY=VALUE, got {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
        if not isinstance(cursor, dict):
            raise ValueError(
                f"Cannot set nested key under non-dict override {dotted_key!r}"
            )
    cursor[keys[-1]] = parse_value(raw_value)


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    if candidate.is_absolute():
        return candidate

    package_candidates = [PACKAGE_CONFIG_DIR / candidate]
    if len(candidate.parts) > 1 and candidate.parts[0].lower() == "configs":
        package_candidates.append(PACKAGE_CONFIG_DIR / Path(*candidate.parts[1:]))
    package_candidates.append(PACKAGE_CONFIG_DIR / candidate.name)

    for package_candidate in package_candidates:
        if package_candidate.is_file():
            return package_candidate.resolve()

    return candidate.resolve()


def load_config(
    path: str | Path | None, overrides: list[str] | None = None
) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        file_config = _read_yaml_with_inherits(resolve_config_path(path))
        deep_update(config, file_config)
    for override in overrides or []:
        apply_override(config, override)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config or a packaged config name.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config keys, e.g. --set training.epochs=5",
    )
