from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .diffusion import GaussianDiffusion
from .model import build_model


def parse_int_list(raw: str, *, name: str = "integer list", minimum: int | None = None) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            raise ValueError(f"{name} must contain only integer values, got {item!r}") from None
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} values must be at least {minimum}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one integer value")
    return values


def parse_positive_int_list(raw: str, *, name: str = "integer list") -> list[int]:
    return parse_int_list(raw, name=name, minimum=1)


def parse_nonnegative_int_list(raw: str, *, name: str = "integer list") -> list[int]:
    return parse_int_list(raw, name=name, minimum=0)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_checkpoint_model(
    run_dir: Path, epoch: int, device: torch.device
) -> tuple[nn.Module, GaussianDiffusion, dict[str, Any], int]:
    checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # Load on CPU first: training checkpoints contain optimizer state, and
    # mapping the entire checkpoint to CUDA can OOM before the optimizer state
    # is discarded for evaluation.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or load_yaml(run_dir / "config.yaml")
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    diffusion = GaussianDiffusion.from_config(config, device)
    return model, diffusion, config, int(checkpoint.get("step", 0))
