from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .diffusion import GaussianDiffusion
from .model import build_model


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_checkpoint_model(
    run_dir: Path, epoch: int, device: torch.device
) -> tuple[nn.Module, GaussianDiffusion, dict[str, Any], int]:
    checkpoint_path = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config") or load_yaml(run_dir / "config.yaml")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion.from_config(config, device)
    return model, diffusion, config, int(checkpoint.get("step", 0))
