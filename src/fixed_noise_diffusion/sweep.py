from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"wp2_(?:\d+ep)_(?P<condition>.+)_seed(?P<seed>\d+)$")


def pool_size_label(pool_size: int) -> str:
    if pool_size >= 1000 and pool_size % 1000 == 0:
        return f"{pool_size // 1000}k"
    return str(pool_size)


def condition_from_config(config: dict[str, Any]) -> str:
    raw_noise_cfg = config.get("noise", {})
    noise_cfg = raw_noise_cfg if isinstance(raw_noise_cfg, dict) else {}
    mode = str(noise_cfg.get("mode", "")).strip()
    pool_size = noise_cfg.get("pool_size")
    if mode == "gaussian":
        return "gaussian"
    if pool_size in (None, ""):
        return "unknown"

    label = pool_size_label(int(pool_size))
    if mode == "fixed_pool_whitened" or bool(noise_cfg.get("whiten", False)):
        return f"fixed_pool_whitened_{label}"
    if mode == "fixed_pool":
        return f"fixed_pool_{label}"
    return mode or "unknown"


def run_identity(run_dir: Path) -> tuple[str, int]:
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return run_dir.name, -1
    return match.group("condition"), int(match.group("seed"))


def run_identity_from_config(run_dir: Path, config: dict[str, Any]) -> tuple[str, int]:
    condition, seed = run_identity(run_dir)
    if seed != -1:
        return condition, seed
    try:
        run_seed = int(config.get("seed", -1))
    except (TypeError, ValueError):
        run_seed = -1
    return condition_from_config(config), run_seed


def select_run_dirs(sweep_dirs: list[Path], run_names: list[str]) -> list[Path]:
    runs: list[Path] = []
    for sweep_dir in sweep_dirs:
        root = sweep_dir.expanduser().resolve()
        if run_names:
            runs.extend(root / name for name in run_names)
        else:
            runs.extend(sorted(path for path in root.iterdir() if path.is_dir()))

    unique: list[Path] = []
    seen: set[Path] = set()
    for run in runs:
        if run not in seen:
            seen.add(run)
            unique.append(run)

    missing = [str(path) for path in unique if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    return unique


def add_common_sweep_eval_args(
    parser: argparse.ArgumentParser,
    *,
    default_epochs: str,
) -> None:
    parser.add_argument(
        "--sweep-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing saved run folders. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="Run directory name")
    parser.add_argument("--epochs", default=default_epochs)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
