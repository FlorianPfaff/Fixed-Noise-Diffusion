from __future__ import annotations

import argparse
import re
from pathlib import Path

RUN_RE = re.compile(r"wp2_(?:\d+ep)_(?P<condition>.+)_seed(?P<seed>\d+)$")


def run_identity(run_dir: Path) -> tuple[str, int]:
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return run_dir.name, -1
    return match.group("condition"), int(match.group("seed"))


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
