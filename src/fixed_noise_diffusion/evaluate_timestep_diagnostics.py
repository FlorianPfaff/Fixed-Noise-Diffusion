from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn

from .checkpoints import load_checkpoint_model, parse_int_list
from .data import make_dataloaders
from .diffusion import GaussianDiffusion
from .evaluate import denoising_loss_from_timesteps
from .noise import GaussianNoiseSampler, make_noise_sampler
from .summarize_sample_quality import condition_kind, condition_pool_size
from .utils import resolve_device, seed_everything

RUN_RE = re.compile(r"wp2_(?:\d+ep)_(?P<condition>.+)_seed(?P<seed>\d+)$")


def _prepare_config(
    config: dict[str, Any],
    batch_size: int,
    batches: int,
    data_dir: str | None,
    num_workers: int,
) -> dict[str, Any]:
    config = deepcopy(config)
    data_cfg = config["data"]
    data_cfg["download"] = True
    data_cfg["eval_batch_size"] = int(batch_size)
    data_cfg["num_workers"] = int(num_workers)
    if data_dir is not None:
        data_cfg["data_dir"] = data_dir
    requested = int(batch_size) * int(batches)
    current_subset = data_cfg.get("eval_subset_size")
    if current_subset is not None:
        data_cfg["eval_subset_size"] = max(int(current_subset), requested)
    return config


@torch.no_grad()
def fixed_timestep_denoising_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader,
    sampler,
    device: torch.device,
    timestep: int,
    batches: int,
) -> tuple[float, int]:
    def make_fixed_timesteps(batch_size: int) -> torch.Tensor:
        return torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)

    return denoising_loss_from_timesteps(
        model=model,
        diffusion=diffusion,
        loader=loader,
        sampler=sampler,
        device=device,
        batches=batches,
        make_timesteps=make_fixed_timesteps,
    )


def _run_identity(run_dir: Path) -> tuple[str, int]:
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return run_dir.name, -1
    return match.group("condition"), int(match.group("seed"))


def evaluate_run(
    run_dir: Path,
    epochs: list[int],
    timesteps: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = resolve_device(args.device)
    condition, run_seed = _run_identity(run_dir)
    rows: list[dict[str, Any]] = []

    model, diffusion, config, step = load_checkpoint_model(run_dir, epochs[0], device)
    config = _prepare_config(
        config, args.batch_size, args.batches, args.data_dir, args.num_workers
    )
    seed = int(config["seed"])
    seed_everything(args.seed + seed + max(run_seed, 0) * 1000)
    loaders = make_dataloaders(config)
    train_base_sampler = make_noise_sampler(config, device, purpose_seed_offset=0)

    for epoch_index, epoch in enumerate(epochs):
        if epoch_index > 0:
            model, diffusion, config, step = load_checkpoint_model(
                run_dir, epoch, device
            )
        for timestep in timesteps:
            if timestep < 0 or timestep >= diffusion.num_timesteps:
                raise ValueError(
                    f"timestep {timestep} is outside [0, {diffusion.num_timesteps})"
                )
            start = time.perf_counter()
            train_sampler = train_base_sampler.fork(
                seed + 10_000 + epoch * 1000 + timestep
            )
            gaussian_sampler = GaussianNoiseSampler(
                image_shape=train_base_sampler.image_shape,
                device=device,
                seed=seed + 20_000 + epoch * 1000 + timestep,
            )
            train_loss, image_count = fixed_timestep_denoising_loss(
                model=model,
                diffusion=diffusion,
                loader=loaders.val,
                sampler=train_sampler,
                device=device,
                timestep=timestep,
                batches=args.batches,
            )
            gaussian_loss, gaussian_image_count = fixed_timestep_denoising_loss(
                model=model,
                diffusion=diffusion,
                loader=loaders.val,
                sampler=gaussian_sampler,
                device=device,
                timestep=timestep,
                batches=args.batches,
            )
            if image_count != gaussian_image_count:
                raise RuntimeError(
                    "Train-law and Gaussian evaluations used unequal data"
                )
            info = train_base_sampler.info
            rows.append(
                {
                    "run_name": run_dir.name,
                    "condition": condition,
                    "kind": condition_kind(condition),
                    "pool_size": "" if info.pool_size is None else info.pool_size,
                    "seed": run_seed,
                    "epoch": epoch,
                    "step": step,
                    "timestep": timestep,
                    "batches": int(args.batches),
                    "images": image_count,
                    "train_noise_loss": train_loss,
                    "gaussian_noise_loss": gaussian_loss,
                    "timestep_gap": gaussian_loss - train_loss,
                    "noise_mode": info.mode,
                    "pool_memory_mb": round(info.pool_memory_mb, 3),
                    "seconds": round(time.perf_counter() - start, 3),
                    "source_run_dir": str(run_dir),
                }
            )
    return rows


def _append_records(
    csv_path: Path, jsonl_path: Path, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sample_mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return mean(clean) if clean else math.nan


def _sample_std(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return stdev(clean) if len(clean) >= 2 else math.nan


def _format_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.12g}"


def _float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def summarize_timestep_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]]
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["kind"]),
                str(row["condition"]),
                str(row.get("pool_size", "")),
                str(row["epoch"]),
                str(row["timestep"]),
            )
        ].append(row)

    summary: list[dict[str, str]] = []
    for (kind, condition, pool_size, epoch, timestep), group in grouped.items():
        train_losses = [_float_or_nan(row["train_noise_loss"]) for row in group]
        gaussian_losses = [_float_or_nan(row["gaussian_noise_loss"]) for row in group]
        gaps = [_float_or_nan(row["timestep_gap"]) for row in group]
        summary.append(
            {
                "kind": kind,
                "condition": condition,
                "pool_size": pool_size,
                "epoch": epoch,
                "timestep": timestep,
                "n": str(len(group)),
                "train_noise_loss_mean": _format_float(_sample_mean(train_losses)),
                "train_noise_loss_std": _format_float(_sample_std(train_losses)),
                "gaussian_noise_loss_mean": _format_float(
                    _sample_mean(gaussian_losses)
                ),
                "gaussian_noise_loss_std": _format_float(_sample_std(gaussian_losses)),
                "timestep_gap_mean": _format_float(_sample_mean(gaps)),
                "timestep_gap_std": _format_float(_sample_std(gaps)),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            row["kind"],
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
            int(row["epoch"]),
            int(row["timestep"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_timestep_gaps(summary: list[dict[str, str]], output: Path) -> None:
    if not summary:
        return
    final_epoch = max(int(row["epoch"]) for row in summary)
    rows = [row for row in summary if int(row["epoch"]) == final_epoch]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for condition, group in sorted(
        grouped.items(),
        key=lambda item: (
            condition_kind(item[0]),
            condition_pool_size(item[0]) or 10**18,
            item[0],
        ),
    ):
        group = sorted(group, key=lambda row: int(row["timestep"]))
        label = condition.replace("strong96_", "")
        axis.plot(
            [int(row["timestep"]) for row in group],
            [_float_or_nan(row["timestep_gap_mean"]) for row in group],
            marker="o",
            label=label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Diffusion timestep")
    axis.set_ylabel("Fresh Gaussian loss - train-law loss")
    axis.set_title(f"Timestep-local denoising gap at epoch {final_epoch}")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _select_runs(sweep_dirs: list[Path], run_names: list[str]) -> list[Path]:
    runs: list[Path] = []
    for sweep_dir in sweep_dirs:
        sweep_dir = sweep_dir.expanduser().resolve()
        if run_names:
            runs.extend(sweep_dir / name for name in run_names)
        else:
            runs.extend(sorted(path for path in sweep_dir.iterdir() if path.is_dir()))
    seen: set[Path] = set()
    unique: list[Path] = []
    for run in runs:
        if run not in seen:
            seen.add(run)
            unique.append(run)
    missing = [str(path) for path in unique if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate timestep-local fixed-noise denoising diagnostics."
    )
    parser.add_argument(
        "--sweep-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing saved run folders. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="Run directory name")
    parser.add_argument("--epochs", default="50,100")
    parser.add_argument("--timesteps", default="0,25,50,100,200,400,600,800,999")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    csv_path = output_dir / "timestep_diagnostics.csv"
    jsonl_path = output_dir / "timestep_diagnostics.jsonl"
    epochs = parse_int_list(args.epochs)
    timesteps = parse_int_list(args.timesteps)
    all_rows: list[dict[str, Any]] = []

    for run_dir in _select_runs(args.sweep_dir, args.run):
        rows = evaluate_run(run_dir, epochs, timesteps, args)
        _append_records(csv_path, jsonl_path, rows)
        all_rows.extend(rows)
        print(json.dumps({"run_name": run_dir.name, "rows": len(rows)}), flush=True)

    summary = summarize_timestep_rows(all_rows)
    _write_csv(output_dir / "timestep_diagnostics_summary.csv", summary)
    plot_timestep_gaps(summary, output_dir / "timestep_gap_by_timestep.png")


if __name__ == "__main__":
    main()
