from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .checkpoints import load_checkpoint_model, parse_int_list
from .data import make_dataloaders
from .diffusion import GaussianDiffusion
from .evaluate import _to_uint8
from .metrics_utils import effective_kid_subset_size
from .sweep import run_identity_from_config
from .utils import generator_for, resolve_device, seed_everything

RUN_RE = re.compile(r"wp2_(?:\d+ep)_(?P<condition>.+)_seed(?P<seed>\d+)$")


def _parse_run_metadata(run_name: str) -> tuple[str, int | None]:
    match = RUN_RE.match(run_name)
    if match:
        return match.group("condition"), int(match.group("seed"))
    return run_name, None


def _evaluation_seed(
    base_seed: int, run_seed: int | None, epoch: int, *, offset: int = 0
) -> int:
    seed_component = 0 if run_seed is None else int(run_seed)
    return int(base_seed) + int(offset) + seed_component * 1000 + int(epoch)


def _prepare_config(
    config: dict[str, Any],
    sample_count: int,
    real_count: int,
    sample_batch_size: int,
    sample_steps: int,
    sampler: str,
    real_split: str,
    download_data: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    if download_data:
        data_cfg["download"] = True
    data_cfg["eval_batch_size"] = int(sample_batch_size)
    if real_split == "val":
        if data_cfg.get("eval_subset_size") is None:
            data_cfg["eval_subset_size"] = int(real_count)
        else:
            data_cfg["eval_subset_size"] = max(
                int(data_cfg["eval_subset_size"]), real_count
            )
    elif data_cfg.get("subset_size") is not None:
        data_cfg["subset_size"] = max(int(data_cfg["subset_size"]), real_count)
    eval_cfg["sample_count"] = int(sample_count)
    eval_cfg["sample_steps"] = int(sample_steps)
    eval_cfg["sampler"] = sampler
    return config


@torch.no_grad()
def _update_real_metrics(
    fid,
    kid,
    config: dict[str, Any],
    device: torch.device,
    count: int,
    split: str,
) -> int:
    loaders = make_dataloaders(config)
    data_cfg = config["data"]
    dataset = loaders.train.dataset if split == "train" else loaders.val.dataset
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    seen = 0
    for images, _ in loader:
        if seen >= count:
            break
        remaining = count - seen
        images = images[:remaining].to(device, non_blocking=True)
        real_uint8 = _to_uint8(images).to(device)
        fid.update(real_uint8, real=True)
        if kid is not None:
            kid.update(real_uint8, real=True)
        seen += int(images.shape[0])
    if seen < count:
        raise ValueError(f"Only collected {seen} real images, requested {count}")
    return seen


@torch.no_grad()
def _generate_fake_metrics(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    config: dict[str, Any],
    device: torch.device,
    fid,
    kid,
    sample_count: int,
    sample_batch_size: int,
    sample_steps: int,
    sampler: str,
    seed: int,
    grid_count: int,
    grid_path: Path,
) -> int:
    data_cfg = config["data"]
    image_shape = (
        int(data_cfg["channels"]),
        int(data_cfg["image_size"]),
        int(data_cfg["image_size"]),
    )
    generator = generator_for(device, seed)
    generated = 0
    grid_samples: list[torch.Tensor] = []
    while generated < sample_count:
        batch_size = min(sample_batch_size, sample_count - generated)
        samples = diffusion.sample(
            model,
            shape=(batch_size, *image_shape),
            sampler=sampler,
            steps=sample_steps,
            generator=generator,
        )
        fake_uint8 = _to_uint8(samples).to(device)
        fid.update(fake_uint8, real=False)
        if kid is not None:
            kid.update(fake_uint8, real=False)
        if len(grid_samples) < grid_count:
            needed = grid_count - len(grid_samples)
            grid_samples.extend(samples[:needed].detach().cpu())
        generated += int(batch_size)

    if grid_samples:
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid = torch.stack(grid_samples).add(1).mul(0.5).clamp(0, 1)
        save_image(grid, grid_path, nrow=max(1, int(len(grid_samples) ** 0.5)))
    return generated


def evaluate_run_epoch(
    run_dir: Path,
    epoch: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    start = time.perf_counter()
    device = resolve_device(args.device)
    model, diffusion, config, step = load_checkpoint_model(run_dir, epoch, device)
    condition, run_seed = run_identity_from_config(run_dir, config)
    seed_everything(args.seed + max(run_seed, 0) * 1000 + epoch)
    config = _prepare_config(
        config,
        sample_count=args.sample_count,
        real_count=args.real_count or args.sample_count,
        sample_batch_size=args.sample_batch_size,
        sample_steps=args.sample_steps,
        sampler=args.sampler,
        real_split=args.real_split,
        download_data=bool(args.download_data),
    )

    requested_real_count = int(args.real_count or args.sample_count)
    fid = FrechetInceptionDistance(feature=args.fid_feature, normalize=False).to(device)
    kid_subset_size = effective_kid_subset_size(
        args.kid_subset_size,
        args.sample_count,
        requested_real_count,
    )
    kid = None
    if kid_subset_size >= 2:
        kid = KernelInceptionDistance(
            feature=args.fid_feature,
            subset_size=kid_subset_size,
            normalize=False,
        ).to(device)
    real_count = _update_real_metrics(
        fid,
        kid,
        config,
        device,
        requested_real_count,
        args.real_split,
    )
    grid_path = output_dir / run_dir.name / f"epoch_{epoch:04d}_samples.png"
    fake_count = _generate_fake_metrics(
        model=model,
        diffusion=diffusion,
        config=config,
        device=device,
        fid=fid,
        kid=kid,
        sample_count=args.sample_count,
        sample_batch_size=args.sample_batch_size,
        sample_steps=args.sample_steps,
        sampler=args.sampler,
        seed=_evaluation_seed(args.seed, run_seed, epoch, offset=50_000),
        grid_count=args.grid_count,
        grid_path=grid_path,
    )
    kid_mean = kid_std = None
    if kid is not None:
        kid_mean, kid_std = kid.compute()
    seconds = time.perf_counter() - start
    data_cfg = config.get("data", {})
    noise_cfg = config.get("noise", {})
    return {
        "run_name": run_dir.name,
        "dataset": str(data_cfg.get("dataset", "")).lower(),
        "condition": condition,
        "seed": run_seed if run_seed is not None else -1,
        "noise_mode": str(noise_cfg.get("mode", "")),
        "pool_size": "" if noise_cfg.get("pool_size") is None else noise_cfg.get("pool_size"),
        "pool_dtype": str(noise_cfg.get("pool_dtype", "")),
        "epoch": epoch,
        "step": step,
        "sample_count": args.sample_count,
        "requested_real_count": requested_real_count,
        "real_split": args.real_split,
        "real_count": real_count,
        "fake_count": fake_count,
        "sample_steps": args.sample_steps,
        "sampler": args.sampler,
        "fid_feature": args.fid_feature,
        "kid_subset_size": kid_subset_size,
        "fid": float(fid.compute().item()),
        "kid_mean": None if kid_mean is None else float(kid_mean.item()),
        "kid_std": None if kid_std is None else float(kid_std.item()),
        "seconds": round(seconds, 3),
        "grid_path": str(grid_path),
    }


def _append_record(csv_path: Path, jsonl_path: Path, record: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _select_runs(sweep_dir: Path, run_names: list[str]) -> list[Path]:
    if run_names:
        runs = [sweep_dir / name for name in run_names]
    else:
        runs = sorted(path for path in sweep_dir.iterdir() if path.is_dir())
    missing = [str(path) for path in runs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated sample quality from saved WP2 checkpoints."
    )
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="Run directory name")
    parser.add_argument("--epochs", default="1,5,10,25,50")
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument(
        "--real-count",
        type=int,
        default=None,
        help="Number of real images for FID/KID. Defaults to --sample-count.",
    )
    parser.add_argument(
        "--real-split",
        choices=["val", "train"],
        default="val",
        help="CIFAR split to use for real FID/KID statistics.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Allow torchvision dataset downloads during evaluation. By default, "
        "the checkpoint config's data.download setting is preserved.",
    )
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--grid-count", type=int, default=64)
    parser.add_argument("--fid-feature", type=int, default=64)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / "sample_quality.csv"
    jsonl_path = output_dir / "sample_quality.jsonl"
    epochs = parse_int_list(args.epochs)

    for run_dir in _select_runs(sweep_dir, args.run):
        for epoch in epochs:
            record = evaluate_run_epoch(run_dir, epoch, output_dir, args)
            _append_record(csv_path, jsonl_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
