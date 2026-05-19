from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .diffusion import GaussianDiffusion
from .noise import FixedPoolNoiseSampler, GaussianNoiseSampler
from .utils import generator_for

NoiseSampler = GaussianNoiseSampler | FixedPoolNoiseSampler

SUPPORTED_FID_FEATURES = frozenset({64, 192, 768, 2048})


def fid_metric_name(fid_feature: int) -> str:
    return f"fid{int(fid_feature)}"


def empty_fid_kid_metrics(fid_feature: int) -> dict[str, float | int | str | None]:
    fid_feature = int(fid_feature)
    metric_name = fid_metric_name(fid_feature)
    return {
        metric_name: None,
        "fid_feature": fid_feature,
        "fid_metric_name": metric_name,
        "kid_mean": None,
        "kid_std": None,
    }


def _validate_fid_feature(fid_feature: int) -> int:
    fid_feature = int(fid_feature)
    if fid_feature not in SUPPORTED_FID_FEATURES:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_FID_FEATURES))
        raise ValueError(
            f"evaluation.fid_feature must be one of {supported}; got {fid_feature}"
        )
    return fid_feature


@torch.no_grad()
def denoising_loss_from_timesteps(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    sampler: NoiseSampler,
    device: torch.device,
    batches: int,
    make_timesteps: Callable[[int], torch.Tensor],
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= int(batches):
            break
        images = images.to(device, non_blocking=True)
        batch_size = int(images.shape[0])
        timesteps = make_timesteps(batch_size)
        noise = sampler.sample(batch_size)
        noisy = diffusion.q_sample(images, timesteps, noise)
        pred_noise = model(noisy, timesteps)
        loss = F.mse_loss(pred_noise, noise, reduction="mean")
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        raise ValueError("Validation loader produced no batches")
    return total_loss / total_count, total_count


@torch.no_grad()
def denoising_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    sampler: NoiseSampler,
    device: torch.device,
    batches: int,
    seed: int,
) -> float:
    timestep_generator = generator_for(device, seed)

    def make_random_timesteps(batch_size: int) -> torch.Tensor:
        return torch.randint(
            0,
            diffusion.num_timesteps,
            (batch_size,),
            device=device,
            generator=timestep_generator,
            dtype=torch.long,
        )

    loss, _ = denoising_loss_from_timesteps(
        model=model,
        diffusion=diffusion,
        loader=loader,
        sampler=sampler,
        device=device,
        batches=batches,
        make_timesteps=make_random_timesteps,
    )
    return loss


def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().clamp(-1, 1).add(1).mul(127.5).round()
    return images.to(torch.uint8)


@torch.no_grad()
def sample_grid(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    config: dict[str, Any],
    device: torch.device,
    output_path: Path,
    seed: int,
) -> torch.Tensor:
    eval_cfg = config["evaluation"]
    data_cfg = config["data"]
    count = int(eval_cfg["sample_count"])
    if count <= 0:
        return torch.empty(0)

    from torchvision.utils import save_image

    generator = generator_for(device, seed)
    shape = (
        count,
        int(data_cfg["channels"]),
        int(data_cfg["image_size"]),
        int(data_cfg["image_size"]),
    )
    model.eval()
    samples = diffusion.sample(
        model,
        shape=shape,
        sampler=str(eval_cfg["sampler"]),
        steps=int(eval_cfg["sample_steps"]),
        generator=generator,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        samples.add(1).mul(0.5).clamp(0, 1), output_path, nrow=max(1, int(count**0.5))
    )
    return samples


@torch.no_grad()
def optional_fid_kid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    fid_feature: int = 64,
) -> dict[str, float | int | str | None]:
    fid_feature = _validate_fid_feature(fid_feature)
    metrics = empty_fid_kid_metrics(fid_feature)
    metric_name = fid_metric_name(fid_feature)
    if fake_images.numel() == 0:
        return metrics
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        return metrics

    real_uint8 = _to_uint8(real_images).to(device)
    fake_uint8 = _to_uint8(fake_images).to(device)
    fid = FrechetInceptionDistance(feature=fid_feature, normalize=False).to(device)
    fid.update(real_uint8, real=True)
    fid.update(fake_uint8, real=False)

    kid = KernelInceptionDistance(
        subset_size=min(50, fake_uint8.shape[0]), normalize=False
    ).to(device)
    kid.update(real_uint8, real=True)
    kid.update(fake_uint8, real=False)
    kid_mean, kid_std = kid.compute()
    metrics[metric_name] = float(fid.compute().item())
    metrics["kid_mean"] = float(kid_mean.item())
    metrics["kid_std"] = float(kid_std.item())
    return metrics


@torch.no_grad()
def first_real_batch(
    loader: DataLoader, device: torch.device, count: int
) -> torch.Tensor:
    """Return exactly ``count`` real images from the loader.

    Training-time FID/KID uses this helper to match the number of real images
    to the number of generated samples. A single validation batch can be
    smaller than ``count``; silently returning that shorter batch biases the
    resulting feature statistics.
    """
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return torch.empty(0, device=device)

    real_batches = []
    total_count = 0
    for images, _ in loader:
        remaining = count - total_count
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        real_batches.append(images)
        total_count += int(images.shape[0])
        if total_count >= count:
            break

    if total_count == 0:
        raise ValueError("Validation loader produced no batches")
    if total_count < count:
        raise ValueError(
            f"Requested {count} real images for sample-quality metrics, "
            f"but the loader only produced {total_count}"
        )
    return torch.cat(real_batches, dim=0)
