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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if parsed < 1 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


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

    sample_steps = _positive_int("evaluation.sample_steps", eval_cfg["sample_steps"])
    if sample_steps > int(diffusion.num_timesteps):
        raise ValueError(
            "evaluation.sample_steps must not exceed diffusion.num_timesteps; "
            f"got sample_steps={sample_steps} and num_timesteps={diffusion.num_timesteps}"
        )

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
        steps=sample_steps,
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
    strict: bool = False,
    feature: int = 64,
) -> dict[str, float | int | str | None]:
    feature = _positive_int("evaluation.fid_feature", feature)

    def empty_metrics(error: str | None = None) -> dict[str, float | int | str | None]:
        return {"fid": None, "kid_mean": None, "kid_std": None, "fid_feature": feature, "metrics_error": error}

    def failed_metrics(message: str) -> dict[str, float | int | str | None]:
        if strict:
            raise RuntimeError(message)
        return empty_metrics(message)

    if real_images.numel() == 0 or fake_images.numel() == 0:
        return failed_metrics(
            "FID/KID require non-empty real and fake image batches; "
            f"got real_images.numel()={real_images.numel()} and "
            f"fake_images.numel()={fake_images.numel()}"
        )
    min_image_count = min(int(real_images.shape[0]), int(fake_images.shape[0]))
    if min_image_count < 2:
        return failed_metrics(
            "FID/KID require at least two real and fake images; "
            f"got real_count={int(real_images.shape[0])} and "
            f"fake_count={int(fake_images.shape[0])}"
        )
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as exc:
        return failed_metrics(f"Unable to import TorchMetrics FID/KID metrics: {exc!r}")

    real_uint8 = _to_uint8(real_images).to(device)
    fake_uint8 = _to_uint8(fake_images).to(device)
    fid_value: float | None = None
    errors: list[str] = []
    try:
        fid = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)
        fid_value = float(fid.compute().item())
    except Exception as exc:
        fid_value = None
        errors.append(f"FID computation failed: {exc!r}")

    kid_mean_value: float | None = None
    kid_std_value: float | None = None
    subset_size = min(50, min_image_count)
    if subset_size >= 2:
        try:
            kid = KernelInceptionDistance(
                feature=feature,
                subset_size=subset_size, normalize=False
            ).to(device)
            kid.update(real_uint8, real=True)
            kid.update(fake_uint8, real=False)
            kid_mean, kid_std = kid.compute()
            kid_mean_value = float(kid_mean.item())
            kid_std_value = float(kid_std.item())
        except Exception as exc:
            errors.append(f"KID computation failed: {exc!r}")
            kid_mean_value = None
            kid_std_value = None

    metrics_error = "; ".join(errors) if errors else None
    if strict and metrics_error is not None:
        raise RuntimeError(metrics_error)

    return {
        "fid": fid_value,
        "kid_mean": kid_mean_value,
        "kid_std": kid_std_value,
        "fid_feature": feature,
        "metrics_error": metrics_error,
    }


@torch.no_grad()
def collect_real_images(
    loader: DataLoader, device: torch.device, count: int
) -> torch.Tensor:
    count = int(count)
    if count <= 0:
        return torch.empty(0, device=device)

    batches: list[torch.Tensor] = []
    seen = 0
    for images, _ in loader:
        remaining = count - seen
        if remaining <= 0:
            break
        selected = images[:remaining].to(device, non_blocking=True)
        batches.append(selected)
        seen += int(selected.shape[0])

    if seen < count:
        raise ValueError(
            f"Validation loader produced only {seen} real images, requested {count}"
        )
    return torch.cat(batches, dim=0)


@torch.no_grad()
def first_real_batch(
    loader: DataLoader, device: torch.device, count: int
) -> torch.Tensor:
    return collect_real_images(loader, device, count)
