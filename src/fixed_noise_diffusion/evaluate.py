from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .diffusion import GaussianDiffusion
from .noise import GaussianNoiseSampler, FixedPoolNoiseSampler
from .utils import generator_for


NoiseSampler = GaussianNoiseSampler | FixedPoolNoiseSampler


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
    model.eval()
    timestep_generator = generator_for(device, seed)
    total_loss = 0.0
    total_count = 0
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= int(batches):
            break
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (batch_size,),
            device=device,
            generator=timestep_generator,
            dtype=torch.long,
        )
        noise = sampler.sample(batch_size)
        noisy = diffusion.q_sample(images, timesteps, noise)
        pred_noise = model(noisy, timesteps)
        loss = F.mse_loss(pred_noise, noise, reduction="mean")
        total_loss += loss.item() * batch_size
        total_count += batch_size
    if total_count == 0:
        raise ValueError("Validation loader produced no batches")
    return total_loss / total_count


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
    from torchvision.utils import save_image

    eval_cfg = config["evaluation"]
    data_cfg = config["data"]
    count = int(eval_cfg["sample_count"])
    if count <= 0:
        return torch.empty(0)
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
    save_image(samples.add(1).mul(0.5).clamp(0, 1), output_path, nrow=max(1, int(count**0.5)))
    return samples


@torch.no_grad()
def optional_fid_kid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> dict[str, float | None]:
    if fake_images.numel() == 0:
        return {"fid": None, "kid_mean": None, "kid_std": None}
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        return {"fid": None, "kid_mean": None, "kid_std": None}

    real_uint8 = _to_uint8(real_images).to(device)
    fake_uint8 = _to_uint8(fake_images).to(device)
    fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)
    fid.update(real_uint8, real=True)
    fid.update(fake_uint8, real=False)

    kid = KernelInceptionDistance(subset_size=min(50, fake_uint8.shape[0]), normalize=False).to(device)
    kid.update(real_uint8, real=True)
    kid.update(fake_uint8, real=False)
    kid_mean, kid_std = kid.compute()
    return {
        "fid": float(fid.compute().item()),
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
    }


@torch.no_grad()
def first_real_batch(loader: DataLoader, device: torch.device, count: int) -> torch.Tensor:
    images, _ = next(iter(loader))
    return images[:count].to(device, non_blocking=True)

