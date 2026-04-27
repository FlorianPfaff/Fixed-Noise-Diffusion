from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def _linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps: int,
        beta_schedule: str,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        if beta_schedule == "linear":
            betas = _linear_beta_schedule(self.num_timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.device = device
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(device)
        self.posterior_log_variance_clipped = posterior_variance.clamp_min(1e-20).log().to(device)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> "GaussianDiffusion":
        cfg = config["diffusion"]
        return cls(
            num_timesteps=int(cfg["num_timesteps"]),
            beta_schedule=str(cfg["beta_schedule"]),
            beta_start=float(cfg["beta_start"]),
            beta_end=float(cfg["beta_end"]),
            device=device,
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_noise = model(x_t, timesteps)
        x_start = self.predict_start_from_noise(x_t, timesteps, pred_noise)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        model_mean = (
            _extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_log_variance = _extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        sampler: Literal["ddpm", "ddim"] = "ddim",
        steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sampler == "ddpm":
            return self._sample_ddpm(model, shape, generator)
        if sampler == "ddim":
            return self._sample_ddim(model, shape, steps, eta, generator)
        raise ValueError(f"Unsupported sampler {sampler!r}")

    @torch.no_grad()
    def _sample_ddpm(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        image = torch.randn(shape, device=self.device, generator=generator)
        for time_index in reversed(range(self.num_timesteps)):
            timesteps = torch.full((shape[0],), time_index, device=self.device, dtype=torch.long)
            model_mean, model_log_variance = self.p_mean_variance(model, image, timesteps)
            if time_index == 0:
                noise = torch.zeros_like(image)
            else:
                noise = torch.randn(shape, device=self.device, generator=generator)
            image = model_mean + (0.5 * model_log_variance).exp() * noise
        return image.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _sample_ddim(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        steps: int,
        eta: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        steps = max(1, min(int(steps), self.num_timesteps))
        times = torch.linspace(
            self.num_timesteps - 1,
            0,
            steps,
            device=self.device,
            dtype=torch.long,
        )
        prev_times = torch.cat([times[1:], torch.full((1,), -1, device=self.device, dtype=torch.long)])
        image = torch.randn(shape, device=self.device, generator=generator)

        for time_index, prev_time_index in zip(times.tolist(), prev_times.tolist()):
            timesteps = torch.full((shape[0],), time_index, device=self.device, dtype=torch.long)
            pred_noise = model(image, timesteps)
            alpha = self.alphas_cumprod[time_index]
            alpha_prev = (
                self.alphas_cumprod[prev_time_index]
                if prev_time_index >= 0
                else torch.ones((), device=self.device)
            )
            pred_x0 = (image - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
            sigma = eta * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
            direction_scale = (1 - alpha_prev - sigma.square()).clamp_min(0).sqrt()
            if prev_time_index >= 0:
                noise = torch.randn(shape, device=self.device, generator=generator)
            else:
                noise = torch.zeros_like(image)
            image = alpha_prev.sqrt() * pred_x0 + direction_scale * pred_noise + sigma * noise
        return image.clamp(-1.0, 1.0)

