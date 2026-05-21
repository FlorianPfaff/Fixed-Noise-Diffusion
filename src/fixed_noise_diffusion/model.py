from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def _groups(channels: int) -> int:
    for group_count in (32, 16, 8, 4, 2, 1):
        if channels % group_count == 0:
            return group_count
    return 1


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


def _probability_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float in [0, 1], got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite float in [0, 1], got {value!r}") from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1], got {value!r}")
    return parsed


def _validate_channel_mults(value: Sequence[int]) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("model.channel_mults must be a non-empty sequence of positive integers")
    if not value:
        raise ValueError("model.channel_mults must contain at least one value")
    return [
        _positive_int(f"model.channel_mults[{index}]", mult)
        for index, mult in enumerate(value)
    ]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim < 1:
            raise ValueError("time_emb_dim must be positive")

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return timesteps.float().unsqueeze(1)

        half = self.dim // 2
        if half == 1:
            freqs = torch.ones(1, device=timesteps.device, dtype=torch.float32)
        else:
            exponent = (
                -math.log(10_000)
                * torch.arange(
                    half,
                    device=timesteps.device,
                    dtype=torch.float32,
                )
                / (half - 1)
            )
            freqs = exponent.exp()
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.dim - embedding.shape[1])
            )
        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 2, 4),
        time_emb_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Conv2d(
            image_channels, base_channels, kernel_size=3, padding=1
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        dims = [base_channels, *[base_channels * int(mult) for mult in channel_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        if not in_out:
            raise ValueError("channel_mults must contain at least one value")

        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_in, dim_out, time_emb_dim, dropout),
                        ResidualBlock(dim_out, dim_out, time_emb_dim, dropout),
                        nn.Identity() if is_last else Downsample(dim_out),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim, dropout)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim, dropout)

        self.ups = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = index == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_out + dim_out, dim_in, time_emb_dim, dropout),
                        ResidualBlock(dim_in + dim_out, dim_in, time_emb_dim, dropout),
                        nn.Identity() if is_last else Upsample(dim_in),
                    ]
                )
            )

        self.final_block = ResidualBlock(
            base_channels * 2, base_channels, time_emb_dim, dropout
        )
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        residual = self.init_conv(x)
        x = residual
        time_emb = self.time_mlp(timesteps)

        skips: list[torch.Tensor] = []
        for block1, block2, downsample in self.downs:
            x = block1(x, time_emb)
            skips.append(x)
            x = block2(x, time_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_block2(x, time_emb)

        for block1, block2, upsample in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, time_emb)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block2(x, time_emb)
            x = upsample(x)

        x = torch.cat([x, residual], dim=1)
        x = self.final_block(x, time_emb)
        return self.final_conv(x)



def _validate_image_size(image_size: object, channel_mults: Sequence[int]) -> int:
    image_size = _positive_int("data.image_size", image_size)
    downsample_factor = 2 ** max(0, len(channel_mults) - 1)
    if image_size < downsample_factor or image_size % downsample_factor != 0:
        raise ValueError(
            "data.image_size must be divisible by the UNet downsampling factor "
            f"{downsample_factor} for model.channel_mults={list(channel_mults)!r}"
        )
    return image_size


def build_model(config: dict) -> UNet:
    data_cfg = config["data"]
    model_cfg = config["model"]
    channel_mults = _validate_channel_mults(model_cfg["channel_mults"])
    image_size = _validate_image_size(data_cfg["image_size"], channel_mults)
    return UNet(
        image_channels=_positive_int("data.channels", data_cfg["channels"]),
        base_channels=_positive_int("model.base_channels", model_cfg["base_channels"]),
        channel_mults=channel_mults,
        time_emb_dim=_positive_int("model.time_emb_dim", model_cfg["time_emb_dim"]),
        dropout=_probability_float("model.dropout", model_cfg["dropout"]),
    )
