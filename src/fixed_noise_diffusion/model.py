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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half <= 1:
            return timesteps.float().unsqueeze(1)
        exponent = -math.log(10_000) * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        ) / (half - 1)
        freqs = exponent.exp()
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
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
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

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

        self.final_block = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, dropout)
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


def build_model(config: dict) -> UNet:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return UNet(
        image_channels=int(data_cfg["channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=model_cfg["channel_mults"],
        time_emb_dim=int(model_cfg["time_emb_dim"]),
        dropout=float(model_cfg["dropout"]),
    )

