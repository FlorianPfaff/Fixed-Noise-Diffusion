from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .utils import generator_for


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class NoiseInfo:
    mode: str
    pool_size: int | None
    pool_memory_mb: float
    whitened: bool


class GaussianNoiseSampler:
    def __init__(self, image_shape: tuple[int, int, int], device: torch.device, seed: int) -> None:
        self.image_shape = image_shape
        self.device = device
        self.seed = int(seed)
        self.generator = generator_for(device, self.seed)
        self.info = NoiseInfo("gaussian", None, 0.0, False)

    def sample(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            (int(batch_size), *self.image_shape),
            device=self.device,
            dtype=torch.float32,
            generator=self.generator,
        )

    def fork(self, seed: int) -> "GaussianNoiseSampler":
        return GaussianNoiseSampler(self.image_shape, self.device, seed)


class FixedPoolNoiseSampler:
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        device: torch.device,
        pool_size: int,
        pool_seed: int,
        index_seed: int,
        dtype: str = "float16",
        chunk_size: int = 8192,
        whiten: bool = False,
        existing_pool: torch.Tensor | None = None,
    ) -> None:
        self.image_shape = image_shape
        self.device = device
        self.pool_size = int(pool_size)
        self.pool_seed = int(pool_seed)
        self.index_seed = int(index_seed)
        self.dtype = DTYPES[str(dtype)]
        self.chunk_size = int(chunk_size)
        self.whiten = bool(whiten)
        self.index_generator = torch.Generator(device="cpu")
        self.index_generator.manual_seed(self.index_seed)

        if existing_pool is None:
            self.pool = self._build_pool()
        else:
            self.pool = existing_pool
        pool_memory_mb = self.pool.numel() * self.pool.element_size() / (1024**2)
        mode = "fixed_pool_whitened" if self.whiten else "fixed_pool"
        self.info = NoiseInfo(mode, self.pool_size, pool_memory_mb, self.whiten)

    def _build_pool(self) -> torch.Tensor:
        pool = torch.empty((self.pool_size, *self.image_shape), dtype=self.dtype, device="cpu")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.pool_seed)
        for start in range(0, self.pool_size, self.chunk_size):
            end = min(start + self.chunk_size, self.pool_size)
            chunk = torch.randn(
                (end - start, *self.image_shape),
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            )
            pool[start:end].copy_(chunk.to(dtype=self.dtype))
        if self.whiten:
            # Per-coordinate moment standardization removes trivial realized-pool mean/std bias.
            work = pool.float()
            mean = work.mean(dim=0, keepdim=True)
            std = work.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            pool = ((work - mean) / std).to(dtype=self.dtype)
        return pool.pin_memory() if torch.cuda.is_available() else pool

    def sample(self, batch_size: int) -> torch.Tensor:
        indices = torch.randint(
            0,
            self.pool_size,
            (int(batch_size),),
            generator=self.index_generator,
            device="cpu",
        )
        return self.pool.index_select(0, indices).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )

    def fork(self, seed: int) -> "FixedPoolNoiseSampler":
        return FixedPoolNoiseSampler(
            image_shape=self.image_shape,
            device=self.device,
            pool_size=self.pool_size,
            pool_seed=self.pool_seed,
            index_seed=seed,
            dtype=str(self.pool.dtype).replace("torch.", ""),
            chunk_size=self.chunk_size,
            whiten=self.whiten,
            existing_pool=self.pool,
        )


def make_noise_sampler(
    config: dict[str, Any],
    device: torch.device,
    purpose_seed_offset: int = 0,
    existing_pool_sampler: FixedPoolNoiseSampler | None = None,
) -> GaussianNoiseSampler | FixedPoolNoiseSampler:
    data_cfg = config["data"]
    noise_cfg = config["noise"]
    seed = int(config["seed"]) + int(purpose_seed_offset)
    image_shape = (
        int(data_cfg["channels"]),
        int(data_cfg["image_size"]),
        int(data_cfg["image_size"]),
    )
    mode = str(noise_cfg["mode"])
    if mode == "gaussian":
        return GaussianNoiseSampler(image_shape, device, seed)
    if mode in {"fixed_pool", "fixed_pool_whitened"}:
        if noise_cfg["pool_size"] is None:
            raise ValueError("noise.pool_size is required for fixed_pool modes")
        whiten = bool(noise_cfg.get("whiten", False)) or mode == "fixed_pool_whitened"
        if existing_pool_sampler is not None:
            return existing_pool_sampler.fork(seed)
        return FixedPoolNoiseSampler(
            image_shape=image_shape,
            device=device,
            pool_size=int(noise_cfg["pool_size"]),
            pool_seed=int(noise_cfg["pool_seed"]),
            index_seed=seed,
            dtype=str(noise_cfg.get("pool_dtype", "float16")),
            chunk_size=int(noise_cfg.get("pool_chunk_size", 8192)),
            whiten=whiten,
        )
    raise ValueError(f"Unsupported noise mode {mode!r}")

