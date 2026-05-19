from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

from .utils import generator_for

DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _parse_pool_dtype(dtype: str) -> torch.dtype:
    dtype_name = str(dtype)
    try:
        return DTYPES[dtype_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPES))
        raise ValueError(
            f"Unsupported noise.pool_dtype {dtype_name!r}; expected one of: {supported}"
        ) from exc


@dataclass(frozen=True)
class NoiseInfo:
    mode: str
    pool_size: int | None
    pool_memory_mb: float
    whitened: bool


class GaussianNoiseSampler:
    def __init__(
        self, image_shape: tuple[int, int, int], device: torch.device, seed: int
    ) -> None:
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
        self.dtype = _parse_pool_dtype(dtype)
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
        pool = torch.empty(
            (self.pool_size, *self.image_shape), dtype=self.dtype, device="cpu"
        )
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
            # Remove trivial realized-pool mean/std bias per coordinate without
            # materializing a full float32 copy of potentially very large pools.
            self._whiten_pool_(pool)
        return pool.pin_memory() if torch.cuda.is_available() else pool

    def _whiten_pool_(self, pool: torch.Tensor) -> None:
        count = int(pool.shape[0])
        if count <= 0:
            raise ValueError("Cannot whiten an empty fixed noise pool")

        sums = torch.zeros(self.image_shape, dtype=torch.float64, device="cpu")
        sums_sq = torch.zeros_like(sums)

        for start in range(0, count, self.chunk_size):
            end = min(start + self.chunk_size, count)
            chunk = pool[start:end].to(dtype=torch.float32, copy=True)
            sums += chunk.sum(dim=0, dtype=torch.float64)
            chunk.square_()
            sums_sq += chunk.sum(dim=0, dtype=torch.float64)

        mean64 = sums / count
        variance64 = (sums_sq / count) - mean64.square()
        mean = mean64.to(dtype=torch.float32).unsqueeze(0)
        std = (
            variance64.clamp_min(0.0)
            .sqrt()
            .clamp_min(1e-6)
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )

        for start in range(0, count, self.chunk_size):
            end = min(start + self.chunk_size, count)
            chunk = pool[start:end].to(dtype=torch.float32)
            chunk.sub_(mean).div_(std)
            pool[start:end].copy_(chunk.to(dtype=pool.dtype))

    def _fingerprint_indices(self, sample_rows: int = 16) -> list[int]:
        sample_rows = max(1, min(int(sample_rows), self.pool_size))
        if sample_rows == 1:
            return [0]
        if sample_rows == self.pool_size:
            return list(range(self.pool_size))
        return [
            round(index * (self.pool_size - 1) / (sample_rows - 1))
            for index in range(sample_rows)
        ]

    def pool_fingerprint(self, sample_rows: int = 16) -> dict[str, Any]:
        """Return a compact deterministic fingerprint for the realized pool."""
        indices = self._fingerprint_indices(sample_rows)
        metadata: dict[str, Any] = {
            "fingerprint_version": 1,
            "image_shape": list(self.image_shape),
            "pool_size": self.pool_size,
            "pool_seed": self.pool_seed,
            "pool_dtype": str(self.pool.dtype).replace("torch.", ""),
            "pool_chunk_size": self.chunk_size,
            "whiten": self.whiten,
            "sample_indices": indices,
        }
        hasher = hashlib.sha256()
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        hasher.update(payload.encode("utf-8"))
        for index in indices:
            row = self.pool[int(index)].detach().cpu().contiguous()
            hasher.update(row.view(torch.uint8).numpy().tobytes())
        return {**metadata, "sha256": hasher.hexdigest()}

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
