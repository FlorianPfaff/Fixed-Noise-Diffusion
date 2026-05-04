from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class RandomImageDataset(Dataset):
    def __init__(self, length: int, channels: int, image_size: int, seed: int) -> None:
        self.length = int(length)
        self.channels = int(channels)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(index))
        image = torch.rand(
            (self.channels, self.image_size, self.image_size),
            generator=generator,
            dtype=torch.float32,
        )
        return image.mul(2.0).sub(1.0), 0


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    val: DataLoader


def _subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: int(size)].tolist()
    return Subset(dataset, indices)


def _make_torchvision_cifar_loaders(
    dataset_name: str,
    data_cfg: dict[str, Any],
    seed: int,
) -> tuple[Dataset, Dataset]:
    from torchvision import datasets, transforms

    dataset_map = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    try:
        dataset_cls = dataset_map[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported CIFAR dataset {dataset_name!r}") from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    root = Path(data_cfg["data_dir"])
    train_dataset = dataset_cls(
        root=root,
        train=True,
        download=bool(data_cfg["download"]),
        transform=transform,
    )
    val_dataset = dataset_cls(
        root=root,
        train=False,
        download=bool(data_cfg["download"]),
        transform=transform,
    )
    train_dataset = _subset(train_dataset, data_cfg.get("subset_size"), seed)
    val_dataset = _subset(val_dataset, data_cfg.get("eval_subset_size"), seed + 1)
    return train_dataset, val_dataset


def make_dataloaders(config: dict[str, Any]) -> LoaderBundle:
    data_cfg = config["data"]
    seed = int(config["seed"])
    dataset_name = str(data_cfg["dataset"]).lower()

    if dataset_name == "fake":
        train_dataset = RandomImageDataset(
            data_cfg["fake_train_size"],
            data_cfg["channels"],
            data_cfg["image_size"],
            seed,
        )
        val_dataset = RandomImageDataset(
            data_cfg["fake_val_size"],
            data_cfg["channels"],
            data_cfg["image_size"],
            seed + 100_000,
        )
    elif dataset_name in {"cifar10", "cifar100"}:
        train_dataset, val_dataset = _make_torchvision_cifar_loaders(
            dataset_name,
            data_cfg,
            seed,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    return LoaderBundle(train=train_loader, val=val_loader)
