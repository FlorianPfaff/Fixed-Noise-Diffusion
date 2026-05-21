from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .sweep import select_run_dirs
from .utils import format_float, sample_mean, sample_sem, sample_std, write_csv_rows


PROTOCOL_COLUMNS = [
    "beta_schedule",
    "num_timesteps",
    "image_size",
    "channels",
    "base_channels",
    "channel_mults",
    "time_emb_dim",
]


SUMMARY_COLUMNS = [
    "dataset",
    "experiment",
    "family",
    "noise_mode",
    "condition",
    "pool_size",
    *PROTOCOL_COLUMNS,
    "epoch",
    "n",
    "denoising_gap_mean",
    "denoising_gap_std",
    "denoising_gap_sem",
]

COMBINED_COLUMNS = [
    "run_name",
    "dataset",
    "experiment",
    "family",
    "noise_mode",
    "condition",
    "pool_size",
    *PROTOCOL_COLUMNS,
    "seed",
    "epoch",
    "step",
    "train_den_loss",
    "gaussian_den_loss",
    "denoising_gap",
    "source_run_dir",
]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_eval_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("type") == "eval"]


def _maybe_int(value: Any) -> int | None:
    if value in (None, "", "None", "null"):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return int(float(value))


def _dataset_label(config: dict[str, Any]) -> str:
    dataset = str(config.get("data", {}).get("dataset", "")).lower()
    if dataset in {"cifar10", "cifar-10"}:
        return "cifar10"
    if dataset in {"stl10", "stl-10"}:
        return "stl10"
    if dataset in {"celeba", "celeba64", "celeba-64"}:
        return "celeba64"
    return dataset or "unknown"


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _channel_mults_label(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _protocol_info(config: dict[str, Any]) -> dict[str, str]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    diffusion_cfg = config.get("diffusion", {}) if isinstance(config.get("diffusion", {}), dict) else {}
    return {
        "beta_schedule": _string_value(diffusion_cfg.get("beta_schedule")),
        "num_timesteps": _string_value(diffusion_cfg.get("num_timesteps")),
        "image_size": _string_value(data_cfg.get("image_size")),
        "channels": _string_value(data_cfg.get("channels")),
        "base_channels": _string_value(model_cfg.get("base_channels")),
        "channel_mults": _channel_mults_label(model_cfg.get("channel_mults")),
        "time_emb_dim": _string_value(model_cfg.get("time_emb_dim")),
    }


def _pool_label(pool_size: int) -> str:
    if pool_size >= 1000 and pool_size % 1000 == 0:
        return f"{pool_size // 1000}k"
    return str(pool_size)


def _noise_info(config: dict[str, Any], row: dict[str, str]) -> tuple[str, int | None, bool]:
    noise_cfg = config.get("noise", {})
    noise_mode = row.get("noise_mode") or str(noise_cfg.get("mode", ""))
    pool_size = _maybe_int(row.get("pool_size"))
    if pool_size is None:
        pool_size = _maybe_int(noise_cfg.get("pool_size"))
    whitened = noise_mode == "fixed_pool_whitened" or bool(noise_cfg.get("whiten", False))
    return noise_mode, pool_size, whitened


def _condition_family(noise_mode: str, pool_size: int | None, whitened: bool) -> tuple[str, str, str]:
    if noise_mode == "gaussian" or pool_size is None:
        return "standard", "gaussian", "gaussian"
    label = _pool_label(pool_size)
    if whitened:
        return "whitened", "whitened fixed pool", f"fixed_pool_whitened_{label}"
    return "standard", "fixed pool", f"fixed_pool_{label}"


def _run_seed(config: dict[str, Any], run_dir: Path) -> str:
    seed = config.get("seed")
    if seed is not None:
        return str(seed)
    marker = "_seed"
    if marker in run_dir.name:
        return run_dir.name.rsplit(marker, 1)[-1]
    return ""


def _pool_sort_value(row: dict[str, str]) -> int:
    return int(row["pool_size"]) if row["pool_size"] else 10**18


def _protocol_sort_value(item: dict[str, str]) -> tuple[str, ...]:
    return tuple(item.get(column, "") for column in PROTOCOL_COLUMNS)


def _combined_sort_key(item: dict[str, str]) -> tuple[Any, ...]:
    return (
        item["dataset"],
        item["experiment"],
        item["family"],
        _pool_sort_value(item),
        item["condition"],
        _protocol_sort_value(item),
        int(item["epoch"]),
        int(item["seed"] or -1),
    )


def _summary_sort_key(item: dict[str, str]) -> tuple[Any, ...]:
    return (
        item["dataset"],
        item["experiment"],
        item["family"],
        _pool_sort_value(item),
        item["condition"],
        _protocol_sort_value(item),
        int(item["epoch"]),
    )


def collect_rows(run_dirs: list[Path], epochs: set[int] | None) -> list[dict[str, str]]:
    combined: list[dict[str, str]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        config = _read_yaml(config_path)
        dataset = _dataset_label(config)
        seed = _run_seed(config, run_dir)
        protocol = _protocol_info(config)

        for row in _read_eval_rows(run_dir):
            epoch = int(row["epoch"])
            if epochs is not None and epoch not in epochs:
                continue
            noise_mode, pool_size, whitened = _noise_info(config, row)
            experiment, family, condition = _condition_family(noise_mode, pool_size, whitened)
            combined.append(
                {
                    "run_name": run_dir.name,
                    "dataset": dataset,
                    "experiment": experiment,
                    "family": family,
                    "noise_mode": noise_mode,
                    "condition": condition,
                    "pool_size": "" if pool_size is None else str(pool_size),
                    **protocol,
                    "seed": seed,
                    "epoch": str(epoch),
                    "step": row.get("step", ""),
                    "train_den_loss": row.get("train_den_loss", ""),
                    "gaussian_den_loss": row.get("gaussian_den_loss", ""),
                    "denoising_gap": row.get("denoising_gap", ""),
                    "source_run_dir": str(run_dir),
                }
            )
    return sorted(combined, key=_combined_sort_key)


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["experiment"],
                row["family"],
                row["noise_mode"],
                row["condition"],
                row["pool_size"],
                *(row.get(column, "") for column in PROTOCOL_COLUMNS),
                row["epoch"],
            )
        ].append(row)

    summary: list[dict[str, str]] = []
    for key, group in grouped.items():
        gaps = [
            float(row["denoising_gap"])
            for row in group
            if row.get("denoising_gap") not in (None, "")
        ]
        dataset, experiment, family, noise_mode, condition, pool_size, *protocol_values, epoch = key
        summary.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "family": family,
                "noise_mode": noise_mode,
                "condition": condition,
                "pool_size": pool_size,
                **dict(zip(PROTOCOL_COLUMNS, protocol_values, strict=True)),
                "epoch": epoch,
                "n": str(len(gaps)),
                "denoising_gap_mean": format_float(sample_mean(gaps), precision=16),
                "denoising_gap_std": format_float(sample_std(gaps), precision=16),
                "denoising_gap_sem": format_float(sample_sem(gaps), precision=16),
            }
        )

    return sorted(summary, key=_summary_sort_key)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    write_csv_rows(path, rows, fieldnames)


def parse_epochs(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize denoising-gap eval rows across WP2 runs.")
    parser.add_argument("--sweep-dir", action="append", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="Optional run directory name inside each sweep dir")
    parser.add_argument("--epochs", default="all", help="Comma-separated epochs to include, or 'all'.")
    parser.add_argument("--output", type=Path, required=True, help="Summary CSV path")
    parser.add_argument("--combined-output", type=Path, default=None, help="Optional per-run eval-row CSV path")
    args = parser.parse_args()

    run_dirs = select_run_dirs(args.sweep_dir, args.run)
    rows = collect_rows(run_dirs, parse_epochs(args.epochs))
    summary = summarize_rows(rows)
    write_csv(args.output.expanduser(), summary, SUMMARY_COLUMNS)
    if args.combined_output is not None:
        write_csv(args.combined_output.expanduser(), rows, COMBINED_COLUMNS)
    print(f"runs={len(run_dirs)}")
    print(f"rows={len(rows)}")
    print(f"summary_rows={len(summary)}")
    print(f"summary={args.output.expanduser()}")


if __name__ == "__main__":
    main()
