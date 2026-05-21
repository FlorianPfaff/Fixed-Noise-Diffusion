from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .plotting import save_figure
from .summarize_sample_quality import (
    QUALITY_PROTOCOL_COLUMNS,
    condition_kind,
    condition_pool_size,
    normalize_dataset_label,
    write_csv,
)
from .utils import float_or_nan, format_float

METRIC_COLUMNS = [
    "fid_mean",
    "denoising_gap_mean",
    "low_mid_mean_timestep_gap",
]

PROTOCOL_COLUMNS = (
    *QUALITY_PROTOCOL_COLUMNS,
    "beta_schedule",
    "num_timesteps",
    "image_size",
    "channels",
    "base_channels",
    "channel_mults",
    "time_emb_dim",
)


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Input spec must be LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Input label is empty in {spec!r}")
    return label, Path(raw_path).expanduser()


def infer_schedule(label: str, condition: str) -> str:
    text = f"{label}_{condition}".lower()
    if "cosine" in text:
        return "cosine"
    if "linear" in text:
        return "linear"
    return "unknown"


def infer_model(label: str, condition: str) -> str:
    text = f"{label}_{condition}".lower()
    if "strong96" in text:
        return "strong96"
    return "base64"


def _pool_size_from_row(row: dict[str, str]) -> int | None:
    for key in ("pool_size", "pool_size_sort"):
        raw_value = row.get(key, "")
        if raw_value in ("", "inf", "None", None):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return int(value)
    return condition_pool_size(row["condition"])


def _kind_from_row(row: dict[str, str], condition: str) -> str:
    explicit = row.get("kind", "")
    if explicit:
        return explicit
    noise_mode = str(row.get("noise_mode", ""))
    if noise_mode == "gaussian":
        return "gaussian"
    if noise_mode == "fixed_pool_whitened":
        return "whitened"
    if noise_mode == "fixed_pool":
        return "fixed_pool"
    return condition_kind(condition)


def _protocol_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple("" if row.get(column) is None else str(row.get(column, "")) for column in PROTOCOL_COLUMNS)


def normalize_summary_row(
    row: dict[str, str], label: str, source_path: Path
) -> dict[str, str]:
    condition = row["condition"]
    pool_size = _pool_size_from_row(row)
    normalized = {
        "series": label,
        "dataset": normalize_dataset_label(row.get("dataset")),
        "schedule": infer_schedule(label, condition),
        "model": infer_model(label, condition),
        "condition": condition,
        "kind": _kind_from_row(row, condition),
        "pool_size": "" if pool_size is None else str(pool_size),
        "epoch": row.get("epoch", ""),
        "n": row.get("n", ""),
        "source": str(source_path),
    }
    for column in METRIC_COLUMNS:
        value = row.get(column, "")
        if column == "low_mid_mean_timestep_gap" and value == "":
            value = row.get("mean_timestep_gap", "")
        normalized[column] = format_float(float_or_nan(value))
    for column in ("fid_std", "denoising_gap_std"):
        normalized[column] = format_float(float_or_nan(row.get(column, "")))
    for column in PROTOCOL_COLUMNS:
        normalized[column] = "" if row.get(column) is None else str(row.get(column, ""))
    return normalized


def read_phase_rows(input_specs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in input_specs:
        label, path = parse_input_spec(spec)
        resolved = path.resolve()
        with resolved.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(normalize_summary_row(row, label, resolved))
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["schedule"],
            row["dataset"],
            row["series"],
            _protocol_key(row),
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
        ),
    )


def _metric_value(row: dict[str, str], column: str) -> float:
    return float_or_nan(row.get(column, ""))


def _metric_std(row: dict[str, str], column: str) -> float:
    std_column = {
        "fid_mean": "fid_std",
        "denoising_gap_mean": "denoising_gap_std",
    }.get(column, "")
    return float_or_nan(row.get(std_column, "")) if std_column else math.nan


def _pool_value(row: dict[str, str]) -> int | None:
    return int(row["pool_size"]) if row.get("pool_size") else None


def _is_fixed_pool_row(row: dict[str, str]) -> bool:
    return row.get("kind") in {"fixed_pool", "whitened"} and _pool_value(row) is not None


def _is_gaussian_row(row: dict[str, str]) -> bool:
    return row.get("kind") == "gaussian"


def _series_groups(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        dataset = normalize_dataset_label(row.get("dataset"))
        groups.setdefault((dataset, row["series"]), []).append(row)
    return groups


def _series_protocol_groups(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str, tuple[str, ...]], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str, tuple[str, ...]], list[dict[str, str]]] = {}
    for row in rows:
        dataset = normalize_dataset_label(row.get("dataset"))
        groups.setdefault(
            (
                dataset,
                row.get("model", ""),
                row.get("schedule", ""),
                row["series"],
                _protocol_key(row),
            ),
            [],
        ).append(row)
    return groups


def _series_legend_label(dataset: str, series: str) -> str:
    return f"{dataset} / {series}" if dataset else series


def _series_protocol_legend_label(
    dataset: str,
    model: str,
    schedule: str,
    series: str,
    protocol_key: tuple[str, ...],
) -> str:
    label = _series_legend_label(dataset, series)
    extras = [value for value in (model, schedule) if value and value not in label]
    if extras:
        label = f"{label} [{'/'.join(extras)}]"
    protocol = {
        column: value
        for column, value in zip(PROTOCOL_COLUMNS, protocol_key, strict=True)
        if value
    }
    if not protocol:
        return label
    selected = (
        "sample_count",
        "requested_real_count",
        "real_split",
        "sample_steps",
        "sampler",
        "fid_feature",
        "kid_subset_size",
    )
    protocol_label = ", ".join(
        f"{column}={protocol[column]}" for column in selected if column in protocol
    )
    return label if not protocol_label else f"{label} ({protocol_label})"


def _row_epoch(row: dict[str, str]) -> int | None:
    raw = row.get("epoch", "")
    if raw in ("", None):
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def select_plot_rows(
    rows: list[dict[str, str]], epoch: int | None = None
) -> list[dict[str, str]]:
    if epoch is not None:
        return [row for row in rows if _row_epoch(row) == int(epoch)]

    latest_by_condition: dict[
        tuple[str, str, str, str, str, str, str, tuple[str, ...]],
        tuple[int, dict[str, str]],
    ] = {}
    rows_without_epoch: list[dict[str, str]] = []
    for row in rows:
        row_epoch = _row_epoch(row)
        if row_epoch is None:
            rows_without_epoch.append(row)
            continue
        key = (
            row.get("series", ""),
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("schedule", ""),
            row.get("kind", ""),
            row.get("condition", ""),
            row.get("pool_size", ""),
            _protocol_key(row),
        )
        previous = latest_by_condition.get(key)
        if previous is None or row_epoch > previous[0]:
            latest_by_condition[key] = (row_epoch, row)

    selected = [item[1] for item in latest_by_condition.values()]
    selected.extend(rows_without_epoch)
    return sorted(
        selected,
        key=lambda row: (
            row.get("dataset", ""),
            row["model"],
            row["schedule"],
            row["series"],
            _protocol_key(row),
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
        ),
    )


def parse_plot_epoch(raw: str) -> int | None:
    normalized = str(raw).strip().lower()
    if normalized in {"", "final", "latest"}:
        return None
    return int(normalized)


def plot_phase_diagram(
    rows: list[dict[str, str]], output: Path, *, epoch: int | None = None
) -> None:
    rows = select_plot_rows(rows, epoch)
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    metric_titles = [
        ("fid_mean", "FID-2048"),
        ("denoising_gap_mean", "Final denoising gap"),
        ("low_mid_mean_timestep_gap", "Low/mid timestep gap"),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, (metric, title) in zip(axes, metric_titles):
        for index, ((dataset, model, schedule, series, protocol_key), group) in enumerate(
            sorted(_series_protocol_groups(rows).items())
        ):
            color = colors[index % len(colors)]
            fixed = [row for row in group if _is_fixed_pool_row(row)]
            fixed = sorted(fixed, key=lambda row: int(row["pool_size"]))
            if fixed:
                x_values = [_pool_value(row) for row in fixed]
                y_values = [_metric_value(row, metric) for row in fixed]
                y_errors = [_metric_std(row, metric) for row in fixed]
                has_errors = any(not math.isnan(value) for value in y_errors)
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors if has_errors else None,
                    marker="o",
                    capsize=3 if has_errors else 0,
                    label=_series_protocol_legend_label(dataset, model, schedule, series, protocol_key),
                    color=color,
                )
            gaussian = [
                _metric_value(row, metric) for row in group if _is_gaussian_row(row)
            ]
            if gaussian:
                axis.axhline(
                    gaussian[0],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("Pool size M")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine and plot WP2 fixed-pool phase-diagram summaries."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Summary CSV as LABEL=PATH. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="wp2_phase_diagram")
    parser.add_argument(
        "--plot-epoch",
        default="final",
        help="Epoch to plot, or 'final'/'latest' for the latest row per series and condition.",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.input:
        raise ValueError("At least one --input LABEL=PATH is required")

    output_dir = args.output_dir.expanduser()
    rows = read_phase_rows(args.input)
    write_csv(output_dir / f"{args.prefix}_combined.csv", rows)
    if not args.no_plot:
        plot_phase_diagram(
            rows,
            output_dir / f"{args.prefix}.png",
            epoch=parse_plot_epoch(args.plot_epoch),
        )


if __name__ == "__main__":
    main()
