from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_eval_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("type") == "eval"]


def plot_runs(run_dirs: list[Path], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plotted = False
    for run_dir in run_dirs:
        rows = _read_eval_rows(run_dir)
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        gaps = [float(row["denoising_gap"]) for row in rows]
        gaussian_losses = [float(row["gaussian_den_loss"]) for row in rows]
        train_losses = [float(row["train_den_loss"]) for row in rows]
        label = run_dir.name
        axes[0].plot(epochs, gaps, marker="o", label=label)
        axes[1].plot(epochs, gaussian_losses, marker="o", label=f"{label} Gaussian")
        axes[1].plot(epochs, train_losses, marker="x", linestyle="--", label=f"{label} train law")
        plotted = True

    if not plotted:
        raise SystemExit("No eval rows found in the provided run directories.")

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Denoising Generalization Gap")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L_gauss_den - L_train_den")
    axes[1].set_title("Denoising Losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WP2 denoising diagnostics.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Run directories.")
    parser.add_argument("--output", type=Path, default=Path("runs/wp2_denoising_gap.png"))
    args = parser.parse_args()
    plot_runs(args.runs, args.output)


if __name__ == "__main__":
    main()

