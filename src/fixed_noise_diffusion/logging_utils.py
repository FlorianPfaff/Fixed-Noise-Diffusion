from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "type",
    "epoch",
    "step",
    "split",
    "loss",
    "train_noise_loss",
    "gaussian_den_loss",
    "train_den_loss",
    "heldout_pool_den_loss",
    "denoising_gap",
    "denoising_eval_timestep_seed",
    "heldout_pool_gap",
    "gaussian_minus_heldout_gap",
    "fid",
    "kid_mean",
    "kid_std",
    "lr",
    "seconds",
    "noise_mode",
    "pool_size",
    "pool_memory_mb",
    "heldout_pool_seed",
    "config_hash",
    "git_commit",
    "samples_path",
]


class MetricLogger:
    def __init__(self, run_dir: Path, *, append: bool = False) -> None:
        self.run_dir = run_dir
        self.jsonl_path = run_dir / "metrics.jsonl"
        self.csv_path = run_dir / "metrics.csv"

        if append:
            if self.csv_path.exists():
                self._validate_existing_csv_header()
            else:
                self._write_csv_header()
            if not self.jsonl_path.exists():
                self.jsonl_path.write_text("", encoding="utf-8")
            return

        existing = [
            path.name
            for path in (self.csv_path, self.jsonl_path)
            if path.exists()
        ]
        if existing:
            names = ", ".join(existing)
            raise FileExistsError(
                "Metric files already exist in "
                f"{run_dir}: {names}. Refusing to append fresh-run metrics "
                "to existing artifacts because this can contaminate denoising "
                "gap summaries and paper tables. Use a new run_name/output_dir, "
                "set overwrite_run=true or pass --overwrite-run to delete the "
                "old run directory, or construct MetricLogger(..., append=True) "
                "only for an explicit resume workflow."
            )

        self._write_csv_header()
        self.jsonl_path.write_text("", encoding="utf-8")

    def _write_csv_header(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()

    def _validate_existing_csv_header(self) -> None:
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
        if header != CSV_FIELDS:
            raise ValueError(
                f"Existing metrics.csv in {self.run_dir} has an incompatible "
                "header. Refusing append mode because mixed metric schemas can "
                "silently corrupt downstream summaries."
            )

    def log(self, record: dict[str, Any]) -> None:
        clean = {
            key: (float(value) if hasattr(value, "item") else value)
            for key, value in record.items()
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, sort_keys=True) + "\n")
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writerow(clean)
