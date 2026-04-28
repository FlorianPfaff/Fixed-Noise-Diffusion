from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .noise import NoiseInfo


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json(payload: Any) -> str:
    return json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_git_dir(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        dot_git = candidate / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            text = dot_git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git_dir = Path(text.split(":", 1)[1].strip())
                return git_dir if git_dir.is_absolute() else candidate / git_dir
    return None


def _read_packed_ref(git_dir: Path, ref: str) -> str | None:
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return None
    for line in packed_refs.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", maxsplit=1)
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def git_metadata(cwd: Path | None = None) -> dict[str, str | None]:
    git_dir = _resolve_git_dir((cwd or Path.cwd()).resolve())
    if git_dir is None:
        return {"branch": None, "commit": None}

    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return {"branch": None, "commit": None}

    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return {"branch": None, "commit": head}

    ref = head.split(" ", maxsplit=1)[1]
    ref_path = git_dir / ref
    commit = None
    if ref_path.exists():
        commit = ref_path.read_text(encoding="utf-8").strip()
    else:
        commit = _read_packed_ref(git_dir, ref)
    branch = ref.removeprefix("refs/heads/")
    return {"branch": branch, "commit": commit}


def torch_metadata(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_name = None
    if device.type == "cuda" and cuda_available:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)

    return {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": device_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def noise_metadata(info: NoiseInfo) -> dict[str, Any]:
    return {
        "mode": info.mode,
        "pool_size": info.pool_size,
        "pool_memory_mb": round(info.pool_memory_mb, 3),
        "whitened": info.whitened,
    }


def build_run_metadata(
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    noise_info: NoiseInfo,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_name": config["run_name"],
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "config.yaml"),
        "config_hash": stable_hash(config),
        "command": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "git": git_metadata(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "torch": torch_metadata(device),
        "seed": int(config["seed"]),
        "data": dict(config["data"]),
        "noise": noise_metadata(noise_info),
    }


def build_run_summary(
    config: dict[str, Any],
    run_dir: Path,
    metadata: dict[str, Any],
    final_epoch: int,
    final_step: int,
    seconds: float,
    noise_info: NoiseInfo,
    last_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed",
        "completed_at": utc_now(),
        "run_name": config["run_name"],
        "run_dir": str(run_dir),
        "seed": int(config["seed"]),
        "config_hash": metadata["config_hash"],
        "git_commit": metadata["git"]["commit"],
        "final_epoch": int(final_epoch),
        "final_step": int(final_step),
        "seconds": round(float(seconds), 3),
        "noise": noise_metadata(noise_info),
        "last_eval": last_eval,
        "artifacts": {
            "config": str(run_dir / "config.yaml"),
            "metadata": str(run_dir / "run_metadata.json"),
            "metrics_jsonl": str(run_dir / "metrics.jsonl"),
            "metrics_csv": str(run_dir / "metrics.csv"),
            "samples": str(run_dir / "samples"),
            "checkpoints": str(run_dir / "checkpoints"),
        },
    }
