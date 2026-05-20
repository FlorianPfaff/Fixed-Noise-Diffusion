import pytest
import torch

from fixed_noise_diffusion.utils import (
    generator_for,
    make_run_dir,
    normalize_seed,
    resolve_device,
    seed_everything,
)


def test_normalize_seed_wraps_negative_values():
    assert normalize_seed(-1) == 2**32 - 1
    seed_everything(-1)


def test_resolve_device_falls_back_for_indexed_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("cuda:0") == torch.device("cpu")


def test_resolve_device_preserves_indexed_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_generator_for_preserves_cuda_device_index(monkeypatch):
    calls = []

    class DummyGenerator:
        def __init__(self) -> None:
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    def fake_generator(*, device):
        calls.append(device)
        return DummyGenerator()

    monkeypatch.setattr(torch, "Generator", fake_generator)

    generator = generator_for(torch.device("cuda:1"), 123)

    assert calls == ["cuda:1"]
    assert generator.seed == 123


def test_make_run_dir_refuses_existing_nonempty_run(tmp_path):
    run_dir = make_run_dir(tmp_path, "run")
    stale_metrics = run_dir / "metrics.csv"
    stale_metrics.write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to append"):
        make_run_dir(tmp_path, "run")

    assert stale_metrics.read_text(encoding="utf-8") == "stale\n"


def test_make_run_dir_overwrite_replaces_existing_run(tmp_path):
    run_dir = make_run_dir(tmp_path, "run")
    (run_dir / "metrics.csv").write_text("stale\n", encoding="utf-8")

    replacement = make_run_dir(tmp_path, "run", overwrite=True)

    assert replacement == run_dir
    assert not (replacement / "metrics.csv").exists()
    assert (replacement / "checkpoints").is_dir()
    assert (replacement / "samples").is_dir()


def test_resolve_device_falls_back_for_indexed_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("cuda:0").type == "cpu"
