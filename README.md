# Fixed-Noise Diffusion Starter

Minimal image DDPM experiment stack for the WP2 fixed-noise reproduction:
fresh Gaussian noise versus reusable Gaussian template pools.

The original controlled experiments use CIFAR-10. For a non-CIFAR validation
dataset, use STL-10 with the `train+unlabeled` split resized to 32x32. This keeps
the fixed-noise pool memory footprint comparable to CIFAR-10 while moving the
validation away from the Tiny Images lineage.

## Environment

Use Python 3.12. On this machine, PyTorch with CUDA is already available for
`py -3.12`. For a fresh environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python -m pip install -e .
```

If you use the existing Python 3.12 installation without installing the package,
set `PYTHONPATH=src` before running modules.

## Smoke Test

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.train --config smoke.yaml
```

The smoke run uses synthetic images, one train step, one denoising validation
pass, and one sample grid.

## WP2 First CIFAR-10 Sweep

Run the Gaussian baseline and fixed-pool configs:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.train --config cifar10_gaussian.yaml
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_1k.yaml
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_10k.yaml
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_100k.yaml
```

The same sweep commands are available in:

```powershell
.\src\fixed_noise_diffusion\scripts\run_wp2_sweep.ps1
```

Each run writes:

- `metrics.jsonl`: all train/eval events
- `metrics.csv`: compact spreadsheet-friendly train/eval rows
- `run_metadata.json`: config hash, Git, Python, Torch, device, seed, and noise info
- `run_summary.json`: final step, final diagnostic values, and artifact paths
- `samples/`: checkpoint sample grids
- `checkpoints/`: model checkpoints and run config

Plot the denoising-gap curves:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.plot_results --runs runs/cifar10_*
```

## STL-10 Validation

STL-10 is intended as a targeted validation, not as a full replacement for the
CIFAR-10 pool-size sweep. The recommended paper-facing check is:

- fresh Gaussian baseline,
- fixed pool with `M=1k`,
- fixed pool with `M=10k`,
- fixed pool with `M=100k`,
- seeds `0,1,2`,
- base64 model, cosine schedule, 100 epochs,
- STL-10 `train+unlabeled` split for training and `test` split for validation,
  resized to 32x32.

The base config is available as:

```powershell
py -3.12 -m fixed_noise_diffusion.train --config stl10_base.yaml
```

For GPU servers registered as self-hosted GitHub runners, use the manual
workflow:

```text
.github/workflows/wp2-stl10-validation.yml
```

The workflow runs the 12 validation jobs above and uploads only compact artifacts
by default: metrics, config, run metadata, and run summary. It does not upload
datasets, generated sample directories, or checkpoints. It uses a persistent
dataset cache on the self-hosted runner and a file lock around the initial
STL-10 download so the matrix jobs do not repeatedly download the dataset.

## Sample-Quality Evaluation

Evaluate saved checkpoints with Inception FID/KID:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.evaluate_sample_quality `
  --sweep-dir runs/wp2_50ep_3seed `
  --output-dir runs/wp2_fid2048_10k `
  --epochs 50 `
  --sample-count 10000 `
  --fid-feature 2048
```

For a larger FID run, use the training split for real statistics:

```powershell
py -3.12 -m fixed_noise_diffusion.evaluate_sample_quality `
  --sweep-dir runs/wp2_100ep_reduced `
  --output-dir runs/wp2_fid2048_50k `
  --epochs 100 `
  --sample-count 50000 `
  --real-count 50000 `
  --real-split train `
  --fid-feature 2048
```

Combine one or more `sample_quality.csv` outputs and optionally join denoising
gap summaries:

```powershell
py -3.12 -m fixed_noise_diffusion.summarize_sample_quality `
  --quality runs/wp2_fid2048_10k_gpu0 `
  --quality runs/wp2_fid2048_10k_gpu1 `
  --gap-summary runs/wp2_50ep_gap_summary.csv `
  --output-dir runs `
  --prefix wp2_fid2048_10k_epoch50
```

Evaluate timestep-local denoising gaps from saved checkpoints:

```powershell
py -3.12 -m fixed_noise_diffusion.evaluate_timestep_diagnostics `
  --sweep-dir runs/wp2_100ep_reduced `
  --output-dir runs/wp2_timestep_diagnostics `
  --epochs 50,100 `
  --timesteps 0,25,50,100,200,400,600,800,999 `
  --batches 16
```

## Key Diagnostic

The main WP2 diagnostic is:

```text
denoising_gap = held_out_gaussian_denoising_loss - training_law_denoising_loss
```

For fixed pools, a positive and growing gap means the model is fitting the
realized reusable noise law better than held-out fresh Gaussian noise. That is
the first signal of support-limited overspecialization.
