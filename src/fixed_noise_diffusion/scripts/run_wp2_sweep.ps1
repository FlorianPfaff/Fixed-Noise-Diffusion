$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

py -3.12 -m fixed_noise_diffusion.train --config cifar10_gaussian.yaml
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_1k.yaml
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_10k.yaml
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_100k.yaml

py -3.12 -m fixed_noise_diffusion.plot_results --runs runs/cifar10_*
