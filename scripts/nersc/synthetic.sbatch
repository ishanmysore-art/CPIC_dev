#!/bin/bash
#SBATCH --job-name=cpic-synthetic
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%j.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%j.err

# Synthetic two-population sanity check for critic-anchored alignment, seeds 0-2.
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/synthetic.sbatch

export UV_CACHE_DIR=$SCRATCH/uv-cache
cd /global/homes/i/imysore/CPIC_dev

for seed in 0 1 2; do
  env PYTHONPATH=src \
  uv run python experiments/cross_session/synthetic_two_population.py --seed "$seed" --device cuda
done
