#!/bin/bash
#SBATCH --job-name=cpic-mlp-beta0
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%j.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%j.err

mkdir -p /global/homes/i/imysore/CPIC_dev/logs

export UV_CACHE_DIR=$SCRATCH/uv-cache

cd /global/homes/i/imysore/CPIC_dev/experiments/real_data_experiments

env PYTHONPATH=../../src \
uv run python real_data_experiment_standard.py \
  --config m1_stochastic_infonce \
  --ydim 64 \
  --hidden-dim 256 \
  --n-layers 0
