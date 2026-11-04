#!/bin/bash
#SBATCH --job-name=cpic-basin
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.err

# Basin-radius test (one source model per seed). shared QOS: a quarter node per GPU.
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/basin_radius.sbatch
# Summarize:   PYTHONPATH=src uv run python experiments/cross_session/basin_radius.py --summarize results/basin_radius

export UV_CACHE_DIR=$SCRATCH/uv-cache
cd /global/homes/i/imysore/CPIC_dev

env PYTHONPATH=src \
uv run python experiments/cross_session/basin_radius.py \
  --seed "${SLURM_ARRAY_TASK_ID}" --device cuda --out results/basin_radius
