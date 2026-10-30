#!/bin/bash
#SBATCH --job-name=cpic-ladder
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.err

# Experiment A attribution ladder: one array task per seed, all rungs sequentially.
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/ladder.sbatch
# Summarize after all tasks finish:
#   PYTHONPATH=src uv run python experiments/cross_session/ladder.py --summarize results/ladder

export UV_CACHE_DIR=$SCRATCH/uv-cache

cd /global/homes/i/imysore/CPIC_dev

env PYTHONPATH=src \
uv run python experiments/cross_session/ladder.py \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --out results/ladder
