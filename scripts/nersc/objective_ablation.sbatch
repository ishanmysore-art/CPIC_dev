#!/bin/bash
#SBATCH --job-name=cpic-ablation
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --array=0-2
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.err

# Compression-weight ablation (beta = 0, 1e-4, 1e-3), one seed per task.
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/objective_ablation.sbatch
# Summarize:   PYTHONPATH=src uv run python experiments/cross_session/objective_ablation.py --summarize results/objective_ablation

export UV_CACHE_DIR=$SCRATCH/uv-cache
cd /global/homes/i/imysore/CPIC_dev

env PYTHONPATH=src \
uv run python experiments/cross_session/objective_ablation.py \
  --seed "${SLURM_ARRAY_TASK_ID}" --device cuda --out results/objective_ablation
