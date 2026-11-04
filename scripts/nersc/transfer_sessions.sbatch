#!/bin/bash
#SBATCH --job-name=cpic-transfer
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.err

# Gate 3 electrode-space transfer, indy_20160627_01 -> {indy_20160624_03, indy_20160630_01},
# 3 seeds each (tasks 0-2: first target, 3-5: second).
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/transfer_sessions.sbatch
# Summarize:   PYTHONPATH=src uv run python experiments/cross_session/transfer_sessions.py --summarize results/transfer_sessions

export UV_CACHE_DIR=$SCRATCH/uv-cache
cd /global/homes/i/imysore/CPIC_dev

DATA=$SCRATCH/cpic-data/real_data/M1
TARGETS=(indy_20160624_03 indy_20160630_01)
id=${SLURM_ARRAY_TASK_ID}
target=${TARGETS[$(( id / 3 ))]}
seed=$(( id % 3 ))

env PYTHONPATH=src \
uv run python experiments/cross_session/transfer_sessions.py \
  --source "$DATA/indy_20160627_01.mat" --target "$DATA/$target.mat" \
  --seed "$seed" --device cuda --out results/transfer_sessions
