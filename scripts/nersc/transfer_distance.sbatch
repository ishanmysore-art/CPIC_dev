#!/bin/bash
#SBATCH --job-name=cpic-distance
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --array=0-17
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.err

# Distance curve: indy_20160627_01 -> 6 more Indy sessions, 3 seeds each (task = 3 * target + seed).
# Same arms as transfer_sessions.sbatch; results land in the same folder so --summarize sees all targets.
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/transfer_distance.sbatch

export UV_CACHE_DIR=$SCRATCH/uv-cache
cd /global/homes/i/imysore/CPIC_dev

DATA=$SCRATCH/cpic-data/real_data/M1
TARGETS=(indy_20160622_01 indy_20160419_01 indy_20160915_01 indy_20161006_02 indy_20161212_02 indy_20170131_02)
id=${SLURM_ARRAY_TASK_ID}
target=${TARGETS[$(( id / 3 ))]}
seed=$(( id % 3 ))

env PYTHONPATH=src \
uv run python experiments/cross_session/transfer_sessions.py \
  --source "$DATA/indy_20160627_01.mat" --target "$DATA/$target.mat" \
  --seed "$seed" --device cuda --out results/transfer_sessions
