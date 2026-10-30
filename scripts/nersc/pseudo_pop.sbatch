#!/bin/bash
#SBATCH --job-name=cpic-pseudo-pop
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --array=0-20
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%A_%a.err

# Gate 2 (I3 + I1/I2), corrected pipeline.
#   tasks 0-8   overlap 0   (disjoint halves): 3 partitions x 3 seeds
#   tasks 9-17  overlap 0.5                  : 3 partitions x 3 seeds
#   tasks 18-20 overlap 1   (full population): 3 seeds, plus identifiability (I1/I2)
# Submit with: mkdir -p ~/CPIC_dev/logs && sbatch scripts/nersc/pseudo_pop.sbatch
# Summarize:   PYTHONPATH=src uv run python experiments/cross_session/pseudo_pop.py --summarize results/pseudo_pop

export UV_CACHE_DIR=$SCRATCH/uv-cache
cd /global/homes/i/imysore/CPIC_dev

id=${SLURM_ARRAY_TASK_ID}
extra=""
if [ "$id" -lt 9 ]; then
  overlap=0; partition=$(( id / 3 )); seed=$(( id % 3 ))
elif [ "$id" -lt 18 ]; then
  overlap=0.5; partition=$(( (id - 9) / 3 )); seed=$(( (id - 9) % 3 ))
else
  overlap=1; partition=0; seed=$(( id - 18 )); extra="--identifiability"
fi

env PYTHONPATH=src \
uv run python experiments/cross_session/pseudo_pop.py \
  --overlap "$overlap" --partition-seed "$partition" --seed "$seed" \
  --pipeline corrected --device cuda --out results/pseudo_pop $extra
