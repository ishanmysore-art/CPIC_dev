#!/bin/bash
#SBATCH --job-name=cpic-legacy-repro
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --account=m2043_g
#SBATCH --output=/global/homes/i/imysore/CPIC_dev/logs/%x-%j.out
#SBATCH --error=/global/homes/i/imysore/CPIC_dev/logs/%x-%j.err

# J2: rerun the unmodified legacy benchmark (expected I_pred 2.1814, R² .3108/.3228/.2737)
# so the ladder's L0_legacy can be checked against it for seed 0.
# The legacy script takes its device from the config file; print it so a CPU run is obvious.

export UV_CACHE_DIR=$SCRATCH/uv-cache

cd /global/homes/i/imysore/CPIC_dev/experiments/real_data_experiments

grep -n '^device' config/config_m1_scaling_benchmark.ini

env PYTHONPATH=../../src \
uv run python real_data_experiment_standard.py \
  --config m1_scaling_benchmark \
  --fixed-epochs \
  --seed 0
