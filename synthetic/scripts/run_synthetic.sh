#!/bin/bash

cd /home/aanant/Data/CPIC_dev/synthetic

export PYTHONPATH="/home/aanant/Data/CPIC_dev/src:$PYTHONPATH"

LOG_FILE="experiment_log_$(date +%Y%m%d_%H%M%S).txt"

echo "starting experiments: $(date)" | tee -a $LOG_FILE

CONFIGS=(
        "lorenz_stochastic_infonce_exploration_conv"
       )

for config in "${CONFIGS[@]}"; do
    echo "doing config: $config" | tee -a $LOG_FILE
    for i in {0..1}; do
        echo "  seed $i/99" | tee -a $LOG_FILE
        python synthetic_experiment.py --config $config --seed $i >> $LOG_FILE 2>&1
        if [ $? -ne 0 ]; then
            echo "  seed $i failed!!!" | tee -a $LOG_FILE
        fi
    done
    echo "completed $config at $(date) !" | tee -a $LOG_FILE
done

echo "all experiments completed at $(date) !!!" | tee -a $LOG_FILE