#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=4       # Number of CPUs per task
PARTITION="gpu_short" # Partition name
TIME="4:00:00"        # Maximum execution time

# Create results directory if it doesn't exist
mkdir -p results/
mkdir -p logs/

# Objectives and acquisitions to test
OBJECTIVES=("SinusoidaSynthetic" "BraninHoo" "Hartmann6")
ACQUISITIONS=("UCB" "POI" "EI")
SURROGATES=("GP" "TP") # Add GP and TP for different surrogate models

# Params
SEED = 0
ITER = 500

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/TPBO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(project_dir)s/logs"

# Overwrite config.ini file
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Loop through each objective, acquisition, and surrogate model
for OBJECTIVE in "${OBJECTIVES[@]}"; do
    for ACQUISITION in "${ACQUISITIONS[@]}"; do
        for SURROGATE in "${SURROGATES[@]}"; do
            # Run each experiment in parallel using sbatch
            sbatch --job-name="${OBJECTIVE}_${ACQUISITION}_${SURROGATE}" \
                   --output="logs/${OBJECTIVE}_${ACQUISITION}_${SURROGATE}_%j.log" \
                   --cpus-per-task=$CPUS_PER_TASK \
                   --partition=$PARTITION \
                   --time=$TIME \
                   --wrap="python3 experiments/2024-09-20/vanilla_bo.py --seed $SEED --objective $OBJECTIVE --acquisition $ACQUISITION --surrogate $SURROGATE --iterations $ITER"
        done
    done
done
