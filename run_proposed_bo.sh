#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=4       # Number of CPUs per task
PARTITION="gpu_short" # Partition name
TIME="4:00:00"        # Maximum execution time

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p logs/

# Objectives and acquisitions to test
OBJECTIVES=("SinusoidalSynthetic" "BraninHoo" "Hartmann6")
ACQUISITIONS=("UCB" "POI" "EI")

# Params
ITER=100
SURROGATE="GP"  # Set surrogate to GP
EXPERIMENTAL_ID="E2"

# Create directories based on experimental ID
mkdir -p logs/${EXPERIMENTAL_ID}/train/

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/TPBO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(project_dir)s/logs/${EXPERIMENTAL_ID}"

# Overwrite config.ini file only if necessary
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Loop through each objective, acquisition, and seed value
for OBJECTIVE in "${OBJECTIVES[@]}"; do
    for ACQUISITION in "${ACQUISITIONS[@]}"; do
        for SEED in {0..4}; do
            # Set up experiment name and log file paths
            EXPERIMENT_NAME="vanilla_bo_${OBJECTIVE}_AGT_${ACQUISITION}_seed${SEED}"
            LOG_DIR="logs/${EXPERIMENTAL_ID}/train"

            # Run each experiment in parallel using sbatch
            sbatch --job-name="${EXPERIMENT_NAME}" \
                   --output="${LOG_DIR}/${EXPERIMENT_NAME}_%j.log" \
                   --cpus-per-task=$CPUS_PER_TASK \
                   --partition=$PARTITION \
                   --time=$TIME \
                   --wrap="python3 experiments/2024-09-21/proposed_bo.py --seed $SEED --objective $OBJECTIVE --acquisition $ACQUISITION --iterations $ITER"
        done
    done
done
