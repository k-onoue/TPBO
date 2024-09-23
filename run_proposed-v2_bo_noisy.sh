#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=4       # Number of CPUs per task
PARTITION="cluster_short" # Partition name
TIME="4:00:00"        # Maximum execution time

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p logs/

# Objectives and acquisitions to test
OBJECTIVES=("SinusoidalSynthetic" "BraninHoo" "Hartmann6")
ACQUISITIONS=("UCB" "POI" "EI")
# ACQUISITIONS=("EI")

# Noise strength
NOISE=(0 1 2)

# Params
ITER=50
EXPERIMENTAL_ID="E4_additional"

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

# Loop through each objective, acquisition, seed value, and noise value
for OBJECTIVE in "${OBJECTIVES[@]}"; do
    for ACQUISITION in "${ACQUISITIONS[@]}"; do
        for SEED in {0..4}; do
            for NOISE_VAL in "${NOISE[@]}"; do
                # Set up experiment name and log file paths
                EXPERIMENT_NAME="proposed_bo-v2_${OBJECTIVE}_AGT_${ACQUISITION}_seed${SEED}_noise${NOISE_VAL}"
                LOG_DIR="logs/${EXPERIMENTAL_ID}/train"

                # Run each experiment in parallel using sbatch
                sbatch --job-name="${EXPERIMENT_NAME}" \
                       --output="${LOG_DIR}/${EXPERIMENT_NAME}_%j.log" \
                       --cpus-per-task=$CPUS_PER_TASK \
                       --partition=$PARTITION \
                       --time=$TIME \
                       --wrap="python3 experiments/2024-09-23/proposed_bo_v2.py --seed $SEED --objective $OBJECTIVE --noise_strength $NOISE_VAL --acquisition $ACQUISITION --iterations $ITER"
            done
        done
    done
done
