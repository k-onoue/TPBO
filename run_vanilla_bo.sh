#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=4       # Number of CPUs per task
PARTITION="gpu_short" # Partition name
TIME="4:00:00"        # Maximum execution time

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p logs/

# Objectives, acquisitions, and surrogate models to test
OBJECTIVES=("SinusoidaSynthetic" "BraninHoo" "Hartmann6")
ACQUISITIONS=("UCB" "POI" "EI")
SURROGATES=("GP" "TP")  # GP and TP for different surrogate models

# Params
SEED=0  
ITER=500  
EXPERIMENTAL_ID="E1"

# Create directories based on experimental ID
mkdir -p logs/${EXPERIMENTAL_ID}/train/

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/TPBO
data_dir = \${project_dir}/data
results_dir = \${project_dir}/results
logs_dir = \${project_dir}/logs/\${EXPERIMENTAL_ID}"

# Overwrite config.ini file only if necessary
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Loop through each objective, acquisition, and surrogate model
for OBJECTIVE in "${OBJECTIVES[@]}"; do
    for ACQUISITION in "${ACQUISITIONS[@]}"; do
        for SURROGATE in "${SURROGATES[@]}"; do
            # Create the log directory if it doesn't exist
            mkdir -p "logs/${EXPERIMENTAL_ID}/train"

            # Set up experiment name and log file paths
            EXPERIMENT_NAME="vanilla_bo_${OBJECTIVE}_${SURROGATE}_${ACQUISITION}_seed[${SEED}]"
            LOG_DIR="logs/${EXPERIMENTAL_ID}/train"

            # Run each experiment in parallel using sbatch
            sbatch --job-name="${EXPERIMENT_NAME}" \
                   --output="${LOG_DIR}/${EXPERIMENT_NAME}_%j.log" \
                   --cpus-per-task=$CPUS_PER_TASK \
                   --partition=$PARTITION \
                   --time=$TIME \
                   --wrap="python3 experiments/2024-09-20/vanilla_bo.py --seed $SEED --objective $OBJECTIVE --acquisition $ACQUISITION --surrogate $SURROGATE --iterations $ITER"
        done
    done
done
