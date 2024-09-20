#!/bin/bash

# SLURM Resource configuration
CPUS_PER_TASK=26           # Full node allocation for each job (26 cores)
PARTITION="cluster_long"    # Partition name
TIME="4:00:00"              # Maximum execution time
NODES="cc21cluster[33-38]"  # Specify the nodes you want to use

# Define log directory
LOG_DIR="logs_2"  # Change this variable to easily update log directory

# Create results and logs directories if they don't exist
mkdir -p results/
mkdir -p ${LOG_DIR}/
mkdir -p ${LOG_DIR}/train/

# Objectives and acquisitions to test
OBJECTIVES=("SinusoidaSynthetic" "BraninHoo" "Hartmann6")
ACQUISITIONS=("UCB" "POI" "EI")
SURROGATES=("GP" "TP")  # Add GP and TP for different surrogate models

# Params
SEED=1  
ITER=500  

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/TPBO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(project_dir)s/${LOG_DIR}"

# Overwrite config.ini file only if necessary
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
                   --output="${LOG_DIR}/train/${OBJECTIVE}_${ACQUISITION}_${SURROGATE}_%j.log" \
                   --cpus-per-task=$CPUS_PER_TASK \
                   --partition=$PARTITION \
                   --nodelist=$NODES \
                   --time=$TIME \
                   --wrap="python3 experiments/2024-09-20/vanilla_bo.py --seed $SEED --objective $OBJECTIVE --acquisition $ACQUISITION --surrogate $SURROGATE --iterations $ITER"
        done
    done
done
