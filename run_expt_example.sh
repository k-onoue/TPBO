#!/bin/bash

# Set default parameters
DEFAULT_ITERATIONS=100
DEFAULT_SURROGATE="GP"

# Create a log directory if it doesn't exist
mkdir -p logs

# List of objectives to test
OBJECTIVES=("SinusoidaSynthetic" "BraninHoo" "Hartmann6")

# List of acquisition functions to test
ACQUISITIONS=("UCB" "POI" "EI")

# Loop through each objective and acquisition combination
for OBJECTIVE in "${OBJECTIVES[@]}"; do
    for ACQUISITION in "${ACQUISITIONS[@]}"; do
        # Run experiment for GP
        echo "Running experiment for Objective: $OBJECTIVE, Acquisition: $ACQUISITION, Surrogate: $DEFAULT_SURROGATE"
        python your_script.py --objective "$OBJECTIVE" --acquisition "$ACQUISITION" --surrogate "$DEFAULT_SURROGATE" --iterations "$DEFAULT_ITERATIONS" > "logs/${OBJECTIVE}_${ACQUISITION}_${DEFAULT_SURROGATE}.log" 2>&1

        # Run experiment for TP
        echo "Running experiment for Objective: $OBJECTIVE, Acquisition: $ACQUISITION, Surrogate: TP"
        python your_script.py --objective "$OBJECTIVE" --acquisition "$ACQUISITION" --surrogate "TP" --iterations "$DEFAULT_ITERATIONS" > "logs/${OBJECTIVE}_${ACQUISITION}_TP.log" 2>&1
    done
done

