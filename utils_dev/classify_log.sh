#!/bin/bash

# Directory containing log files
log_dir="logs/E3/"
# Destination directory for classified logs
output_dir="logs/E3/"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Loop over each log file in the log directory
for log_file in "$log_dir"/*.log; do
  # Check if the file exists
  if [[ -f "$log_file" ]]; then
    # Extract the noise_strength using grep and awk
    noise_strength=$(grep -m 1 "'noise_strength'" "$log_file" | awk -F "[:,]" '{gsub(/[^0-9.]/,"",$3); print $3}')

    # If noise_strength is found, proceed
    if [[ ! -z "$noise_strength" ]]; then
      # Create directory based on noise_strength
      noise_dir="$output_dir/noise_$noise_strength"
      mkdir -p "$noise_dir"
      
      # Move the log file to the appropriate directory
      mv "$log_file" "$noise_dir/"
      echo "Moved $log_file to $noise_dir"
    else
      echo "No noise_strength found in $log_file"
    fi
  fi
done
