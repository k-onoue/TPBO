#!/bin/bash

# Directory containing the files
DIRECTORY="experiments/2024-09-03_botorch_griewank"

# Loop over all files in the directory
for file in "$DIRECTORY"/*; do
  # Extract the base filename without the path
  base_name=$(basename "$file")
  
  # Replace 'Schubert' with 'Griewank' in the filename
  new_name=$(echo "$base_name" | sed 's/Schubert/Griewank/')
  
  # Rename the file
  mv "$file" "$DIRECTORY/$new_name"
done

echo "Renaming completed."