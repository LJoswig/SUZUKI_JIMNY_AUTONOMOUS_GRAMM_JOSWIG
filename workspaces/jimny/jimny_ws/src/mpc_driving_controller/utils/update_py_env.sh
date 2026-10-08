#!/bin/bash

# Directory containing the Python scripts
SCRIPTS_DIR="./src/mpc_driving_controller/scripts"

# Get the path of the Python3 interpreter
PYTHON_PATH=$(which python3)

# Loop through each Python file in the directory
for file in "$SCRIPTS_DIR"/*.py; do
  # Check if the file exists and is a regular file
  if [[ -f "$file" ]]; then
    # Print a message indicating the file is being processed
    echo "Processing $file..."

    # Update the first line of the file to be the new shebang line
    # Use 'sed' to replace the first line with the new shebang
    sed -i "1s|.*|#!$PYTHON_PATH|" "$file"

    # Print a message indicating the file was updated
    echo "Updated shebang line to #!$PYTHON_PATH in $file"
  fi
done
