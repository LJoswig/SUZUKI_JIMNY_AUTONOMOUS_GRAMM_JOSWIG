#!/bin/bash

# Directory containing the Python scripts
LAUNCH_DIR="./src/mpc_driving_controller/launch"

# Check if the file name argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <waypoint_file_name>"
  exit 1
fi

# Get the desired waypoint file name from the command line
NEW_WAYPOINT_FILE_NAME="$1"

# Check if the launch directory exists
if [ ! -d "$LAUNCH_DIR" ]; then
  echo "Error: Launch directory '$LAUNCH_DIR' does not exist."
  exit 1
fi

# Loop through all Python launch files in the directory
for file in "$LAUNCH_DIR"/*.py; do
  if [ -f "$file" ]; then
    # Use sed to update the waypoint_file_name
    sed -i "s/waypoint_file_name = \".*\"/waypoint_file_name = \"$NEW_WAYPOINT_FILE_NAME\"/" "$file"
    echo "Updated waypoint_file_name in $file"
  fi
done

echo "All launch files in '$LAUNCH_DIR' have been updated."