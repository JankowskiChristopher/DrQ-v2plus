#!/bin/bash

# Set the directory path
results_dir="./experiments/results"

# Find the latest file based on creation time
latest_file=$(find "$results_dir" -maxdepth 1 -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

# Check if a file was found
if [ -n "$latest_file" ]; then
    # Display the contents of the latest file using cat
    cat "$latest_file"
else
    echo "No files found in $results_dir directory."
fi
