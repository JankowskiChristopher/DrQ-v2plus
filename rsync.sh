#!/bin/bash

# ANSI color codes
RED='\033[0;31m' # Red
G='\033[0;32m' # Green
NC='\033[0m' # Reset color

# Display script usage
function show_usage {
  echo "Rsync script syncs the src and experiments directories to the host. Can also sync a single file."
  echo "Usage: $0 [-f] [-h]"
  echo "  -f: rsync one file with filename as argument"
  echo "  -h: Display this help message"
  exit 1
}

# Default values
src="./src"
experiments="./experiments"
remote_dir="krzysztofj@entropy.mimuw.edu.pl:/home/krzysztofj/distributional-sac/"

# Parse command-line options
while getopts ":f:h" opt; do
  case $opt in
    f)
      echo -e "${G}Syncing file $OPTARG to ${remote_dir}${NC}"
      rsync -av  "$OPTARG"  "$remote_dir"
      exit 0
      ;;
    h)
      show_usage
      ;;
    \?)
      echo -e "${RED}Error: Invalid option: -${OPTARG}${NC}"
      show_usage
      ;;
  esac
done

# Perform rsync of src and experiments directories. Files on receiver side not present in sender side are deleted.
echo -e "${G}Syncing $src to $remote_dir${NC}"
rsync -av --delete --exclude='__pycache__' --exclude="./outputs/" "$src"  "$remote_dir"
rsync -av --delete "$experiments/configs" "$remote_dir/experiments"

# Get results from the host
echo -e "${G}Syncing results from $remote_dir to $experiments${NC}"
scp -r "$remote_dir/experiments/results" "$experiments"

# copy csv_drqv2 from host
echo -e "${G}Syncing csv_drqv2 from $remote_dir to $experiments${NC}"
scp -r "$remote_dir/csv_drqv2" "$experiments"
