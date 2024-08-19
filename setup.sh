#!/bin/bash

# Display script usage
function show_usage {
  echo "Setup script creates a virtual environment and optionally installs the requirements."
  echo "In order for the venv to work you need to call the script with source (see usage)."
  echo "To just see help, run the script without source."
  echo "Usage: source setup.sh [-i] [-h]"
  echo "  -i: Install packages from requirements.txt"
  echo "  -h: Display this help message"
  exit 1
}

venv_name="venv"
install_packages=false

# Parse command-line options
while getopts ":ih" opt; do
  case $opt in
    i)
      echo -e "\033[1;32mRequirements will be installed.\033[0m"
      install_packages=true
      ;;
    h)
      show_usage
      ;;
    \?)
      echo -e "\033[1;31mError: Invalid option: -$OPTARG\033[0m"
      show_usage
      ;;
  esac
done

# Check if the virtual environment already exists
if [ -d "$venv_name" ]; then
    echo -e "\033[1;32mVirtual environment $venv_name already exists.\033[0m"
else
    # Create a virtual environment
    python3 -m venv "$venv_name"
    echo -e "\033[1;32mVirtual environment $venv_name created.\033[0m"
fi

# Activate the virtual environment
if [ -z "${VIRTUAL_ENV}" ]; then
    source "$venv_name/bin/activate"
    echo -e "\033[1;32mVirtual environment $venv_name activated.\033[0m"
else
    echo -e "\033[1;32mAlready in a virtual environment.\033[0m"
fi

# Export Python path
export PYTHONPATH="$PYTHONPATH:$PWD"

# Install the requirements from requirements.txt if install_packages is true
if [ "$install_packages" = true ]; then
    echo -e "\033[1;32mInstalling requirements...\033[0m"
    pip install -r requirements.txt
    echo -e "\033[1;32mRequirements installed.\033[0m"
fi
