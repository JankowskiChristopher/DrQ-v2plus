#!/bin/bash

# ANSI color codes
RED='\033[0;31m' # Red
G='\033[0;32m' # Green
NC='\033[0m' # Reset color

# Display script usage
function show_usage {
  echo "Runs the srun with all confings that are in the experiments/configs directory."
  echo "Usage: $0 [-p] [-q] [-g] [-t] [-f] [-m] [-h]"
  echo "  -p: Specify the partition - default is 'common'"
  echo "  -q: Specify the qos - default is '16gpu3d'"
  echo "  -g: Specify the number of gpus - default is '1'. Might be also <model:num> e.g. 'titanv:1'"
  echo "  -t: Specify the time - default is '3-0' meaning 3 days. Format is 'days-HH:MM:SS'"
  echo "  -f: Specify the main python file"
  echo "  -m: Specify the memory - default is '17000' meaning 17GB"
  echo "  -r: This flag enables profiling of the code."
  echo "  -h: Display this help message"
  exit 1
}

# default data for srun
partition="common"
qos="16gpu3d"
gpus=1 # either num e.g. 1 or gpu and num e.g. titanv:1
time="2-23:59:59" # 1-0 means 1 day
main_py="src/main.py"
profile=0
mem=20000

# get options
while getopts ":p:q:g:t:f:m:rh" opt; do
  case $opt in
    p)
      partition=$OPTARG
      ;;
    q)
      qos=$OPTARG
      ;;
    g)
      gpus=$OPTARG
      ;;
    t)
      time=$OPTARG
      ;;
    f)
      main_py=$OPTARG
      ;;
    h)
      show_usage
      ;;
    r)
      profile=1
      ;;
    m)
      mem=$OPTARG
      ;;
    \?)
      echo -e "${RED}Error: Invalid option: -$OPTARG${NC}"
      show_usage
      ;;
  esac
done

# Show the values to the user
echo -e "${G}Running srun with the following parameters:${NC}"
echo -e "${G}Partition:${NC} $partition"
echo -e "${G}Qos:${NC} $qos"
echo -e "${G}Gpus:${NC} $gpus"
echo -e "${G}Time:${NC} $time"
echo -e "${G}Memory:${NC} $mem"
echo -e "${G}Main python file:${NC} $main_py"

# Specify the directory path to experiments' configs
configs_dir="./experiments/configs"

# Iterate over all configs and check that directory and files have correct types.
if [ -d "$configs_dir" ]; then
  experiment_id=0
  for file in "$configs_dir"/*; do
    # Extract filename and check if file
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      filename="${filename%.*}"
      experiment_directory="experiment_$(date +%Y_%m_%d_%H-%M-%S)_${filename}_id_${experiment_id}"

      # make temporary directory for experiment
      mkdir "${experiment_directory}"
      mkdir "${experiment_directory}/experiments"

      # copy files to temporary directory
      cp -r ./src "$experiment_directory"
      cp -r ./experiments/configs "${experiment_directory}/experiments/configs"

      # Specify directories for srun
      results_output="./experiments/results/${experiment_directory}"
      main_file_dir="./${experiment_directory}/${main_py}"

      # Run srun
      if [ $profile -eq 1 ]; then
        echo -e "${G}Profiling enabled.${NC}"
        echo -e "${G}Starting srun for${NC} ${filename}${G} code from ${NC}${main_file_dir} and results saved to ${results_output}.txt and ${results_output}.profile"
#        screen -dm bash -c "trap 'rm -r ${experiment_directory}' EXIT; srun --partition=${partition} --qos=${qos} --gres=gpu:${gpus} --time=${time} --mem=${mem} python3 -m cProfile -o ${results_output}.profile ${main_file_dir} -cn ${filename} 2>&1 | tee ${results_output}.txt"
        screen -dm bash -c "trap 'rm -r ${experiment_directory}' EXIT; srun --partition=${partition} --qos=${qos} --gres=gpu:${gpus} --time=${time} --mem=${mem} singularity exec --nv dreamer/ python3 -m cProfile -o ${results_output}.profile ${main_file_dir} -cn ${filename} 2>&1 | tee ${results_output}.txt"
      else
        echo -e "${G}Starting srun for${NC} ${filename}${G} code from ${NC}${main_file_dir} and results saved to ${results_output}.txt"
#       screen -dm bash -c "trap 'rm -r ${experiment_directory}' EXIT; srun --partition=${partition} --qos=${qos} --gres=gpu:${gpus} --time=${time} --mem=${mem} python3 ${main_file_dir} -cn ${filename} 2>&1 | tee ${results_output}.txt"
#       manually set interpreter path as due to Conda it stopped working.
        screen -dm bash -c "trap 'rm -r ${experiment_directory}' EXIT; srun --partition=${partition} --qos=${qos} --gres=gpu:${gpus} --time=${time} --mem=${mem} singularity exec --nv dreamer/ /home/krzysztofj/distributional-sac/venv/bin/python ${main_file_dir} -cn ${filename} 2>&1 | tee ${results_output}.txt"

      fi

      # Increase id for next experiment
      experiment_id=$((experiment_id+1))
    fi
  done
else
  echo "${RED}Directory not found: $configs_dir${NC}"
fi
