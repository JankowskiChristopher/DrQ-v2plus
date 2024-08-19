# distributional-sac

## Running experiments

Locally run script

```bash
rsync.sh
```

Or with option -n (see description below)

On Entropy cluster start the virtual environment:

```bash
source setup.sh
```

If you need to install (this won't probably activate venv) requirements run:

```bash
setup.sh -i 
```

For some reasons this time without source.
And run the job script:

```bash
srun_experiment.sh
```

## Important commands

### Entropy

To see the current queue of jobs run (either version):

```bash
squeue
entropy_queue
entropy_my_queue
```

To see the stats and memory allocated by jobs run:
```bash
sacct --format='JobID,Elapsed,MaxVMSize'
```

### Python path

It's possible that you will have to specify the Python path.
cd into the project directory (distributional-sac) and run:

```bash
export PYTHONPATH="$PYTHONPATH:$PWD"
```

This line is also added to the setup.sh script, so probably no need to run it manually.

### Linux screen

Srun script uses the screen functionality, to run the jobs in the background even when the terminal is closed.
To see the list of screens run:

```bash
screen -ls
```

To attach to a screen run:

```bash
screen -r <screen_name>
```

To detach from a screen press Ctrl+A and then D.
To kill all screens see script kill_screens.sh below.

## Useful scripts

When running the scripts for the first time remember to run:

```bash
chmod u+x <script_name>.sh
```

### setup.sh

Scripts sets up the environment on the cluster and optionally installs the dependencies.
**Note: in order to work must be used with source**

```bash
source setup.sh
```

And to create (if does not already exist) and activate the virtual environment with installation from requirements.txt:

```bash
source setup.sh -i
```

### rsync.sh

Rsync sends the code to the cluster.
**Remember to change the paths to your username in the script**
When used without flags it copies the whole src to the cluster under directory src:

```bash
./rsync.sh
```

When used with flag -n it copies the src but puts it inside an experiment folder with a timestamp.

```bash
./rsync.sh -n
```

### srun_experiment.sh

Script runs the experiment on the cluster by creating a screen and later detaching it. The results of srun are copied
from stdout and stderr to a file called results.txt that is stored in the directory of the experiment.
Usage with default options (change them for your defaults) is:

```bash
./srun_experiment.sh
```

You can specify the options (see help of the script) like this:

```bash
./srun_experiment.sh -n <name e.g. common without ""> -q <qos e.g. student> -g <e.g. titanv:2> -t <time e.g. 1-0 meaning 1 day> -m <path to main.py>
```

### kill_screens.sh

Script kills all the detached screens.

```bash
./kill_screens.sh
```

## Singularity

### Installation
https://www.linuxwave.info/2022/02/installing-singularity-in-ubuntu-2004.html

### Build image
```bash
sudo singularity build <image_name.sif> <image_file.def>
```