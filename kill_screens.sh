#!/bin/bash

# Stop all detached screens
screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs kill

# Stop one screen
# screen -X -S <session_ID> quit