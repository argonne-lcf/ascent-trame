#!/bin/bash

# Check if Jupyter is running, start it if not
echo "---------------------------------------------------------------------------------"
pid=$(pgrep -f "jupyter-notebook")

if [[ -z $pid ]]; then
    echo "Jupyter is not running, starting on http://localhost:8888/tree"
    nohup jupyter notebook --ip 0.0.0.0 --port 8888 --IdentityProvider.token='' --no-browser --NotebookApp.token='' --NotebookApp.password='' >/dev/null 2>&1 &
else
    echo "Jupyter is already running, available on http://localhost:8888/tree"
    echo "PID=$pid"
fi

# Print the SSH command
echo ""
echo "Copy and run this command in a local terminal to forward the Jupyter server from the compute node to your local machine"
echo "ssh -v -N -L 8888:$(hostname):8888 $USER@polaris.alcf.anl.gov"
echo "---------------------------------------------------------------------------------"

# Give the user a chance to copy the command
echo ""
echo "Waiting 5 seconds before running the simulation..."
sleep 5
