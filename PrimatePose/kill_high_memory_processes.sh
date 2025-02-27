#!/bin/bash
# Set memory usage percentage threshold
MEM_THRESHOLD=2

# Get the PIDs of all processes that exceed the memory threshold
PIDS=$(ps -u ti -o pid,pmem,cmd --sort=-pmem | awk -v threshold=$MEM_THRESHOLD '{if($2 > threshold) print $1}')

# Terminate these processes
for pid in $PIDS
do
    echo "Killing PID $pid"
    kill -9 $pid
done
