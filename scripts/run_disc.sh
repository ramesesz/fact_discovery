#!/bin/bash

# Define the arguments
args=("wnrr" "fb15k-237" "codex-l" "yago3-10")

# Loop over the arguments and run the Python script
for arg in "${args[@]}"
do
    python discover.py "$arg"
done