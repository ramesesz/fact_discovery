#!/bin/bash

args1=("wnrr")
args2=("rotate" "transe")
args3=("random_uniform" "entity_frequency" "graph_degree" "cluster_coefficient" "cluster_triangles")

for arg1 in "${args1[@]}"
do
    for arg2 in "${args2[@]}"
    do
        for arg3 in "${args3[@]}"
        do
            command="$arg1 $arg2 $arg3"
            echo "Executing command: $command"
            python eval.py "$arg1" "$arg2" "$arg3"
        done
    done
done