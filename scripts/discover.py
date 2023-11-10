#!/usr/bin/env python3

import os
import sys
import time
from utils import *

import numpy as np

dataset = sys.argv[1]
dataset_list = ['fb15k-237', 'wnrr', 'yago3-10', 'codex-l']

if dataset not in dataset_list:
    raise ValueError('Please select from: fb15k-237, wnrr, yago3-10, codex-l')

print(f'Extracting from dataset {dataset} -----------------------------------------------')

path_to_data = f'../kge/data/{dataset}'

## Customizable
## --------------------------------------------------------------------------------
# Load dataset
X = np.loadtxt(f'{path_to_data}/train.del', dtype=str)
valid = np.loadtxt(f'{path_to_data}/valid.del', dtype=str)
test = np.loadtxt(f'{path_to_data}/test.del', dtype=str)

filter_triples = np.vstack((X, valid, test))
## --------------------------------------------------------------------------------

# Get all relations
rel_list = np.unique(X[:, 1])

strategy_list = ['random_uniform', 'entity_frequency', 'graph_degree', 'cluster_coefficient', 'cluster_triangles']

for strategy in strategy_list:
    print(f'Extracting using {strategy}')
    discoveries = []
    generation_time = 0

    for rel in rel_list:
        print(f'Generating candidates for relation: {rel}')

        start_generation = time.time()
        candidates = generate_candidates(X, strategy, rel, max_candidates=500, seed=0)
        end_generation = time.time()
        generation_time += end_generation - start_generation

        print(f'Generated {len(candidates)} new candidates')
        discoveries.append(candidates)
    
    discoveries = flatten_list(discoveries)
    discoveries_arr = np.vstack(discoveries)

    # Filter positive triples from extracted facts.
    filter_set = {tuple(row) for row in filter_triples}
    filtered_triples = [row for row in discoveries_arr if tuple(row) not in filter_set]
    discoveries_arr = np.array(filtered_triples)

    np.savetxt(f'{path_to_data}/{strategy}.del', discoveries_arr, delimiter='\t', fmt='%s')
    
    # Save extraction logs
    log_path = f'discovery_logs/{dataset}/{strategy}'
    if not os.path.exists(log_path):
            # Create the directory if it does not exist
            os.makedirs(log_path)

    with open(f'{log_path}/log.txt', 'a') as f:
        f.write(f"Generation time: {generation_time}\nNum discoveries: {len(discoveries)}\n")

print(f'Fact generation for {dataset} complete.')