#!/usr/bin/env python3

import os
import sys
import time

import numpy as np
import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint

dataset = sys.argv[1]
model = sys.argv[2]
strategy = sys.argv[3]

dataset_list = ['fb15k-237', 'wnrr', 'yago3-10', 'codex-l', 'wikidata5m']
if dataset not in dataset_list:
    raise ValueError('Please select from: fb15k-237, wnrr, yago3-10, codex-l, wikidata5m')

model_list = ['complex', 'conve', 'distmult', 'rescal', 'rotate', 'transe']
if model not in model_list:
    raise ValueError('Please select from: complex, conve, distmult, rescal, rotate, transe')

strategy_list = ['random_uniform', 'entity_frequency', 'graph_degree', 'cluster_coefficient', 'cluster_triangles']
if strategy not in strategy_list:
    raise ValueError('Please select from: random_uniform, entity_frequency, graph_degree, cluster_coefficient, cluster_triangles')

print(f'\nEvaluating {strategy} facts from dataset {dataset} with {model} ----------------------------------\n')

# Load extracted facts
path_to_data = f'../kge/data/{dataset}'
X = np.loadtxt(f'{path_to_data}/{strategy}.del', dtype=int)

# Load model
checkpoint = load_checkpoint(f'./models/{dataset}/{model}.pt')
kge_model = KgeModel.create_from(checkpoint)

ranks = []

# Calculate ranks of triples
start_evaluation = time.time()

## Customizable
## --------------------------------------------------------------------------------
for triple in X:
    # Calculate score of triple
    s = torch.LongTensor([triple[0]])
    p = torch.LongTensor([triple[1]]) 
    o = torch.LongTensor([triple[2]])

    triple_score_s = kge_model.score_spo(s, p, o, direction="s")
    triple_score_o = kge_model.score_spo(s, p, o, direction="o")
    triple_score = max(triple_score_s, triple_score_o)
    triple_score = triple_score.tolist()[0]

    # Calculate corruption scores
    head_corruption_scores = kge_model.score_po(p, o)
    head_corruption_scores_list = head_corruption_scores.tolist()
    head_corruption_scores_list = [item for sublist in head_corruption_scores_list for item in sublist] 
    head_corruption_scores_list.append(triple_score)
    head_corruption_scores_list.sort(reverse=True)
    
    tail_corruption_scores = kge_model.score_sp(s, p)
    tail_corruption_scores_list = tail_corruption_scores.tolist()
    tail_corruption_scores_list = [item for sublist in tail_corruption_scores_list for item in sublist]
    tail_corruption_scores_list.append(triple_score)
    tail_corruption_scores_list.sort(reverse=True)

    # Calculate rank
    head_rank = head_corruption_scores_list.index(triple_score) + 1
    tail_rank = tail_corruption_scores_list.index(triple_score) + 1
    rank = np.mean([head_rank, tail_rank])
    ranks.append(rank)
## --------------------------------------------------------------------------------

# Filter triples and ranks with top_n=500
mask = np.array(ranks) <= 500
X_filtered = np.array(X)[mask]
ranks_filtered = np.array(ranks)[mask]

end_evaluation = time.time()
evaluation_time = end_evaluation - start_evaluation

# Save filtered triples and ranks
log_path = f'discovered_facts/{dataset}/{model}/{strategy}'
if not os.path.exists(log_path):
        # Create the directory if it does not exist
        os.makedirs(log_path)

np.savetxt(f'{log_path}/ranks.del', ranks_filtered, fmt='%s')
np.savetxt(f'{log_path}/triples.del', X_filtered, fmt='%s')

# Save metrics
with open(f'{log_path}/log.txt', 'a') as f:
    f.write(f"Corrected --------------------------------")
    f.write(f"Evaluation time: \t{evaluation_time}\n")
    f.write(f"Filtered triples: \t{len(X_filtered)}\n")
    f.write(f"MR: \t\t\t{np.mean(ranks_filtered)}\n")
    f.write(f"MRR: \t\t\t{np.mean(1/np.array(ranks_filtered))}\n")
    
print(f'\nEvaluation finished --------------------------------------------------------------------\n')    