import sys
import os
import json
import csv

import MappingScripts as ms
import networkx as nx
import numpy as np

# delete TODO
from datetime import datetime

from Setup.MappingSetup import ForceFeatureRecomputation, ForceDataRecomputation
from Setup.initialize import workingPath

# DFS dummy data
# graph = nx.generators.random_graphs.gnp_random_graph(10, 0.50, seed=2)
# nx.drawing.nx_pylab.draw_networkx(graph)
# adj = nx.linalg.graphmatrix.adjacency_matrix(graph).tolil()
# adj = adj.astype(float)
# i, j = adj.nonzero()
# for ind in range(len(i)):
#     adj[i[ind], j[ind]] = np.random.random_sample()
# for i in range(10):
#     adj[i, i] = 1

# paths = ms.DFS(adj, thresh=0.5, startInd=0, finalInd=5, maxDepth=3)



Names = []
if os.path.exists(workingPath + 'Names.csv'):
    with open(workingPath + 'Names.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            Names = row
else:
    raise Exception('Names not found. Terminating, please run Execute01_RunPreparation first')

Flags = {}
if os.path.exists(workingPath + 'Flags.json'):
    with open(workingPath + 'Flags.json') as json_file:
        Flags = json.load(json_file)
else:
    print('Please run Execute01_RunPreparation first!')
    raise Exception('Terminating, running scripts out of order')

Flags = ms.MappingSetupAndFlowExtraction(Names, Flags)


if not ('featureMappings' in Flags) or ForceFeatureRecomputation:
    start = datetime.now()
    ms.ComputeFeatureMatching(Names, Flags)
    print("Compute Feature Matching took:", datetime.now() - start)
else:
    print('Feature mappings already computed')
    # FeatureMatches = np.load(workingPath + 'MappingData/FeatureMatches.npy')
