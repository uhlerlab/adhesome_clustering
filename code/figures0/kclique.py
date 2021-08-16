# Import libraries
import numpy as np
from scipy import sparse
import scipy.stats as ss
import networkx as nx
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import os.path
import math
import pickle
import networkx as nx
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community.kclique import k_clique_communities
import itertools
import random

def main():
    
    # Load edge list
    saving_dir = '/home/louiscam/projects/gpcr/save/figures/'
    adhesome_edge_list = pickle.load(open(saving_dir+'edge_list_hclust.pkl', 'rb'))
    
    # Initialize
    list_thresholds = sorted(np.arange(0.5,1,0.05), reverse=True)
    list_k = np.arange(3,11)
    results_df = pd.DataFrame(index = range(len(list_thresholds)*len(list_k)),
                              columns=['n_nodes_G',
                                       't',
                                       'k',
                                       'n_kcliques',
                                       'size_kcliques',
                                       'percent_nodes_in_kcliques',
                                       'percent_multiclique_nodes',
                                       'list_kcliques'])
    i = 0
    for t in list_thresholds:
        print('Current threshold = '+str(t))
        # Threshold edgelist
        adhesome_edge_list_t = adhesome_edge_list[adhesome_edge_list['scaled_hic']>t]
        # Create graph
        G = nx.from_pandas_edgelist(adhesome_edge_list_t, edge_attr=['hic','scaled_hic'])

        # Node labels as integers
        F = G.copy()
        mapping = {list(F.nodes)[i]:i for i in range(len(F.nodes))}
        F = nx.relabel_nodes(F,mapping)

        # Find cliques in F
        cliques = list(nx.find_cliques(F))
        cliques = [set(c) for c in cliques]

        for k in list_k:

            # Apply k-clique
            frozenlist_kcliques = list(k_clique_communities(F, k, cliques))
            list_kcliques = [np.array(list(c)) for c in frozenlist_kcliques]

            # Fill out results_df
            results_df.iloc[i]['n_nodes_G'] = len(G.nodes)
            results_df.iloc[i]['t'] = t
            results_df.iloc[i]['k'] = k
            results_df.iloc[i]['n_kcliques'] = len(list_kcliques)
            results_df.iloc[i]['size_kcliques'] = [len(c) for c in list_kcliques]
            results_df.iloc[i]['percent_nodes_in_kcliques'] = sum([len(c) for c in list_kcliques])/len(G.nodes)
            results_df.iloc[i]['percent_multiclique_nodes'] = np.sum(np.unique(list(itertools.chain.from_iterable(list_kcliques)), return_counts=True)[1]>1)/len(G.nodes)
            results_df.iloc[i]['list_kcliques'] = list_kcliques
            i = i+1

    pickle.dump(results_df, open(saving_dir+'kclique_results_df.pkl', 'wb'))


if __name__ == "__main__":
    main()