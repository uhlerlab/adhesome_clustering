# Import standard libraries
import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import adjusted_mutual_info_score
import csv
import pandas as pd
import networkx as nx
import community
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import math
import time
from tqdm import tqdm
import random
# Custom libraries
import utils as lu
import correlation_clustering as cc

''' This script performs a robustness analysis of clustering techniques (weighted correlation clustering and 
hierarchical clustering) on the network of adhesome genes across different Hi-C thresholds and presence/absence 
of intraX edges (used to build the adhesome network).'''

def parse_config(config_filename):
    '''
    Reads config file
    Args:
        config_filename: (string) configuration file name
    Returns:
        A dictionary specifying the main directories, cell type, resolution, quality, chromosomes
    '''
    config = json.load(open(config_filename))
    return(config)


def main():
    
    # Parse command line arguments
    argv = sys.argv[1:]
    try:
        options, args = getopt.getopt(argv, "c:", ["config="])
    except:
        print('Incorrect arguments!')
        
    for name, value in options:
        if name in ('-c', '--config'):
            config_filename = value
    
    # Parse config file
    print('Options successfully parsed, read arguments...')
    config = parse_config(config_filename)
    genome_dir = config['GENOME_DIR']
    processed_hic_data_dir = config['PROCESSED_HIC_DATA_DIR']
    epigenome_dir = config['EPIGENOME_DIR']
    processed_epigenome_data_dir = config['PROCESSED_EPIGENOME_DIR']
    adhesome_dir = config['ADHESOME_DIR']
    saving_dir = config['SAVING_DIR']
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    
    ##################################################################
    # Load data and set parameters
    ##################################################################
    print('Load data and set parameters')
    
    # Load data
    adhesome_interX_edge_list = pickle.load(open(saving_dir+'adhesome_interX_edge_list.pkl', 'rb'))
    adhesome_intraX_edge_list = pickle.load(open(saving_dir+'adhesome_intraX_edge_list.pkl', 'rb')) 
    active_adhesome_genes = pickle.load(open(saving_dir+'active_adhesome_genes.pkl', 'rb'))
    adhesome_loc_corr = pickle.load(open(saving_dir+'adhesome_loc_corr.pkl','rb'))
    
    # Add pairwise Spearman correlations between adhesome genes to edge lists
    adhesome_interX_edge_list['spearman_corr'] = [adhesome_loc_corr.loc[adhesome_interX_edge_list.iloc[i]['source'],
                                                                    adhesome_interX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_interX_edge_list))]
    adhesome_intraX_edge_list['spearman_corr'] = [adhesome_loc_corr.loc[adhesome_intraX_edge_list.iloc[i]['source'],
                                                                    adhesome_intraX_edge_list.iloc[i]['target']]
                                               for i in range(len(adhesome_intraX_edge_list))]
 
    # Set parameters
    hic_threshold_list = np.arange(0.70,0.82,0.02)
    weights = 'spearman_corr'
    hc_threshold = 0.62

    ##################################################################
    # Robustness of weighted correlation clustering (w/ intraX edges)
    ##################################################################
    print('Robustness of WCC (w/ intraX edges)')
    
    # Parameters
    with_intra = True

    # Dictionary to store clusterings
    wcc_dict = dict()
    # Table to store adjusted mutual information
    ami_table = pd.DataFrame(1,index=hic_threshold_list, columns=hic_threshold_list)

    # Compute clusterings
    for t in hic_threshold_list:
        wcc = cc.weighted_correlation_clustering_micha(adhesome_interX_edge_list, 
                                                       adhesome_intraX_edge_list,
                                                       active_adhesome_genes,
                                                       t,
                                                       with_intra,
                                                       weights,
                                                       num_calls=100)
        wcc_dict[t] = wcc[0]
    pickle.dump(wcc_dict, open(saving_dir+'dict_intra_wcc.pickle','wb'))

    # Fill in Adjusted Mutual Information between clusterings
    for t,u in itertools.combinations(hic_threshold_list,2):
        ami_table.loc[t,u] = adjusted_mutual_info_score(labels_true=wcc_dict[t],labels_pred=wcc_dict[u],
                                                        average_method='min')
        ami_table.loc[u,t] = ami_table.loc[t,u]
    pickle.dump(ami_table, open(saving_dir+'ami_intra_wcc.pickle','wb'))
    
    ##################################################################
    # Robustness of hierarchical clustering (w/ intraX edges)
    ##################################################################
    print('Robustness of HC (w/ intraX edges)')
    
    # Parameters
    with_intra = True

    # Dictionary to store clusterings
    hc_dict = dict()
    # Table to store adjusted mutual information
    ami_table = pd.DataFrame(1,index=hic_threshold_list, columns=hic_threshold_list)

    # Compute clusterings
    for t in hic_threshold_list:
        hc_dict[t] = cc.hierarchical_clustering(adhesome_interX_edge_list, 
                                                adhesome_intraX_edge_list,
                                                active_adhesome_genes,
                                                t,
                                                with_intra,
                                                weights,
                                                hc_threshold)
    pickle.dump(hc_dict, open(saving_dir+'dict_intra_hc.pickle','wb'))

    # Fill in Adjusted Mutual Information between clusterings
    for t,u in itertools.combinations(hic_threshold_list,2):
        ami_table.loc[t,u] = adjusted_mutual_info_score(labels_true=hc_dict[t],labels_pred=hc_dict[u],
                                                        average_method='min')
        ami_table.loc[u,t] = ami_table.loc[t,u]
    pickle.dump(ami_table, open(saving_dir+'ami_intra_hc.pickle','wb'))
    
    ##################################################################
    # Robustness of weighted correlation clustering (no intraX edges)
    ##################################################################
    print('Robustness of WCC (w/o intraX edges)')
    
    # Parameters
    with_intra = False

    # Dictionary to store clusterings
    wcc_dict = dict()
    # Table to store adjusted mutual information
    ami_table = pd.DataFrame(1,index=hic_threshold_list, columns=hic_threshold_list)

    # Compute clusterings
    for t in hic_threshold_list:
        wcc = cc.weighted_correlation_clustering_micha(adhesome_interX_edge_list, 
                                                       adhesome_intraX_edge_list,
                                                       active_adhesome_genes,
                                                       t,
                                                       with_intra,
                                                       weights,
                                                       num_calls=100)
        wcc_dict[t] = wcc[0]
    pickle.dump(wcc_dict, open(saving_dir+'dict_nointra_wcc.pickle','wb'))

    # Fill in Adjusted Mutual Information between clusterings
    for t,u in itertools.combinations(hic_threshold_list,2):
        ami_table.loc[t,u] = adjusted_mutual_info_score(labels_true=wcc_dict[t],labels_pred=wcc_dict[u],
                                                        average_method='min')
        ami_table.loc[u,t] = ami_table.loc[t,u]
    pickle.dump(ami_table, open(saving_dir+'ami_nointra_wcc.pickle','wb'))
    
    ##################################################################
    # Robustness of hierarchical clustering (no intraX edges)
    ##################################################################
    print('Robustness of HC (w/o intraX edges)')
    
    # Parameters
    with_intra = False

    # Dictionary to store clusterings
    hc_dict = dict()
    # Table to store adjusted mutual information
    ami_table = pd.DataFrame(1,index=hic_threshold_list, columns=hic_threshold_list)

    # Compute clusterings
    for t in hic_threshold_list:
        hc_dict[t] = cc.hierarchical_clustering(adhesome_interX_edge_list, 
                                                adhesome_intraX_edge_list,
                                                active_adhesome_genes,
                                                t,
                                                with_intra,
                                                weights,
                                                hc_threshold)
    pickle.dump(hc_dict, open(saving_dir+'dict_nointra_hc.pickle','wb'))

    # Fill in Adjusted Mutual Information between clusterings
    for t,u in itertools.combinations(hic_threshold_list,2):
        ami_table.loc[t,u] = adjusted_mutual_info_score(labels_true=hc_dict[t],labels_pred=hc_dict[u],
                                                        average_method='min')
        ami_table.loc[u,t] = ami_table.loc[t,u]
    pickle.dump(ami_table, open(saving_dir+'ami_nointra_hc.pickle','wb'))
    
    

if __name__ == "__main__":
    main()
    