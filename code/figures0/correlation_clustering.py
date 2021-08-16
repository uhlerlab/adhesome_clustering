# Import standard libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import csv
import pandas as pd
import networkx as nx
import community
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
import time
from tqdm import tqdm
import random
import subprocess


def create_interX_edgelist(contacts_df, selected_loci, locus_gene_dict):
    '''
    Creates an interX edge list between genes corresponding to a list of loci
    Args:
        contacts_df: (pandas DataFrame) a dataframe containing HiC contacts between loci
        selected_loci: (Numpy array) the set of loci of interest
        locus_gene_dict: (dict) a dictionary mapping each locus to the corresponding genes
    Return:
        An interX edge list (pandas DataFrame) with min-max scaled HiC values
    '''
    # Stack HiC matrix
    selected_interX_edge_list = contacts_df.loc[selected_loci,
                                                selected_loci].stack().reset_index()
    selected_interX_edge_list.columns = ['locus_source', 'locus_target', 'hic']
    selected_interX_edge_list['id'] = ['_'.join(np.sort(selected_interX_edge_list.iloc[i][['locus_source','locus_target']].values)) 
                                       for i in range(len(selected_interX_edge_list))]
    selected_interX_edge_list = selected_interX_edge_list.sort_values(['id','locus_source','locus_target'])

    # Drop duplicate interactions (same interaction partners)
    selected_interX_edge_list = selected_interX_edge_list.drop_duplicates('id')

    # Add gene labels to the source and target loci
    selected_interX_edge_list = selected_interX_edge_list[selected_interX_edge_list['hic']>0]
    selected_interX_edge_list['source'] = [locus_gene_dict[selected_interX_edge_list.iloc[i,0]] 
                                           for i in np.arange(len(selected_interX_edge_list))]
    selected_interX_edge_list['target'] = [locus_gene_dict[selected_interX_edge_list.iloc[i,1]] 
                                           for i in np.arange(len(selected_interX_edge_list))]
    selected_interX_edge_list = selected_interX_edge_list[['source','target','hic']]
    selected_interX_edge_list = selected_interX_edge_list.sort_values(by=['source','target'])

    # Explode source and target
    assign_target = selected_interX_edge_list.assign(**{'target':selected_interX_edge_list['target'].str.split('_')})
    explode_target = pd.DataFrame({col:np.repeat(assign_target[col].values, assign_target['target'].str.len()) 
                                   for col in assign_target.columns.difference(['target'])}).assign(**{'target':np.concatenate(assign_target['target'].values)})[assign_target.columns.tolist()]
    assign_source = explode_target.assign(**{'source':explode_target['source'].str.split('_')})
    explode_source = pd.DataFrame({col:np.repeat(assign_source[col].values, assign_source['source'].str.len()) 
                                   for col in assign_source.columns.difference(['source'])}).assign(**{'source':np.concatenate(assign_source['source'].values)})[assign_source.columns.tolist()]
    selected_interX_edge_list = explode_source

    # Aggregate interactions between same genes
    selected_interX_edge_list = selected_interX_edge_list.groupby(['source','target'])['hic'].agg('mean').reset_index()

    # Add scaled_hic values
    selected_interX_edge_list['scaled_hic'] = (selected_interX_edge_list['hic']-selected_interX_edge_list['hic'].min())/(selected_interX_edge_list['hic'].max()-selected_interX_edge_list['hic'].min())
    return(selected_interX_edge_list)


def create_intraX_edgelist(selected_loci, locus_gene_dict, df_loc, resol, dir_processed_hic):
    '''
    Creates an intraX edge list between genes corresponding to a list of loci
    Args:
        selected_loci: (Numpy array) the set of loci of interest
        locus_gene_dict: (dict) a dictionary mapping each locus to the corresponding genes
        df_loc: (pandas DataFrame) dtaframe containing informaion on gene locations in the genome
        resol: (int) HiC resolution
        dir_processed_hic: (String) directory of processed HiC maps
    Return:
        An intraX edge list (pandas DataFrame) with min-max scaled HiC values
    '''
    # Dictionary of adhesome loci per chromosome
    loci_per_chrom_dict = {chrom: sorted([int(locus.split('_')[3]) for locus in selected_loci if (int(locus.split('_')[1])==chrom)])
                               for chrom in np.arange(1,22+1,1)}

    # Create edge list containing intraX edges between active loci
    chr_list = np.arange(2,22+1,1)
    selected_intraX_edge_list = pd.DataFrame(columns=['source','target','hic','chrom','gen_dist'])
    for chrom in tqdm(chr_list):
        time.sleep(.01)
        subindex = ['chr_'+str(chrom)+'_'+'loc_'+str(loc) for loc in loci_per_chrom_dict[chrom]]

        # Load HiC data for this chromosome pair
        processed_hic_filename = 'hic_'+'chr'+str(chrom)+'_'+'chr'+str(chrom)+'_norm1_filter3'+'.pkl'
        hic_chpair_df = pickle.load(open(dir_processed_hic+processed_hic_filename, 'rb'))

        # Select adhesome loci and stack HiC matrix
        hic_chpair_adh_tf_df = hic_chpair_df.loc[loci_per_chrom_dict[chrom],loci_per_chrom_dict[chrom]]
        hic_chpair_adh_tf_df.index = subindex
        hic_chpair_adh_tf_df.columns = subindex
        new_index = pd.MultiIndex.from_tuples(itertools.combinations(subindex,2), names=["locus_source","locus_target"])
        hic_chpair_df1 = hic_chpair_adh_tf_df.stack().reindex(new_index).reset_index(name='hic')

        # Drop duplicate interactions (same interaction partners)
        hic_chpair_df1['id'] = ['_'.join(np.sort(hic_chpair_df1.iloc[i][['locus_source','locus_target']].values)) 
                                           for i in range(len(hic_chpair_df1))]
        hic_chpair_df1 = hic_chpair_df1.sort_values(['id','locus_source','locus_target'])
        hic_chpair_df1 = hic_chpair_df1.drop_duplicates('id')
        hic_chpair_df1 = hic_chpair_df1[['locus_source','locus_target','hic']]

        # Add gene labels to the source and target loci
        hic_chpair_df1 = hic_chpair_df1[hic_chpair_df1['hic']>0]
        hic_chpair_df1['source'] = [locus_gene_dict[hic_chpair_df1.iloc[i,0]]
                                    for i in np.arange(len(hic_chpair_df1))]
        hic_chpair_df1['target'] = [locus_gene_dict[hic_chpair_df1.iloc[i,1]]
                                    for i in np.arange(len(hic_chpair_df1))]
        hic_chpair_df1 = hic_chpair_df1[['source','target','hic']]
        hic_chpair_df1 = hic_chpair_df1.sort_values(by=['source','target'])

        # Explode source and target
        assign_target = hic_chpair_df1.assign(**{'target':hic_chpair_df1['target'].str.split('_')})
        explode_target = pd.DataFrame({col:np.repeat(assign_target[col].values, assign_target['target'].str.len()) 
                                       for col in assign_target.columns.difference(['target'])}).assign(**{'target':np.concatenate(assign_target['target'].values)})[assign_target.columns.tolist()]
        assign_source = explode_target.assign(**{'source':explode_target['source'].str.split('_')})
        explode_source = pd.DataFrame({col:np.repeat(assign_source[col].values, assign_source['source'].str.len()) 
                                       for col in assign_source.columns.difference(['source'])}).assign(**{'source':np.concatenate(assign_source['source'].values)})[assign_source.columns.tolist()]
        hic_chpair_df1 = explode_source

        # Aggregate interactions between same genes
        hic_chpair_df1 = hic_chpair_df1.groupby(['source','target'])['hic'].agg('mean').reset_index()

        # Add chromosome information
        hic_chpair_df1['chrom'] = chrom

        # Add genomic distance information
        hic_chpair_df1['gen_dist'] = [np.abs(df_loc[df_loc['geneSymbol']==hic_chpair_df1.iloc[i]['source']][['chromStart','chromEnd']].values.mean()/resol -
                                             df_loc[df_loc['geneSymbol']==hic_chpair_df1.iloc[i]['target']][['chromStart','chromEnd']].values.mean()/resol)
                                     for i in range(len(hic_chpair_df1))]

        # Concatenate to intraX edge list for previous chromosomes
        selected_intraX_edge_list = pd.concat([selected_intraX_edge_list,hic_chpair_df1])

    selected_intraX_edge_list = selected_intraX_edge_list.sort_values(by=['source','target'])

    # Add scaled_hic values
    selected_intraX_edge_list['scaled_hic'] = (selected_intraX_edge_list['hic']-selected_intraX_edge_list['hic'].min())/(selected_intraX_edge_list['hic'].max()-selected_intraX_edge_list['hic'].min())

    # Drop self-interactions
    selected_intraX_edge_list = selected_intraX_edge_list[selected_intraX_edge_list['source']!=selected_intraX_edge_list['target']]
    return(selected_intraX_edge_list)


def compute_costs(G, edge_attribute):
    '''
    Function to compute costs for the VOTE algorithm
    Args:
        G: (networkX graph) the network on which to run VOTE
        edge_attribute: (String) the edge attribute used to comput costs
    Return:
        A list including the adjacency matrix and the cost matrices
    '''
    # Get adjacency matrix
    nodes_sorted = sorted(G.nodes())
    A = nx.to_pandas_adjacency(G, weight=edge_attribute).reindex(index=nodes_sorted, columns=nodes_sorted)
    A = np.matrix(A)
    np.fill_diagonal(A,0)
    # Get cost of removing edges
    w_plus = np.array(A)
    # Get cost of cutting edges
    w_minus = 1-w_plus
    # Get cost difference
    w_pm = w_plus-w_minus
    return([A, w_plus, w_minus, w_pm])


def compute_objective(n, C, w_minus, w_plus):
    '''
    Computes objective of the VOTE algorithm for a given cluster assignment
    Args:
        n: (int) number of nodes in graph
        C: (list) list of cluster assignments
        w_minus: (matrix) cost of keeping edges
        w_plus: (matrix) cost of cutting edges
    Return:
        A list including the adjacency matrix of the current clique graph 
        and the current value of the objective
    '''
    k = len(C) # number of clusters
    x = np.zeros((n,n)) # clique graph adjacency matrix
    for c in range(k):
        x[np.ix_(np.where(C[c])[0],np.where(C[c])[0])] = 1    
    obj = np.triu(x*w_minus,1).sum()+np.triu((1-x)*w_plus,1).sum()
    return([x,obj])


def VOTE(n, w_minus, w_plus, w_pm, n_runs, seed):
    '''
    Runs several runs of VOTE on the cost difference matrix wpm
    Args:
        n: (int) number of nodes in graph
        w_minus: (matrix) cost of keeping edges
        w_plus: (matrix) cost of cutting edges
        w_pm: (matrix) difference between cost of cutting edges ad cost of keeping edges
        n_runs: (int) number of runs of VOTE (each run starts with a random permutation of the nodes)
        seed: (int) random seed
    Return:
        A list including a dictionary of cluster assignments for the best clustering out of the n_runs
        runs, and the list of objective values for all n_runs runs
    '''
    # Initialize
    np.random.seed(seed)
    objective_vals = []
    min_obj = np.inf

    # VOTE
    for run in tqdm(range(n_runs)):
        time.sleep(.01)
        k = 0 # number of clusters created so far
        C = [] # list of cluster memberships

        for i in np.random.permutation(n):
            quality = np.zeros(k)
            for c in range(k):
                quality[c] = np.sum(w_pm[i,:]*C[c]) # quality of cluster c for node i

            if k>0:
                c_star = np.argmax(quality)
                q_star = quality[c_star]
            else:
                c_star = -1
                q_star = -1

            if q_star>0:
                C[c_star][i] = 1 # add node i to cluster c_star
            else:
                k += 1
                C.append(np.eye(1,n,i).flatten()) # create new cluster for i

        x, objective_curr = compute_objective(n, C, w_minus, w_plus) # current x and objective value
        objective_vals.append(objective_curr) # record objective value for each run
        if objective_curr < min_obj:
            C_vote = C # best clustering so far
            x_vote = x
            min_obj = objective_curr

    # Represent VOTE cluster assignment as a dictionary
    loc = np.where(np.array(C_vote))
    vote_dict = {loc[1][i]: loc[0][i]  for i in range(n)}
    return([vote_dict,objective_vals])


def VOTE_notqdm(n, w_minus, w_plus, w_pm, n_runs, seed):
    '''
    Runs several runs of VOTE on the cost difference matrix wpm
    Args:
        n: (int) number of nodes in graph
        w_minus: (matrix) cost of keeping edges
        w_plus: (matrix) cost of cutting edges
        w_pm: (matrix) difference between cost of cutting edges ad cost of keeping edges
        n_runs: (int) number of runs of VOTE (each run starts with a random permutation of the nodes)
        seed: (int) random seed
    Return:
        A list including a dictionary of cluster assignments for the best clustering out of the n_runs
        runs, and the list of objective values for all n_runs runs
    '''
    # Initialize
    np.random.seed(seed)
    objective_vals = []
    min_obj = np.inf

    # VOTE
    for run in range(n_runs):
        k = 0 # number of clusters created so far
        C = [] # list of cluster memberships

        for i in np.random.permutation(n):
            quality = np.zeros(k)
            for c in range(k):
                quality[c] = np.sum(w_pm[i,:]*C[c]) # quality of cluster c for node i

            if k>0:
                c_star = np.argmax(quality)
                q_star = quality[c_star]
            else:
                c_star = -1
                q_star = -1

            if q_star>0:
                C[c_star][i] = 1 # add node i to cluster c_star
            else:
                k += 1
                C.append(np.eye(1,n,i).flatten()) # create new cluster for i

        x, objective_curr = compute_objective(n, C, w_minus, w_plus) # current x and objective value
        objective_vals.append(objective_curr) # record objective value for each run
        if objective_curr < min_obj:
            C_vote = C # best clustering so far
            x_vote = x
            min_obj = objective_curr

    # Represent VOTE cluster assignment as a dictionary
    loc = np.where(np.array(C_vote))
    vote_dict = {loc[1][i]: loc[0][i]  for i in range(n)}
    return([vote_dict,objective_vals])


# Function to compute objective using dictionary of node:cluster
def compute_objective_from_dict(c_dict, w_minus, w_plus):
    '''
    Function to compute the VOTE objective using dictionary of node:cluster
    Args:
        c_dict: (dict) dictionary with nodes as keys and clusters as values
        w_minus: (matrix) cost of keeping edges
        w_plus: (matrix) cost of cutting edges
    Return:
        The value of the objective function for the input cluster assignment
    '''
    # Number of nodes and number of clusters
    n = len(np.unique(list(c_dict.keys())))
    k = len(np.unique(list(c_dict.values())))
    # 0/1 encoding of the clusters
    c_mat = np.matrix(np.concatenate([np.eye(1,k,c_dict[i]) for i in range(n)],axis=0))
    x = np.array(c_mat*np.transpose(c_mat))
    # Compute objective
    obj = np.triu(x*w_minus,1).sum()+np.triu((1-x)*w_plus,1).sum()
    return(obj)


def BOEM(vote_dict, n, w_minus, w_plus, max_iter):
    '''
    Best One Element Move local search heuristic starting from the VOTE solution
    Args:
        vote_dict: (dict) dictionary of node assignments to clusters from VOTE
        n: (int) number of nodes in graph
        w_minus: (matrix) cost of keeping edges
        w_plus: (matrix) cost of cutting edges
        max_iter: (int) maximum number of iterations of BOEM
    Return:
        An improved dictionary of node assignments to cluster
    '''
    flag = True
    current_dict = vote_dict.copy()
    current_k = len(np.unique(list(vote_dict.values())))
    min_obj = compute_objective_from_dict(current_dict, w_minus, w_plus)
    print('VOTE objective = '+str(min_obj))
    counter = 0
    while flag:

        counter += 1
        best_c_for_node = np.zeros(n)-1
        best_obj_for_node = np.zeros(n)-1
        for i in tqdm(range(n)):
            time.sleep(.01)
            tmp_dict = current_dict.copy()
            # Move i to new cluster
            obj_move_i = np.zeros(current_k)
            for c in range(current_k):
                if c != current_dict[i]:
                    tmp_dict[i] = c
                else:
                    tmp_dict[i] = current_k+1
                # Compute and store new objective value
                obj_move_i[c] = compute_objective_from_dict(tmp_dict, w_minus, w_plus)
            # Identify best move for i
            best_c_for_node[i] = np.argmin(obj_move_i)
            best_obj_for_node[i] = obj_move_i[int(best_c_for_node[i])]

        # Identify and perform BOEM
        i_star = np.argmin(best_obj_for_node)
        c_star = best_c_for_node[i_star]
        previous_dict = current_dict.copy()
        if c_star != current_dict[i_star]:
            current_dict[i_star] = int(c_star)
        else:
            current_dict[i_star] = current_k+1
            current_k += 1

        # Update flag
        new_obj = compute_objective_from_dict(current_dict, w_minus, w_plus)
        if (new_obj < min_obj) and (counter <= max_iter):
            min_obj = new_obj
            print('VOTE+BOEM '+str(counter)+' objective = '+str(min_obj))
        else:
            flag = False
    best_dict = previous_dict
    return(best_dict)


def BOEM_notqdm(vote_dict, n, w_minus, w_plus, max_iter):
    '''
    Best One Element Move local search heuristic starting from the VOTE solution
    Args:
        vote_dict: (dict) dictionary of node assignments to clusters from VOTE
        n: (int) number of nodes in graph
        w_minus: (matrix) cost of keeping edges
        w_plus: (matrix) cost of cutting edges
        max_iter: (int) maximum number of iterations of BOEM
    Return:
        An improved dictionary of node assignments to cluster
    '''
    flag = True
    current_dict = vote_dict.copy()
    current_k = len(np.unique(list(vote_dict.values())))
    min_obj = compute_objective_from_dict(current_dict, w_minus, w_plus)
    counter = 0
    while flag:

        counter += 1
        best_c_for_node = np.zeros(n)-1
        best_obj_for_node = np.zeros(n)-1
        for i in range(n):
            tmp_dict = current_dict.copy()
            # Move i to new cluster
            obj_move_i = np.zeros(current_k)
            for c in range(current_k):
                if c != current_dict[i]:
                    tmp_dict[i] = c
                else:
                    tmp_dict[i] = current_k+1
                # Compute and store new objective value
                obj_move_i[c] = compute_objective_from_dict(tmp_dict, w_minus, w_plus)
            # Identify best move for i
            best_c_for_node[i] = np.argmin(obj_move_i)
            best_obj_for_node[i] = obj_move_i[int(best_c_for_node[i])]

        # Identify and perform BOEM
        i_star = np.argmin(best_obj_for_node)
        c_star = best_c_for_node[i_star]
        previous_dict = current_dict.copy()
        if c_star != current_dict[i_star]:
            current_dict[i_star] = int(c_star)
        else:
            current_dict[i_star] = current_k+1
            current_k += 1

        # Update flag
        new_obj = compute_objective_from_dict(current_dict, w_minus, w_plus)
        if (new_obj < min_obj) and (counter <= max_iter):
            min_obj = new_obj
        else:
            flag = False
    best_dict = previous_dict
    return(best_dict)


def weighted_correlation_clustering(adhesome_interX_edge_list, 
                                    adhesome_intraX_edge_list,
                                    active_adhesome_genes,
                                    hic_threshold,
                                    with_intra,
                                    weights):
    '''
    Performs weighted correlation clustering on a given graph
    Args:
        adhesome_interX_edge_list: (pandas DataFrame) interchromosomal edge list
        adhesome_intraX_edge_list: (pandas DataFrame) intrachromosomal edge list
        active_adhesome_genes: (Numpy array) array of active adhesome genes
        hic_threshold: (float) quantile to filter the edge lists
        with_intra: (Boolean) whether to include intraX edges in the final edge list
        weights: (str) edge attribute to be used as weights
    Returns:
        A clustering of the nodes (Numpy array) in alphabetical order
    '''
    
    # Construct network
    t = np.quantile(adhesome_interX_edge_list['scaled_hic'],hic_threshold)
    u = np.quantile(adhesome_intraX_edge_list['scaled_hic'],hic_threshold)
    inter_selected = adhesome_interX_edge_list[adhesome_interX_edge_list['scaled_hic']>t]
    intra_selected = adhesome_intraX_edge_list[adhesome_intraX_edge_list['scaled_hic']>u][['source','target','hic','scaled_hic','spearman_corr']]
    adhesome_edge_list = inter_selected
    if with_intra == True:
        adhesome_edge_list = pd.concat([adhesome_edge_list,intra_selected])        
    G = nx.from_pandas_edgelist(adhesome_edge_list, edge_attr=['hic','scaled_hic','spearman_corr'])
    G.add_nodes_from(active_adhesome_genes)
    
    # Get cost matrices
    A, w_plus, w_minus, w_pm = compute_costs(G, weights)
    n = G.number_of_nodes()
    
    # Run VOTE
    vote_dict,objective_vals = VOTE_notqdm(n, w_minus, w_plus, w_pm, n_runs=100, seed=13)    
    # Run BOEM
    boem_dict = BOEM_notqdm(vote_dict, G.number_of_nodes(), w_minus, w_plus, max_iter=100)
    #boem_dict = vote_dict
    best_dict = boem_dict

    # Final clustering
    clustering = [int(best_dict[i]) for i in range(n)]
    return(clustering)


def weighted_correlation_clustering_micha(adhesome_interX_edge_list, 
                                          adhesome_intraX_edge_list,
                                          active_adhesome_genes,
                                          hic_threshold,
                                          with_intra,
                                          weights,
                                          num_calls=100,
                                          seed=13):
    '''
    Performs weighted correlation clustering on a given graph
    Args:
        adhesome_interX_edge_list: (pandas DataFrame) interchromosomal edge list
        adhesome_intraX_edge_list: (pandas DataFrame) intrachromosomal edge list
        active_adhesome_genes: (Numpy array) array of active adhesome genes
        hic_threshold: (float) quantile to filter the edge lists
        with_intra: (Boolean) whether to include intraX edges in the final edge list
        weights: (str) edge attribute to be used as weights
        num_calls: (int) number of iterations (keep the best final objective value)
        seed: (int) the seed for replication
    Returns:
        A clustering of the nodes (Numpy array) in alphabetical order
    '''
    
    # Construct network
    t = np.quantile(adhesome_interX_edge_list['scaled_hic'],hic_threshold)
    u = np.quantile(adhesome_intraX_edge_list['scaled_hic'],hic_threshold)
    inter_selected = adhesome_interX_edge_list[adhesome_interX_edge_list['scaled_hic']>t]
    intra_selected = adhesome_intraX_edge_list[adhesome_intraX_edge_list['scaled_hic']>u][['source','target','hic','scaled_hic','spearman_corr']]
    adhesome_edge_list = inter_selected
    if with_intra == True:
        adhesome_edge_list = pd.concat([adhesome_edge_list,intra_selected])        
    G = nx.from_pandas_edgelist(adhesome_edge_list, edge_attr=['hic','scaled_hic','spearman_corr'])
    G.add_nodes_from(active_adhesome_genes)
    num_nodes = G.number_of_nodes()
    
    # Construct network adjacency matrix (nodes are ordered alphabetically)
    corr_dok = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes), weight='spearman_corr', format = 'dok')
    # Store adjacency matrix as a dictionary
    keys = corr_dok.keys()
    values = corr_dok.values()
    d = dict(zip(keys,values))

    # Write the dictionary to temporary file
    filename = 'tmp_wcc_adjacency.txt'
    outMat = open(filename, 'w')
    outMat.write("%s\n" %num_nodes)
    for ii in range(num_nodes):
        for jj in range(ii + 1, num_nodes):
            if (jj, ii) in d:
                outMat.write("%s\n" %d[(jj, ii)])
            else:
                outMat.write("%s\n" %.0)
    
    # Run VOTE/BOEM
    best_boem_objective = np.inf
    np.random.seed(seed)
    seeds = np.random.permutation(100*num_calls)[0:num_calls]
    for k in range(num_calls):
        outname = 'tmp_wcc_clustering.txt'
        result = subprocess.run("~/correlation-distr2/bin64/chainedSolvers vote boem stats print " + filename + " > " + outname +" "+str(seeds[k]), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
#         result = subprocess.run("~/correlation-distr/bin64/chainedSolvers vote boem stats print " + filename + " > " + outname, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        shell_output = result.stderr.split(' ')
        vote_objective = float(shell_output[9])
        boem_objective = float(shell_output[26])
        num_clusters = int(shell_output[102])
        # Read in the results
        clustering = np.loadtxt(outname, dtype = 'int')
        subprocess.call("rm " + outname, shell=True)
        if (boem_objective < best_boem_objective):
            best_clustering = clustering
            best_vote_objective = vote_objective
            best_boem_objective = boem_objective
            best_num_clusters = num_clusters
    subprocess.call("rm " + filename, shell=True)
        
    return([best_clustering, best_vote_objective, best_boem_objective, best_num_clusters])


def hierarchical_clustering(adhesome_interX_edge_list, 
                            adhesome_intraX_edge_list,
                            active_adhesome_genes,
                            hic_threshold,
                            with_intra,
                            weights,
                            hc_threshold,
                            plot=False):
    '''
    Performs weighted correlation clustering on a given graph
    Args:
        adhesome_interX_edge_list: (pandas DataFrame) interchromosomal edge list
        adhesome_intraX_edge_list: (pandas DataFrame) intrachromosomal edge list
        active_adhesome_genes: (Numpy array) array of active adhesome genes
        hic_threshold: (float) quantile to filter the edge lists
        with_intra: (Boolean) whether to include intraX edges in the final edge list
        weights: (str) edge attribute to be used as weights
        plot: (Boolean) whether to plot the dendrogram
    Returns:
        A clustering of the nodes (Numpy array) in alphabetical order
    '''
    
    # Construct network
    t = np.quantile(adhesome_interX_edge_list['scaled_hic'],hic_threshold)
    u = np.quantile(adhesome_intraX_edge_list['scaled_hic'],hic_threshold)
    inter_selected = adhesome_interX_edge_list[adhesome_interX_edge_list['scaled_hic']>t]
    intra_selected = adhesome_intraX_edge_list[adhesome_intraX_edge_list['scaled_hic']>u][['source','target','hic','scaled_hic','spearman_corr']]
    adhesome_edge_list = inter_selected
    if with_intra == True:
        adhesome_edge_list = pd.concat([adhesome_edge_list,intra_selected])        
    G = nx.from_pandas_edgelist(adhesome_edge_list, edge_attr=['hic','scaled_hic','spearman_corr'])
    G.add_nodes_from(active_adhesome_genes)

    # Get adjacency matrix of G
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='spearman_corr')
    A = np.array(A.todense())
    A = np.exp(-A)
    # Select upper triangular entries and flatten
    y = A[np.triu_indices(n=len(A),k=1)]
    
    # Hierarchical clustering
    linked = linkage(y, method='complete')
    
    # Record flat clustering
    clustering = fcluster(linked, t=hc_threshold, criterion='distance')
    
    # Plot dendrogram
    if plot==True:
        plt.figure(figsize=(10, 40))
        dendrogram(linked,
                    orientation='left',
                    labels=sorted(G.nodes),
                    distance_sort='descending',
                    show_leaf_counts=True,
                    leaf_font_size=10,
                    color_threshold=hc_threshold)
        plt.show()
    
    return(clustering)



def compare_clusterings(clustering1, clustering2, node_names, matching_thresh=0.4):
    '''
    Function to compare two different clusterings on the same set of genes
    Args:
        clustering1: (Numpy array) the first clustering (first cluster indexed by 1)
        clustering2: (Numpy array) the second clustering (first cluster indexed by 1)
        node_names: (Numpy array) the list of node names corresponding to the indices of the clustering arrays
        matching_thresh: (float) the minimum proportion of common nodes required to output a pair fo clusters as matching
    Returns:
        A pandas DataFrame with the list of matching clusters between clustering1 and clustering2
    '''
    # Compute overlap matrix
    overlap = pd.DataFrame(0,index=np.unique(clustering1), columns=np.unique(clustering2))
    for i in overlap.index:
        for j in overlap.columns:
            cluster1 = set(node_names[np.where(clustering1==i)[0]])
            cluster2 = set(node_names[np.where(clustering2==j)[0]])
            if (len(cluster1)==1 or len(cluster2)==1):
                overlap.loc[i,j] = 0
            else:
                overlap.loc[i,j] = len(cluster1.intersection(cluster2))/len(cluster1.union(cluster2))

    # Matching clusters
    clustering1_ids = np.where(overlap>matching_thresh)[0]+1
    clustering1_clusters = [node_names[np.where(clustering1==c)[0]] for c in clustering1_ids]
    n_clustering1_clusters = [len(node_names[np.where(clustering1==c)[0]]) for c in clustering1_ids]
    clustering2_ids = np.where(overlap>matching_thresh)[1]+1
    clustering2_clusters = [node_names[np.where(clustering2==c)[0]] for c in clustering2_ids]
    n_clustering2_clusters = [len(node_names[np.where(clustering2==c)[0]]) for c in clustering2_ids]
    match_prop = [overlap.loc[np.where(overlap>matching_thresh)[0][i]+1,np.where(overlap>matching_thresh)[1][i]+1] 
                  for i in range(len(np.where(overlap>matching_thresh)[0]))]
    matching_clusters = pd.DataFrame({ 'clustering1_cluster_id': clustering1_ids,
                                       'clustering2_cluster_id': clustering2_ids,
                                       'clustering1_cluster': clustering1_clusters,                                   
                                       'clustering2_cluster': clustering2_clusters,
                                       'n_clustering1_clusters': n_clustering1_clusters,
                                       'n_clustering2_clusters': n_clustering2_clusters,
                                       'match': match_prop})
    matching_clusters = matching_clusters.sort_values(by=['match','n_clustering1_clusters','n_clustering2_clusters'], 
                                                            ascending=False)

    # Describe matching clusters
    print('Number of clustering1 clusters = '+str(overlap.shape[0]))
    print('Number of clustering2 clusters = '+str(overlap.shape[1]))
    print('Number of matching clusters = '+str(len(matching_clusters)))
    print('Number of exactly matching clusters = '+str(len(matching_clusters[matching_clusters['match']==1])))
    print('Number of genes in matching clustering1 clusters = '+str(matching_clusters['n_clustering1_clusters'].sum()))
    print('Number of genes in matching clustering2 clusters = '+str(matching_clusters['n_clustering2_clusters'].sum()))
    return(matching_clusters)


def compare_clusterings_with_tfs(clustering1, clustering2, node_names1, node_names2, matching_thresh=0.4):
    '''
    Function to compare two different clusterings on the same set of genes
    Args:
        clustering1: (Numpy array) the first clustering, which has the smallest number of nodes (first cluster indexed by 1)
        clustering2: (Numpy array) the second clustering (first cluster indexed by 1)
        node_names1: (Numpy array) the list of node names corresponding to the indices of clustering1
        node_names2: (Numpy array) the list of node names corresponding to the indices of clustering2
        matching_thresh: (float) the minimum proportion of common nodes required to output a pair fo clusters as matching
    Returns:
        A pandas DataFrame with the list of matching clusters between clustering1 and clustering2
    '''
    # Compute overlap matrix
    overlap = pd.DataFrame(0,index=np.unique(clustering1), columns=np.unique(clustering2))
    for i in overlap.index:
        for j in overlap.columns:
            cluster1 = set(node_names1[np.where(clustering1==i)[0]])
            cluster2 = set(node_names2[np.where(clustering2==j)[0]])
            if (len(cluster1)==1 or len(cluster2)==1):
                overlap.loc[i,j] = 0
            else:
                overlap.loc[i,j] = len(cluster1.intersection(cluster2))/len(cluster1)

    # Matching clusters
    clustering1_ids = np.where(overlap>matching_thresh)[0]+1
    clustering1_clusters = [node_names1[np.where(clustering1==c)[0]] for c in clustering1_ids]
    n_clustering1_clusters = [len(node_names1[np.where(clustering1==c)[0]]) for c in clustering1_ids]
    clustering2_ids = np.where(overlap>matching_thresh)[1]+1
    clustering2_clusters = [node_names2[np.where(clustering2==c)[0]] for c in clustering2_ids]
    n_clustering2_clusters = [len(node_names2[np.where(clustering2==c)[0]]) for c in clustering2_ids]
    match_prop = [overlap.loc[np.where(overlap>matching_thresh)[0][i]+1,np.where(overlap>matching_thresh)[1][i]+1] 
                  for i in range(len(np.where(overlap>matching_thresh)[0]))]
    matching_clusters = pd.DataFrame({ 'clustering1_cluster_id': clustering1_ids,
                                       'clustering2_cluster_id': clustering2_ids,
                                       'clustering1_cluster': clustering1_clusters,                                   
                                       'clustering2_cluster': clustering2_clusters,
                                       'n_clustering1_clusters': n_clustering1_clusters,
                                       'n_clustering2_clusters': n_clustering2_clusters,
                                       'match': match_prop})
    matching_clusters = matching_clusters.sort_values(by=['match','n_clustering1_clusters','n_clustering2_clusters'], 
                                                            ascending=False)

    # Describe matching clusters
    print('Number of clustering1 clusters = '+str(overlap.shape[0]))
    print('Number of clustering2 clusters = '+str(overlap.shape[1]))
    print('Number of matching clusters = '+str(len(matching_clusters)))
    print('Number of exactly matching clusters = '+str(len(matching_clusters[matching_clusters['match']==1])))
    print('Number of genes in matching clustering1 clusters = '+str(matching_clusters['n_clustering1_clusters'].sum()))
    print('Number of genes in matching clustering2 clusters = '+str(matching_clusters['n_clustering2_clusters'].sum()))
    return(matching_clusters)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





