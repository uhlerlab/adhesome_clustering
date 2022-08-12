# Import standard libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
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
import utils as lu


def create_interX_edgelist(contacts_df, selected_loci, locus_gene_dict, gene2chrom):
    '''
    Creates an interX edge list between genes corresponding to a list of loci
    Args:
        contacts_df: (pandas DataFrame) a dataframe containing HiC contacts between loci
        selected_loci: (Numpy array) the set of loci of interest
        locus_gene_dict: (dict) a dictionary mapping each locus to the corresponding genes
        gene2chrom: (dict) mapping each gene to the corresponding chromosome
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
    #selected_interX_edge_list = selected_interX_edge_list[selected_interX_edge_list['hic']>0]
    selected_interX_edge_list['source'] = [locus_gene_dict[selected_interX_edge_list.iloc[i,0]] 
                                           for i in np.arange(len(selected_interX_edge_list))]
    selected_interX_edge_list['target'] = [locus_gene_dict[selected_interX_edge_list.iloc[i,1]] 
                                           for i in np.arange(len(selected_interX_edge_list))]
    selected_interX_edge_list = selected_interX_edge_list[['source','target','hic']]

    # Explode source and target
    selected_interX_edge_list = lu.unnesting(selected_interX_edge_list,['target'])
    selected_interX_edge_list = lu.unnesting(selected_interX_edge_list,['source'])

    # Aggregate interactions between same genes
    selected_interX_edge_list = selected_interX_edge_list.groupby(['source','target'])['hic'].agg('mean').reset_index()

    # Add scaled_hic values
    selected_interX_edge_list['scaled_hic'] = (selected_interX_edge_list['hic']-selected_interX_edge_list['hic'].min())/(selected_interX_edge_list['hic'].max()-selected_interX_edge_list['hic'].min())
    
    # Drop all intraX links
    rows2keep = [i for i in range(len(selected_interX_edge_list)) 
                 if (gene2chrom[selected_interX_edge_list.iloc[i]['source']] != gene2chrom[selected_interX_edge_list.iloc[i]['target']])]
    selected_interX_edge_list = selected_interX_edge_list.iloc[rows2keep]
    
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
    chr_list = np.arange(1,22+1,1)
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
        new_index = pd.MultiIndex.from_tuples(itertools.combinations_with_replacement(subindex,2), names=["locus_source","locus_target"])
        hic_chpair_df1 = hic_chpair_adh_tf_df.stack().reindex(new_index).reset_index(name='hic')

        # Drop duplicate interactions (same interaction partners)
        hic_chpair_df1['id'] = ['_'.join(np.sort(hic_chpair_df1.iloc[i][['locus_source','locus_target']].values)) 
                                           for i in range(len(hic_chpair_df1))]
        hic_chpair_df1 = hic_chpair_df1.sort_values(['id','locus_source','locus_target'])
        hic_chpair_df1 = hic_chpair_df1.drop_duplicates('id')
        hic_chpair_df1 = hic_chpair_df1[['locus_source','locus_target','hic']]

        # Add gene labels to the source and target loci
        #hic_chpair_df1 = hic_chpair_df1[hic_chpair_df1['hic']>0]
        hic_chpair_df1['source'] = [locus_gene_dict[hic_chpair_df1.iloc[i,0]]
                                    for i in np.arange(len(hic_chpair_df1))]
        hic_chpair_df1['target'] = [locus_gene_dict[hic_chpair_df1.iloc[i,1]]
                                    for i in np.arange(len(hic_chpair_df1))]
        hic_chpair_df1 = hic_chpair_df1[['source','target','hic']]
        
        try:
            # Explode source and target
            hic_chpair_df1 = lu.unnesting(hic_chpair_df1,['target'])
            hic_chpair_df1 = lu.unnesting(hic_chpair_df1,['source'])

            # Aggregate interactions between same genes
            hic_chpair_df1 = hic_chpair_df1.groupby(['source','target'])['hic'].agg('mean').reset_index()

            # Add chromosome information
            hic_chpair_df1['chrom'] = chrom

            # Add genomic distance information
            hic_chpair_df1['gen_dist'] = [np.abs(df_loc[df_loc['geneSymbol']==hic_chpair_df1.iloc[i]['source']][['chromStart','chromEnd']].values.mean()/resol -
                                                 df_loc[df_loc['geneSymbol']==hic_chpair_df1.iloc[i]['target']][['chromStart','chromEnd']].values.mean()/resol)
                                         for i in range(len(hic_chpair_df1))]
            
            # Drop duplicates
            hic_chpair_df1['id'] = ['_'.join(np.sort(hic_chpair_df1.iloc[i][['source','target']].values)) 
                                           for i in range(len(hic_chpair_df1))]
            hic_chpair_df1 = hic_chpair_df1.drop_duplicates(subset=['id'], keep='first').iloc[:,:-1]

            # Concatenate to intraX edge list for previous chromosomes
            selected_intraX_edge_list = pd.concat([selected_intraX_edge_list,hic_chpair_df1])
        except AttributeError:
            print('No interaction for chromosome '+str(chrom))

    selected_intraX_edge_list = selected_intraX_edge_list.sort_values(by=['source','target'])

    # Add scaled_hic values
    selected_intraX_edge_list['scaled_hic'] = (selected_intraX_edge_list['hic']-selected_intraX_edge_list['hic'].min())/(selected_intraX_edge_list['hic'].max()-selected_intraX_edge_list['hic'].min())

    # Drop self-interactions
    # selected_intraX_edge_list = selected_intraX_edge_list[selected_intraX_edge_list['source']!=selected_intraX_edge_list['target']]
    return(selected_intraX_edge_list)

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


def hierarchical_clustering2(adhesome_interX_edge_list, 
                            adhesome_intraX_edge_list,
                            active_adhesome_genes,
                            hic_threshold,
                            with_intra,
                            weights,
                            hc_threshold,
                            hc_method,
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
    intra_selected = adhesome_intraX_edge_list[adhesome_intraX_edge_list['scaled_hic']>u][['source','target','hic','scaled_hic','distance']]
    adhesome_edge_list = inter_selected
    if with_intra == True:
        adhesome_edge_list = pd.concat([adhesome_edge_list,intra_selected])
    G = nx.from_pandas_edgelist(adhesome_edge_list, edge_attr=['hic','scaled_hic','distance'])
    G.add_nodes_from(active_adhesome_genes)

    # Get adjacency matrix of G
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='distance')
    A = np.array(A.todense())
    A[A==0] = A.max().max()
    np.fill_diagonal(A,0)
    # Select upper triangular entries and flatten
    y = A[np.triu_indices(n=len(A),k=1)]
    
    # Hierarchical clustering
    linked = linkage(y, method=hc_method)
    
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
        plt.vlines(x=hc_threshold, ymin=0, ymax=4000, color='red', linestyle='dashed')
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
        matching_thresh: (float) the minimum number of common nodes required to output a pair of clusters as matching
    Returns:
        A pandas DataFrame with the list of matching clusters between clustering1 and clustering2
    '''
    # Compute overlap matrix
    overlap_num = pd.DataFrame(0,index=np.unique(clustering1), columns=np.unique(clustering2))
    overlap_prop = pd.DataFrame(0,index=np.unique(clustering1), columns=np.unique(clustering2))
    for i in overlap_num.index:
        for j in overlap_num.columns:
            cluster1 = set(node_names1[np.where(clustering1==i)[0]])
            cluster2 = set(node_names2[np.where(clustering2==j)[0]])
            if (len(cluster1)==1 or len(cluster2)==1):
                overlap_num.loc[i,j] = 0
                overlap_prop.loc[i,j] = 0
            else:
                overlap_num.loc[i,j] = len(cluster1.intersection(cluster2))
                overlap_prop.loc[i,j] = len(cluster1.intersection(cluster2))/len(cluster1)

    # Matching clusters
    clustering1_ids = np.where(overlap_num>=matching_thresh)[0]+1
    clustering1_clusters = [node_names1[np.where(clustering1==c)[0]] for c in clustering1_ids]
    n_clustering1_clusters = [len(node_names1[np.where(clustering1==c)[0]]) for c in clustering1_ids]
    clustering2_ids = np.where(overlap_num>=matching_thresh)[1]+1
    clustering2_clusters = [node_names2[np.where(clustering2==c)[0]] for c in clustering2_ids]
    n_clustering2_clusters = [len(node_names2[np.where(clustering2==c)[0]]) for c in clustering2_ids]
    match_num = [overlap_num.loc[np.where(overlap_num>=matching_thresh)[0][i]+1,np.where(overlap_num>=matching_thresh)[1][i]+1] 
                  for i in range(len(np.where(overlap_num>=matching_thresh)[0]))]
    match_prop = [overlap_prop.loc[np.where(overlap_num>=matching_thresh)[0][i]+1,np.where(overlap_num>=matching_thresh)[1][i]+1] 
                  for i in range(len(np.where(overlap_num>=matching_thresh)[0]))]
    matching_clusters = pd.DataFrame({ 'clustering1_cluster_id': clustering1_ids,
                                       'clustering2_cluster_id': clustering2_ids,
                                       'clustering1_cluster': clustering1_clusters,                                   
                                       'clustering2_cluster': clustering2_clusters,
                                       'n_clustering1_clusters': n_clustering1_clusters,
                                       'n_clustering2_clusters': n_clustering2_clusters,
                                       'match_num': match_num,
                                       'match_prop': match_prop})
    matching_clusters = matching_clusters.sort_values(by=['match_num','match_prop','n_clustering1_clusters','n_clustering2_clusters'], 
                                                            ascending=False)

    # Describe matching clusters
    print('Number of clustering1 clusters = '+str(overlap_num.shape[0]))
    print('Number of clustering2 clusters = '+str(overlap_num.shape[1]))
    print('Number of matching clusters = '+str(len(matching_clusters)))
    print('Number of exactly matching clusters = '+str(len(matching_clusters[matching_clusters['match_prop']==1])))
    print('Number of genes in matching clustering1 clusters = '+str(matching_clusters['n_clustering1_clusters'].sum()))
    print('Number of genes in matching clustering2 clusters = '+str(matching_clusters['n_clustering2_clusters'].sum()))
    return(matching_clusters)

    
def get_4chrom_clustering(dend, genes, gene2chrom, inc=0.01, max_chrom=4, criterion='distance'):
    '''
    Function to determine the threshold to use to obtain a clustering with max 4 chromosomes per cluster
    (and the maximum number of genes)
    Args:
        dend: (Numpy array) the dendrogram
        genes: (Numpy array) genes (in the correct ordering with respect to dend)
        gene2chrom: (dict) dictionary mapping genes to the corresponding chromosome
        inc: (float) the incremental value for testing thresholds
        max_chrom: (int) maximum number of chromosome in a cluster
    Returns:
        A threshold to cut the dendrogram so that the resulting clustering has at most 4 chromosomes per cluster
    '''
    
    if criterion=='distance':
        # Determine height of dendrogram
        h = dend[-1,2]
        thresholds = np.arange(0,h+inc,inc)
        # Find threshold corresponding to max 4 chromosomes per cluster
        for t in thresholds:
            clustering = fcluster(dend, t, criterion=criterion)
            cluster_df = pd.DataFrame({'gene': genes,
                                       'cluster': clustering,
                                       'chrom': [gene2chrom[g] for g in genes]})
            max_chrom_current = cluster_df.groupby('cluster')['chrom'].nunique().max()
            if max_chrom_current >= max_chrom+1:
                break
        return(t-inc)
    
    elif criterion=='inconsistent':
        # Determine maximum inconsistency
        h = np.max(inconsistent(dend)[:,3])
        thresholds = np.arange(0,h+inc,inc)
        # Find threshold corresponding to max 4 chromosomes per cluster
        for t in thresholds:
            clustering = fcluster(dend, t, criterion=criterion)
            cluster_df = pd.DataFrame({'gene': genes,
                                       'cluster': clustering,
                                       'chrom': [gene2chrom[g] for g in genes]})
            max_chrom_current = cluster_df.groupby('cluster')['chrom'].nunique().max()
            if max_chrom_current >= max_chrom+1:
                break
        return(t-inc)
    

def format_node_label(raw_label):
    '''
    Function to put node label in a square format
    Args:
        raw_label: (Numpy array) array of gene names
    Returns:
        The formatted label (a string)
    '''
    label = raw_label.copy()   
    # Line breaks
    for i in range(1, len(label)//2+1):
        if 2*i<len(label):
            label[2*i-1] = label[2*i-1]+' \n '
    label = ' '.join(label)
    return label


def from_nx(graph, pyvisnet, default_node_size=1, 
            default_edge_weight=1, edge_weight_scale=1, edge_color_scale=1, edge_threshold=0, hidden_edges=[], 
            shape='circle'):
    """
    This method takes an exisitng Networkx graph and translates
    it to a PyVis graph format that can be accepted by the VisJs
    API in the Jinja2 template. This operation is done in place.
    """
    nx_graph = graph.copy()
    assert(isinstance(nx_graph, nx.Graph))
    edges = nx_graph.edges(data=True)
    nodes = nx_graph.nodes(data=True)

    if len(edges) > 0:
        for e in edges:
            # Specify node size
            if 'size' not in nodes[e[0]].keys():
                nodes[e[0]]['size'] = default_node_size
            nodes[e[0]]['size'] = int(nodes[e[0]]['size'])
            if 'size' not in nodes[e[1]].keys():
                nodes[e[1]]['size'] = default_node_size
            nodes[e[1]]['size'] = int(nodes[e[1]]['size'])
            # Specify node color
            if nodes[e[0]]['n_TFs']>0:
                nodes[e[0]]['color'] = 'lightcoral'
            else:
                nodes[e[0]]['color'] = 'lightcoral'
            if nodes[e[1]]['n_TFs']>0:
                nodes[e[1]]['color'] = 'lightcoral'
            else:
                nodes[e[1]]['color'] = 'lightcoral'
            # Specify node title
            if len(nodes[e[0]]['all_TFs'])>0:
                nodes[e[0]]['title'] = 'adhesome: '+','.join(nodes[e[0]]['all_adhesomes'])+' / '+'TFs: '+','.join(nodes[e[0]]['all_TFs'])
            else:
                nodes[e[0]]['title'] = 'adhesome: '+','.join(nodes[e[0]]['all_adhesomes'])+' / '+'no TF'
            if nodes[e[1]]['n_TFs']>0:
                nodes[e[1]]['title'] = 'adhesome: '+','.join(nodes[e[1]]['all_adhesomes'])+' / '+'TFs: '+','.join(nodes[e[1]]['all_TFs'])
            else:
                nodes[e[1]]['title'] = 'adhesome: '+','.join(nodes[e[1]]['all_adhesomes'])+' / '+'no TF'
            pyvisnet.add_node(e[0], **nodes[e[0]], shape=shape)
            pyvisnet.add_node(e[1], **nodes[e[1]], shape=shape)

            if 'weight' not in e[2].keys():
                e[2]['weight'] = default_edge_weight
            edge_dict = e[2].copy()
            edge_dict["value"] = e[2]['weight']*edge_weight_scale
            edge_dict["title"] = 'TF: '+e[2]['TFs']+'\n'+'val: '+str(e[2]['weight'])
            edge_dict["label"] = e[2]['TFs']
            edge_dict["color"] = colfunc(e[2]['weight']*edge_color_scale)
            edge_dict["arrowStrikethrough"] = False
            if (e[2]['weight']>edge_threshold) and (e[2]['TFs'] not in hidden_edges):
                pyvisnet.add_edge(e[0], e[1], **edge_dict)

    for node in nx.isolates(nx_graph):
        if 'size' not in nodes[node].keys():
            nodes[node]['size']=default_node_size
            nodes[node]['color'] = 'lightcoral'
        pyvisnet.add_node(node, **nodes[node], shape=shape)
    
    
def colfunc(val, minval=0, maxval=1):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    RED, YELLOW, GREEN  = (1, 0, 0), (1, 1, 0), (0, 1, 0)
    CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
    WHITE = (1, 1, 1)
    f = float(val-minval) / (maxval-minval)
    return mpl.colors.rgb2hex(tuple(f*(b-a)+a for (a, b) in zip(WHITE, RED)))
    
    
def tfnx_from_nx(graph, pyvisnet, 
                 default_node_size=1, default_edge_weight=1, 
                 shape='circle'):
    """
    This method takes an exisitng Networkx graph and translates
    it to a PyVis graph format that can be accepted by the VisJs
    API in the Jinja2 template. This operation is done in place.
    """
    nx_graph = graph.copy()
    assert(isinstance(nx_graph, nx.Graph))
    edges = nx_graph.edges(data=True)
    nodes = nx_graph.nodes(data=True)

    if len(edges) > 0:
        for e in edges:
            # Specify node size
            if 'size' not in nodes[e[0]].keys():
                nodes[e[0]]['size'] = default_node_size
            nodes[e[0]]['size'] = int(nodes[e[0]]['size'])
            if nodes[e[0]]['significant_tf']==1:
                nodes[e[0]]['color'] = 'lightcoral'
            else:
                nodes[e[0]]['color'] = 'dodgerblue'
            if 'size' not in nodes[e[1]].keys():
                nodes[e[1]]['size'] = default_node_size
            nodes[e[1]]['size'] = int(nodes[e[1]]['size'])
            if nodes[e[1]]['significant_tf']==1:
                nodes[e[1]]['color'] = 'lightcoral'
            else:
                nodes[e[1]]['color'] = 'dodgerblue'
            pyvisnet.add_node(e[0], **nodes[e[0]], shape=shape)
            pyvisnet.add_node(e[1], **nodes[e[1]], shape=shape)
            # Specify edge characteristics
            e[2]['weight'] = default_edge_weight
            e[2]['color'] = '#cbcff5'
            if e[0]==e[1]:
                e[2]['weight'] = default_edge_weight
                e[2]['color'] = '#fa0202'
            edge_dict = e[2].copy()
            edge_dict["width"] = e[2]['weight']
            edge_dict["color"] = e[2]['color']
            edge_dict["arrowStrikethrough"] = False
            pyvisnet.add_edge(e[0], e[1], **edge_dict)

    for node in nx.isolates(nx_graph):
        if 'size' not in nodes[node].keys():
            nodes[node]['size']=default_node_size
        if nodes[node]['significant_tf']==1:
            nodes[node]['color'] = 'lightcoral'
        else:
            nodes[node]['color'] = 'dodgerblue'
        pyvisnet.add_node(node, **nodes[node], shape=shape)
    
    
    
    
    
    





