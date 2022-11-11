# Import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from importlib import reload
import numpy as np
from numpy.linalg import multi_dot
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet, inconsistent
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, silhouette_samples
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
import csv
import pandas as pd
import networkx as nx
import community
import communities as com
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score
from networkx.algorithms.community.kclique import k_clique_communities
import pybedtools
import time
from tqdm import tqdm
import random
import OmicsIntegrator as oi
import gseapy
from gseapy.plot import barplot, dotplot
from ortools.linear_solver import pywraplp
from pyvis.network import Network
from scipy.spatial.distance import pdist, squareform
from umap import UMAP
# Custom libraries
import utils as lu
import correlation_clustering as cc
# Reload modules in case of modifications
reload(lu)
reload(cc)

############################################################################
# Directories and info
############################################################################

# Relevant information
cell_type = 'IMR90'
resol_str = '250kb'
resol = 250000
quality = 'MAPQGE30'
norm = 'INTERKR'

# Directory of genome data
dir_genome = '/home/louiscam/projects/gpcr/data/genome_data/'
# Directory of adhesome data
dir_adhesome = '/home/louiscam/projects/gpcr/data/adhesome_data/'
# Directory of processed HiC
dir_processed_hic = f'/home/louiscam/projects/gpcr/save/processed_hic_data/processed_hic_data_IMR90/final_BP250000_intraKR_inter{norm}/'
# Directory of epigenomic data
epigenome_dir = '/home/louiscam/projects/gpcr/data/regulatory_data/regulatory_data_IMR90/'
processed_epigenome_data_dir = '/home/louiscam/projects/gpcr/save/processed_regulatory_marks/processed_epigenome_data_IMR90/'
# Directpry of TF target data
dir_htftarget = '/home/louiscam/projects/gpcr/data/tf_data/hTFtarget/'
# Saving directory
saving_dir = '/home/louiscam/projects/gpcr/save/figures/'

# Load gene location in hg19
gene_locations_filename = dir_genome+'chrom_hg19.loc_canonical'
gene_id_filename = dir_genome+'chrom_hg19.name'
df_loc = lu.get_all_gene_locations(gene_locations_filename, gene_id_filename)

# Active/inactive genes
with open(saving_dir+'active_genes.pkl', 'rb') as f:
    all_active_genes = pickle.load(f)
with open(saving_dir+'inactive_genes.pkl', 'rb') as f:
    all_inactive_genes = pickle.load(f)

# Mapping gene to chromosome
with open(saving_dir+'gene2chrom.pkl', 'rb') as f:
    gene2chrom = pickle.load(f)

# Adhesome genes and loci
with open(saving_dir+'active_adhesome_genes.pkl', 'rb') as f:
    active_adhesome_genes = pickle.load(f)
with open(saving_dir+'active_adhesome_genes_loci.pkl', 'rb') as f:
    active_adhesome_loci = pickle.load(f)
with open(saving_dir+'adhesome_chr_loci.pkl', 'rb') as f:
    adhesome_chr_loci = pickle.load(f)
    
# Adhesome TF genes and loci
with open(saving_dir+'active_lung_adhesome_tf_genes.pkl', 'rb') as f:
    active_lung_adhesome_tf_genes = pickle.load(f)
with open(saving_dir+'active_lung_adhesome_tf_loci.pkl', 'rb') as f:
    active_lung_adhesome_tf_loci = pickle.load(f)

# Selected genes = Adhesome genes + adhesome TFs
selected_genes = np.unique(np.concatenate([active_adhesome_genes,active_lung_adhesome_tf_genes], axis=0))
selected_loci = np.unique(np.concatenate([active_adhesome_loci,active_lung_adhesome_tf_loci], axis=0))
with open(saving_dir+'adh_and_tf_gene2locus.pkl', 'rb') as f:
    selected_gene2locus = pickle.load(f)
with open(saving_dir+'adh_and_tf_locus2gene.pkl', 'rb') as f:
    selected_locus2gene = pickle.load(f)
with open(saving_dir+'adh_and_tf_chr_loci.pkl', 'rb') as f:
    adh_and_tf_chr_loci = pickle.load(f)
with open(saving_dir+'contacts_df.pkl', 'rb') as f:
    contacts_df = pickle.load(f)
    
# Load hTFtarget data set
with open(saving_dir+'active_lung_tf2target.pkl', 'rb') as f:
    active_lung_tf2target = pickle.load(f)
with open(saving_dir+'active_lung_target2tf.pkl', 'rb') as f:
    active_lung_target2tf = pickle.load(f)

############################################################################
# Load and process edge lists
############################################################################

# Build interX edge list
adhesome_interX_edge_list = cc.create_interX_edgelist(contacts_df, selected_loci, selected_locus2gene, gene2chrom)
with open(saving_dir+f'adhesome_interX_edge_list_{norm}.pkl', 'wb') as f:
    pickle.dump(adhesome_interX_edge_list, f)
# Build intraX edge list
adhesome_intraX_edge_list = cc.create_intraX_edgelist(selected_loci, selected_locus2gene, df_loc, resol, dir_processed_hic)
with open(saving_dir+f'adhesome_intraX_edge_list_{norm}.pkl', 'wb') as f:
    pickle.dump(adhesome_intraX_edge_list, f)

# Load edge lists
with open(saving_dir+f'adhesome_interX_edge_list_{norm}.pkl', 'rb') as f:
    adhesome_interX_edge_list = pickle.load(f)
with open(saving_dir+f'adhesome_intraX_edge_list_{norm}.pkl', 'rb') as f:
    adhesome_intraX_edge_list = pickle.load(f)
    
# Load epigenomic features
df_all_norm = pd.read_csv(saving_dir+'features_matrix_all_genes_norm.csv', header=0, index_col=0)
# Restrict to genes in cluster
df_cluster_norm = df_all_norm.loc[:,selected_genes]

# Create Pearson distance matrix
adhesome_pearson_distance_df = (1-df_cluster_norm.corr(method='pearson'))/2
# Create Spearman distance matrix
adhesome_spearman_distance_df = (1-df_cluster_norm.corr(method='spearman'))/2
# Create cosine distance matrix
adhesome_cosine_df = pd.DataFrame(
    squareform(pdist(df_cluster_norm.T, metric='cosine')),
    columns = df_cluster_norm.columns,
    index = df_cluster_norm.columns
)/2

# Add distance and Spearman correlation to interX edge list
adhesome_interX_edge_list['pearson_dist'] = [adhesome_pearson_distance_df.loc[adhesome_interX_edge_list.iloc[i]['source'],
                                                                              adhesome_interX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_interX_edge_list))]
adhesome_interX_edge_list['spearman_dist'] = [adhesome_spearman_distance_df.loc[adhesome_interX_edge_list.iloc[i]['source'],
                                                                                adhesome_interX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_interX_edge_list))]
adhesome_interX_edge_list['cosine_dist'] = [adhesome_cosine_df.loc[adhesome_interX_edge_list.iloc[i]['source'],
                                                                                adhesome_interX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_interX_edge_list))]
# Add distance and Spearman correlation to intraX edge list
adhesome_intraX_edge_list['pearson_dist'] = [adhesome_pearson_distance_df.loc[adhesome_intraX_edge_list.iloc[i]['source'],
                                                                              adhesome_intraX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_intraX_edge_list))]
adhesome_intraX_edge_list['spearman_dist'] = [adhesome_spearman_distance_df.loc[adhesome_intraX_edge_list.iloc[i]['source'],
                                                                                adhesome_intraX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_intraX_edge_list))]
adhesome_intraX_edge_list['cosine_dist'] = [adhesome_cosine_df.loc[adhesome_intraX_edge_list.iloc[i]['source'],
                                                                                adhesome_intraX_edge_list.iloc[i]['target']]
                                              for i in range(len(adhesome_intraX_edge_list))]


# Derive a null distribution for regulatory strength of clusters
null_dist_dict = {n: [] for n in np.arange(2,8)}
np.random.seed(13)
for _ in tqdm(range(1000)):
    for n_genes in np.arange(2,8):
        random_genes = np.random.choice(all_active_genes, size=n_genes, replace=False)
        df_random = df_all_norm.loc[:,random_genes]
        df_random_corr = (1+df_random.corr(method='pearson'))/2
        null_dist_dict[n_genes].append(np.mean(df_random_corr.values[np.triu_indices(n=len(random_genes), k=1)]))
null_dist_dict = {key: np.array(val) for key, val in null_dist_dict.items()}

############################################################################
# Perform clustering with different parameters
############################################################################

params_ls = [['hic', 'reciprocal', 'average', 'distance'],
             ['hic', 'reciprocal', 'average', 'inconsistent'],
             ['hic', 'dijkstra', 'average', 'distance'],
             ['hic', 'dijkstra', 'average', 'inconsistent'],
             ['regulatory_marks', 'cosine', 'average', 'distance'],
             ['regulatory_marks', 'cosine', 'average', 'inconsistent'],
             ['regulatory_marks', 'Pearson', 'average', 'distance'],
             ['regulatory_marks', 'Pearson', 'average', 'inconsistent']]

for params in tqdm(params_ls):
    
    edge_weights, distance, linkage_fn, criterion = params
    
    # Combine interX and intraX edge lists
    inter_selected = adhesome_interX_edge_list.copy()
    inter_selected = inter_selected[inter_selected['scaled_hic']>0]
    intra_selected = adhesome_intraX_edge_list.copy()
    intra_selected = intra_selected[['source','target','hic','scaled_hic',
                                     'pearson_dist','spearman_dist',
                                     'cosine_dist']]
    intra_selected = intra_selected[intra_selected['scaled_hic']>0]
    adhesome_edge_list = pd.concat([inter_selected,intra_selected], axis=0)
    adhesome_edge_list = adhesome_edge_list[adhesome_edge_list['source'] != adhesome_edge_list['target']]
    
    # Create additional similarity measures/distances
    adhesome_edge_list['hic_dist'] = 1-adhesome_edge_list['scaled_hic']
    adhesome_edge_list['log_hic'] = -np.log(adhesome_edge_list['scaled_hic'])
    
    # Create network
    G = nx.from_pandas_edgelist(adhesome_edge_list, edge_attr=['hic','scaled_hic', 'log_hic', 'hic_dist',
                                                               'pearson_dist','spearman_dist','cosine_dist'])

    # Compute hic_threshold (only used for edge_weights = 'regulatory_marks')
    hic_threshold = np.quantile(adhesome_edge_list['scaled_hic'].values, 0.75)
    
    # Select distance
    if edge_weights == 'hic':

        if distance == 'reciprocal':
            dist_hic = 1-np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
            np.fill_diagonal(dist_hic,0)

        elif distance == 'dijsktra':
            len_path = dict(nx.all_pairs_dijkstra(G, weight='log_hic'))
            df_list = [pd.DataFrame(
                {'genes': sorted(G.nodes),
                 g: [np.exp(-len_path[g][0][g1]) for g1 in sorted(G.nodes)]}
                                   ).set_index('genes')
                       for g in sorted(G.nodes)]
            dist_hic = 1-pd.concat(df_list, axis=1)
            dist_hic = dist_hic.values
            np.fill_diagonal(dist_hic,0)

        elif distance == 'custom':
            all_hic_values = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
            all_hic_values = all_hic_values[np.tril_indices(len(all_hic_values),1)]
            all_nonzero_hic_values = all_hic_values[np.nonzero(all_hic_values)]
            min_nonzero_hic = np.min(all_nonzero_hic_values)
            c = min_nonzero_hic/10
            sim_hic = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
            dist_hic = np.log10((1+c)/(sim_hic+c))
            np.fill_diagonal(dist_hic,0)

    elif edge_weights == 'regulatory_marks':

        if distance == 'Pearson':
            adj_mat_hic = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
            hic_mask = adj_mat_hic>hic_threshold
            dist_mat_reg = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='pearson_dist').todense())
            dist_hic = dist_mat_reg*hic_mask+(1-hic_mask)
            np.fill_diagonal(dist_hic,0)

        elif distance == 'Spearman':
            adj_mat_hic = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
            hic_mask = adj_mat_hic>hic_threshold
            dist_mat_reg = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='spearman_dist').todense())
            dist_hic = dist_mat_reg*hic_mask+(1-hic_mask)
            np.fill_diagonal(dist_hic,0)

        elif distance == 'cosine':
            adj_mat_hic = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
            hic_mask = adj_mat_hic>hic_threshold
            dist_mat_reg = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='cosine_dist').todense())
            dist_hic = dist_mat_reg*hic_mask+(1-hic_mask)
            np.fill_diagonal(dist_hic,0)
    
    # Create flattened distance matrix
    y = dist_hic[np.triu_indices(n=len(dist_hic),k=1)]
    
    # Construct dendrogram with selected linkage
    linked = linkage(y, method=linkage_fn)
    
    # Perform clustering
    if criterion == 'distance':
        t = cc.get_4chrom_clustering(linked, sorted(G.nodes), gene2chrom, 
                                  inc=0.01, max_chrom=4, criterion=criterion)
        print(f'Chosen cophenetic distance threshold = {t}')
        clust = fcluster(linked, t, criterion=criterion)

    elif criterion == 'inconsistent':
        t = cc.get_4chrom_clustering(linked, sorted(G.nodes), gene2chrom, 
                                  inc=0.01, max_chrom=4, criterion=criterion)
        print(f'Chosen inconsistency coefficient threshold = {t}')
        clust = fcluster(linked, t, criterion=criterion)
    
    if edge_weights == 'regulatory_marks':
        
        # Identify which nodes are in non-singleton clusters
        gene_clusters_df = pd.DataFrame({'gene': sorted(G.nodes), 'cluster': clust})
        nonsingleton_cluster_ids = set([c for c in list(clust) if list(clust).count(c)>1])
        nonsingleton_genes = gene_clusters_df[gene_clusters_df['cluster'].isin(nonsingleton_cluster_ids)]['gene'].values

        # Print out clusters ordered by size
        clusters_df = gene_clusters_df.groupby('cluster')['gene'].agg(list).to_frame()
        clusters_df.columns = ['genes']
        clusters_df['size'] = [len(c) for c in clusters_df['genes']]
        clusters_df = clusters_df.sort_values('size', ascending=False)
        
        # Save coclustering matrix
        coclust_df = pd.DataFrame(0, index=sorted(G.nodes), columns=sorted(G.nodes))
        for c in clusters_df['genes']:
            coclust_df.loc[c,c] = 1
        np.fill_diagonal(coclust_df.values, 0)
        with open(saving_dir+f'coclust_df_{edge_weights}_{distance}_{linkage_fn}_{criterion}.pkl', 'wb') as f:
            pickle.dump(coclust_df, f)
    
    elif edge_weights == 'hic':
        
        # Get clustering dataframe
        clustering_hic_df = pd.DataFrame({'gene': sorted(G.nodes),
                                          'chrom': [gene2chrom[g] for g in sorted(G.nodes)],
                                          'cluster': clust})
        clustering_hic_df = clustering_hic_df.groupby('cluster').agg({'gene': lambda x: list(x),
                                                                      'chrom': lambda x: list(x)})
        clustering_hic_df['unique_chrom'] = [set(ls) for ls in clustering_hic_df['chrom']]
        clustering_hic_df['size'] = [len(ls) for ls in clustering_hic_df['gene']] 

        # Add adhesome, TF annotations and specificity
        clustering_hic_df['adhesome'] = [[g for g in clustering_hic_df.iloc[i]['gene'] if g in active_adhesome_genes] 
                                    for i in range(len(clustering_hic_df))]
        clustering_hic_df['TFs'] = [[g for g in clustering_hic_df.iloc[i]['gene'] if g in active_lung_adhesome_tf_genes] 
                                    for i in range(len(clustering_hic_df))]
        clustering_hic_df['prop_adhesome_targets'] =[[np.round(len(active_lung_tf2target[tf].intersection(set(active_adhesome_genes)))/len(active_adhesome_genes),3) 
                                                      for tf in tf_list]
                                                      for tf_list in clustering_hic_df['TFs']]
        clustering_hic_df['prop_genome_targets'] =[[np.round(len(active_lung_tf2target[tf].intersection(set(all_active_genes)))/len(all_active_genes),3) 
                                                    for tf in tf_list]
                                                    for tf_list in clustering_hic_df['TFs']]

        # Only select non singleton clusters
        clustering_hic_df = clustering_hic_df[clustering_hic_df['size']>1]
        clustering_hic_df = clustering_hic_df.sort_values(by='size', ascending=False)
        clustering_hic_df['TF_adhesome_targets'] = [
            [sorted(active_lung_tf2target[tf].intersection(set(active_adhesome_genes))) 
             for tf in clustering_hic_df.loc[c,'TFs']
            ] 
            for c in clustering_hic_df.index]

        # Obtain adjacency matrix based on scaled_hic
        A = np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='scaled_hic').todense())
        np.fill_diagonal(A,1)
        A = pd.DataFrame(A, index=sorted(G.nodes), columns=sorted(G.nodes))
        # Compute HiC strength of each cluster
        hic_strength = []
        for c in clustering_hic_df.index:
            genes = clustering_hic_df.loc[c,'gene']
            hic_strength.append(np.mean(A.loc[genes,genes].values[np.triu_indices(n=len(genes), k=1)]))
        clustering_hic_df['hic_strength'] = hic_strength

        # Obtain adjacency matrix based on regulatory features
        A = 1-np.array(nx.adjacency_matrix(G, nodelist=sorted(G.nodes), weight='pearson_dist').todense())
        np.fill_diagonal(A,1)
        A = pd.DataFrame(A, index=sorted(G.nodes), columns=sorted(G.nodes))
        # Compute regulatory strength of each cluster
        reg_strength = []
        for c in clustering_hic_df.index:
            genes = clustering_hic_df.loc[c,'gene']
            reg_strength.append(np.mean(A.loc[genes,genes].values[np.triu_indices(n=len(genes), k=1)]))
        clustering_hic_df['reg_strength'] = reg_strength

        # Compute regulatory p-value for each cluster
        clustering_hic_df['pval'] = [
            np.mean(null_dist_dict[clustering_hic_df.iloc[i]['size']]>clustering_hic_df.iloc[i]['reg_strength']) 
                                     for i in range(len(clustering_hic_df))
        ]
        clustering_hic_df['significant cluster'] = clustering_hic_df['pval']<0.1
        clustering_hic_df = clustering_hic_df.sort_values(by='pval', ascending=True)
        clustering_hic_df = clustering_hic_df.reset_index()
        clustering_hic_df = clustering_hic_df[clustering_hic_df['significant cluster']]

        # Save coclustering matrix
        coclust_df = pd.DataFrame(0, index=sorted(G.nodes), columns=sorted(G.nodes))
        for c in clustering_hic_df['gene']:
            coclust_df.loc[c,c] = 1
        np.fill_diagonal(coclust_df.values, 0)
        with open(saving_dir+f'coclust_df_{edge_weights}_{distance}_{linkage_fn}_{criterion}.pkl', 'wb') as f:
            pickle.dump(coclust_df, f)

    
    
    

