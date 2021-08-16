# Import standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.stats as ss
import csv
import pandas as pd
import networkx as nx
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
import pybedtools


def get_gene_locations(gene_locations_filename, gene_id_filename):
    '''
    Rerieves gene location on the genome (hg19 reference genome) from UCSC tables,
    and matches the name of every UCSC gene to the corresponding HGNC name
    Args:
        gene_locations_filename: (String) name of the gene location file
        gene_id_filename: (String) name of the UCSC-HGNC gene name map file
    Returns:
        A pandas DataFrame with the location of each gene on the hg19 genome
    '''
    # Load gene locations
    df_loc0 = pd.read_csv(gene_locations_filename, sep = '\t', header = 0)
    
    # Load correspondance between UCSC gene name and HGNC gene name
    df_name0 = pd.read_csv(gene_id_filename, sep = '\t', header = 0,dtype={"rfamAcc": str, "tRnaName": str})
    # Only keep UCSC name and geneSymbol
    df_name0 = df_name0[['#kgID','geneSymbol']]
    df_name0.columns = ['transcript','geneSymbol']
    df_name0['geneSymbol'] = df_name0['geneSymbol'].str.upper()
    
    # Merge df_loc0 and df_name0
    df_loc1 = pd.merge(df_name0, df_loc0, on=['transcript'])
    
    # Filter out irregular chromosomes
    keep_chrom = ['chr'+str(i) for i in np.arange(1,23)]
    df_loc2 = df_loc1[df_loc1['#chrom'].isin(keep_chrom)]
    df_loc2 = df_loc2.sort_values(by=['geneSymbol'])
    
    # Remove ribosomal RNA (may be able to remove more irrelevant things, how?)
    df_loc3 = df_loc2[~df_loc2['geneSymbol'].str.contains('RRNA')]
    
    # Drop duplicates
    df_loc4 = df_loc3.drop_duplicates(subset=['geneSymbol'], keep='first')
    
    return(df_loc4)


def get_chrom_sizes(genome_dir,resol):
    '''
    Constructs dataframe of chromosome sizes (in bp and in loci counts at the chosen resolution)
    Args:
        genome_dir: (string) directory of genome data
        resol: (int) hic resolution
    Returns:
        A pandas datafrawith columns chr (chromosome), size, size_loci, size_roundup
    '''
    sizes_filename = genome_dir+'chrom_hg19.sizes'
    df_sizes = pd.read_csv(sizes_filename, sep = '\t', header = None, names=['chr','size'])
    df_sizes['chr'] = df_sizes['chr'].str.split('chr',expand=True)[1]
    df_sizes['size_loci'] = np.ceil(df_sizes['size']/resol).astype(int)
    df_sizes['size_roundup'] = np.ceil(df_sizes['size']/resol).astype(int)*resol
    df_sizes = df_sizes[~df_sizes['chr'].str.contains('hap|alt|Un|rand')]
    return(df_sizes)


def unnesting(df, explode):
    """ Helper function to explode a dataframe based on a given column
    Args:
        df: (pandas DataFrame) the datframe to explode
        explode: (list of String) list of columns to explode on
    
    Returns:
        Exploded dataframe
    """
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    return df1.join(df.drop(explode, 1), how='left')
    

def community_layout(g, partition):
    '''
    Compute the layout for a modular graph.
    Args:
        g: (networkx.Graph or networkx.DiGraph instance) graph to plot
        partition: (dict mapping int node -> int community) graph partition
    Returns:
        pos: (dict mapping int node -> (float x, float y)) node positions
    '''
    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)
    # Combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]
    return pos


def _position_communities(g, partition, **kwargs):
    '''
    Create a weighted graph, in which each node corresponds to a community,
    and each edge weight to the number of edges between communities
    Args:
        g: (networkx.Graph or networkx.DiGraph instance) graph to plot
        partition: (dict mapping int node -> int community) graph partition
    Returns:
        pos: (dict mapping int node -> (float x, float y)) node positions
    '''
    between_community_edges = _find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight= 100)#np.power(len(edges),0.1))
    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)
    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
    return pos


def _find_between_community_edges(g, partition):
    '''
    Args:
        g: (networkx.Graph or networkx.DiGraph instance) graph to plot
        partition: (dict mapping int node -> int community) graph partition
    Returns:
        edges: (dict) mapping pairs of communities to the number of edges 
        between these communities
    '''
    edges = dict()
    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]
        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]
    return edges


def _position_nodes(g, partition, **kwargs):
    '''
    Positions nodes within communities.
    Args:
        g: (networkx.Graph or networkx.DiGraph instance) graph to plot
        partition: (dict mapping int node -> int community) graph partition
    Returns:
        pos: (dict mapping int node -> (float x, float y)) node positions
    '''
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]
    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)
    return pos
    