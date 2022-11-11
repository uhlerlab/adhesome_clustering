# Import standard libraries
import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
import csv
import pandas as pd
import networkx as nx
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools
import os.path
import math
import time
from tqdm import tqdm


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
    return df1.join(df.drop(labels=explode, axis=1), how='left')


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


def load_adhesome_data(adhesome_components_filename):
    '''
    Loads dataframe of intrinsic and associated adhesome genes
    Args:
        adhesome_components_filename: (String) adhesome file name
    Returns:
        A pandas DataFrame including adhesome genes with metadata
    '''
    df_components = pd.read_csv(adhesome_components_filename, sep = ',', header = 0)
    df_components['Official Symbol'] = df_components['Official Symbol'].str.upper()
    df_components.columns = ['geneSymbol','geneID','proteinName','swisprotID','synonyms','functionalCategory','FA']
    return(df_components)
    

def get_all_gene_locations(gene_locations_filename, gene_id_filename):
    '''
    Loads dataframe of gene locations on the hg19 reference genome using the UCSC
    gene nomenclature, and matches the name of every UCSC gene to the corresponding 
    HGNC name
    Args:
        gene_locations_filename: (String) file name for the UCSC gene locations
        gene_id_filename: (String) file name for the UCSC-HGNC gene names map
    Returns:
        A pandas DataFrame of HGNC gene locations on the hg19 reference genome
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
    
    
def get_selected_genes_location(df_components, df_loc):
    '''
    Retrieves the location of each selected gene on the hg19 reference genome
    Args:
        df_components: (pandas DataFrame) dataframe of adhesome genes with metadata
        df_loc: (pandas DataFrame) dataframe of gene locations on the hg19 reference genome
    Returns:
        A pandas DataFrame indicating the position of every adhesome gene on the hg19 reference genome
    '''
    # Find location of adhesome genes by merging df_components and df_loc
    selected_loc_df0 = pd.merge(df_components, df_loc, on=['geneSymbol'], how='inner')
    missing_selected_genes = list(set(df_components['geneSymbol']).difference(set(df_loc['geneSymbol'])))
    
    # Only keep relevant columns for subsequent analysis
    selected_loc_df = selected_loc_df0[['geneSymbol','#chrom','chromStart','chromEnd']]
    selected_loc_df.columns = ['gene','chrom','genoStart','genoEnd']
    selected_loc_df.loc[:,'geneLength'] = list(selected_loc_df['genoEnd']-selected_loc_df['genoStart'])
    
    return(selected_loc_df, missing_selected_genes)
 

def get_selected_genes_loci(selected_loc_df, resol):
    '''
    Finds all loci on the hg19 reference genome corresponding to selected genes
    Args:
        selected_loc_df: (pandas DataFrame) dataframe containing the location of selected genes on the hg19 reference genome
        resol: (int) the resolution of HiC data
    Returns:
        A pandas DataFrame where each row corresponds to one selected gene locus, including gene coverage information (i.e. the 
        proportion of the locus occupied by the corresponding gene)
    '''
    # Specify start locus and end locus
    selected_loc_df.loc[:,'startLocus_id'] = list((selected_loc_df['genoStart']//resol))
    selected_loc_df.loc[:,'endLocus_id'] = list((selected_loc_df['genoEnd']//resol))
    selected_loc_df.loc[:,'startLocus'] = list((selected_loc_df['genoStart']//resol)*resol)
    selected_loc_df.loc[:,'endLocus'] = list((selected_loc_df['genoEnd']//resol)*resol)

    # Compute coverage of the gene on its start locus and end locus
    mask = selected_loc_df['startLocus']==selected_loc_df['endLocus']
    selected_loc_df.loc[mask,'startLocus_coverage'] = selected_loc_df.loc[mask,'geneLength']/resol
    selected_loc_df.loc[mask,'endLocus_coverage'] = selected_loc_df.loc[mask,'startLocus_coverage']
    mask = selected_loc_df['startLocus']<selected_loc_df['endLocus']
    selected_loc_df.loc[mask,'startLocus_coverage'] = (selected_loc_df.loc[mask,'startLocus']+resol-selected_loc_df.loc[mask,'genoStart'])/resol
    selected_loc_df.loc[mask,'endLocus_coverage'] = (selected_loc_df.loc[mask,'genoEnd']-selected_loc_df.loc[mask,'endLocus'])/resol

    # Add column with list of loci
    ngenes = selected_loc_df.shape[0]
    selected_loc_df.loc[:,'loci'] = [np.arange(selected_loc_df.loc[i,'startLocus_id'], selected_loc_df.loc[i,'endLocus_id']+1) for i in range(ngenes)]

    # Explode dataframe based on loci
    selected_loc_df = unnesting(selected_loc_df,['loci'])

    # Reorganize columns of dataframe
    selected_loc_df = selected_loc_df[['gene','chrom','loci',
                                       'genoStart','genoEnd','geneLength',
                                       'startLocus_id','startLocus','startLocus_coverage',
                                       'endLocus_id','endLocus','endLocus_coverage']]
    return(selected_loc_df)   


def annotate_genes(df_components, df_loc, resol):
    '''
    Annotates a list of genes with location and locus information
    Args:
        df_components: (pandas DataFrame) dataframe containing genes as rows with column name 'geneSymbol'
        df_loc: (pandas DataFrame) dataframe containing the location of all genes on hg19
        resol: (int) HiC resolution
    '''
    # Find location of selected genes
    selected_loc_df, missing_selected_genes = get_selected_genes_location(df_components, df_loc)
    selected_loc_df = get_selected_genes_loci(selected_loc_df, resol)
    selected_loc_df['chrom_int'] = selected_loc_df['chrom'].str.split('chr').str[1].astype(int)
    selected_genes = np.unique(selected_loc_df['gene'])
    # Construct data frame annotating each selected locus with gene
    selected_chr_loci = selected_loc_df[['chrom','chrom_int','loci','gene']]
    selected_chr_loci = selected_chr_loci.sort_values(['chrom_int','loci'])
    selected_loci = [selected_chr_loci.iloc[i]['chrom']+'_'+'loc'+str(selected_chr_loci.iloc[i]['loci']*resol) 
                     for i in range(len(selected_chr_loci))]
    # Add locus ID column
    selected_chr_loci['locus_id'] = ['chr_'+str(selected_chr_loci.iloc[i]['chrom_int'])+'_loc_'
                                     +str(selected_chr_loci.iloc[i]['loci']*resol) 
                                     for i in range(len(selected_chr_loci))]
    return(selected_chr_loci, missing_selected_genes)


def build_adhesome_gene_contact_mat(gene_list, adhesome_chr_loci, dir_processed_hic, resol):
    '''
    Builds a gene x gene matrix where each entry (i,j) corresponds to the total Hi-C contacts between adhesome genes i and j (chosen from gene_list), obtained by summing up Hi-C contacts between the loci corresponding to genes i and j. Contacts are 0 for adhesome genes sitting on the same chromosome.
    Args:
        gene_list: (Numpy array) list of adhesome genes to consider when building the matrix
        adhesome_chr_loci: (pandas DataFrame) dataframe with chromosome and loci information for each adhesome gene
        dir_processed_hic: (String) directory of processed Hi-C data
        resol: (int) Hi-C map resolution
    Returns:
        A pandas DataFramce where each entry (i,j) is the the sum of contacts between loci of gene i and loci of gene j
    '''
    # Create empty dataframe indexed by gene
    gene_contacts_df = pd.DataFrame(0, index=gene_list, columns=gene_list)

    # Loop over all chromosome pairs and fill gene_contacts_df
    chr_list = np.arange(1,23,1)
    chr_pairs = list(itertools.combinations(chr_list, 2))
    for pair in tqdm(chr_pairs):
        time.sleep(.01)
        chr1, chr2 = pair

        # Restrict adhesome genes dataframe to chr1 and chr2
        adhesome_chr1_df = adhesome_chr_loci.loc[adhesome_chr_loci['chrom']=='chr'+str(chr1)]
        adhesome_genes_chr1 = np.unique(adhesome_chr1_df['gene'])
        adhesome_chr2_df = adhesome_chr_loci.loc[adhesome_chr_loci['chrom']=='chr'+str(chr2)]
        adhesome_genes_chr2 = np.unique(adhesome_chr2_df['gene'])

        # Load HiC data for this chromosome pair
        processed_hic_filename = 'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter3'+'.pkl'
        hic_chpair_df = pickle.load(open(dir_processed_hic+processed_hic_filename, 'rb'))

        # Fill in corresponding submatrix of gene_contacts_df by sum of Hi-C contacts across all gene loci
        gene_pairs = itertools.product(adhesome_genes_chr1,adhesome_genes_chr2)
        for gene1, gene2 in gene_pairs:
            gene1_loci = np.array(adhesome_chr1_df[adhesome_chr1_df['gene']==gene1]['loci']*resol)
            gene2_loci = np.array(adhesome_chr2_df[adhesome_chr2_df['gene']==gene2]['loci']*resol)
            gene_contacts_df.loc[gene1, gene2] = hic_chpair_df.loc[gene1_loci,gene2_loci].sum().sum()
            
    return(gene_contacts_df)


def plot_heatmap(df, xticklabels, yticklabels, xlabel, ylabel, size, vmax, vmin=0, fontsize=3, save_to='', add_patches=[], cmap='rocket'):
    '''
    Plots the heatmap of the input dataframe.
    Args:
        df: (pandas DataFrame) dataframe to plot
        xticklabels: (Numpy array) labels of x-ticks
        ticklabels: (Numpy array) labels of y-ticks
        xlabel: (String) label of x-axis
        ylabel: (String) label of y-axis
        size: (int) size of figure
        vmax: (float) color bar upper limit
        save_to: (String) name of figure for saving
        add_patches: (list) list of patches to add on the image
    Returns:
        void
    '''
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111)
    im = ax.imshow(df, origin='upper', interpolation='none', cmap=cmap)
    # Place ticks at the middle of every pixel
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_ylim(len(yticklabels)-0.5, -0.5)
    # Use input dataframe row and column names as ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    # Put x axis labels on top
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    # Define axis name
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add patches to the axis
    if len(add_patches)>0:
        for patch in add_patches:
            rect = patches.Rectangle((patch[0], patch[1]), patch[2], patch[3], 
                                     linewidth=2, edgecolor=patch[4], facecolor='none')
            ax.add_patch(rect)
    
    # Add colorbar
    fig.colorbar(im, fraction=0.046, pad=0.04)
    im.set_clim(vmin, vmax)
    if save_to != '':
        # Save plot
        plt.savefig(save_to+'.pdf', format='pdf')
    plt.show()


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


def from_nx(graph, pyvisnet, node_color_list,
            default_node_size=1, default_edge_weight=1, 
            edge_weight_scale=1, edge_color_scale=1,
            shape='circle'):
    """
    This method takes an exisitng Networkx graph and translates
    it to a PyVis graph format that can be accepted by the VisJs
    API in the Jinja2 template. This operation is done in place.
    """
#     assert(isinstance(nx_graph, nx.Graph))
#     edges = nx_graph.edges(data=True)
#     nodes = nx_graph.nodes(data=True)

#     if len(edges) > 0:
#         for e in edges:
#             if 'size' not in nodes[e[0]].keys():
#                 nodes[e[0]]['size'] = default_node_size
#             nodes[e[0]]['size'] = default_node_size
#             if 'size' not in nodes[e[1]].keys():
#                 nodes[e[1]]['size'] = default_node_size
#             nodes[e[1]]['size'] = default_node_size
#             pyvisnet.add_node(e[0], **nodes[e[0]], shape=shape)
#             pyvisnet.add_node(e[1], **nodes[e[1]], shape=shape)

#             if 'weight' not in e[2].keys():
#                 e[2]['weight'] = default_edge_weight
#             pyvisnet.add_edge(e[0], e[1], **e[2])

#     for node in nx.isolates(nx_graph):
#         if 'size' not in nodes[node].keys():
#             nodes[node]['size']=default_node_size
#         pyvisnet.add_node(node, **nodes[node], shape=shape)

    
    
    
    
    
    nx_graph = graph.copy()
    assert(isinstance(nx_graph, nx.Graph))
    edges = nx_graph.edges(data=True)
    nodes = nx_graph.nodes(data=True)

    if len(edges) > 0:
        for e in edges:
            # Specify node size
            nodes[e[0]]['size'] = default_node_size
            nodes[e[1]]['size'] = default_node_size
            # Specify node color
            nodes[e[0]]['color'] = colfunc2(nodes[e[0]]['chrom']-1, minval=0, maxval=22)
            nodes[e[1]]['color'] = colfunc2(nodes[e[1]]['chrom']-1, minval=0, maxval=22)
            # Specify node title
            nodes[e[0]]['title'] = e[0]
            nodes[e[1]]['title'] = e[1]
            pyvisnet.add_node(e[0], **nodes[e[0]], shape=shape)
            pyvisnet.add_node(e[1], **nodes[e[1]], shape=shape)
            # Edge weight
            edge_dict = e[2].copy()
            edge_dict["value"] = e[2]['weight']*edge_weight_scale
            edge_dict["color"] = colfunc(e[2]['weight']*edge_color_scale)
            pyvisnet.add_edge(e[0], e[1], **edge_dict)

#     for node in nx.isolates(nx_graph):
#         nodes[node]['size']= 100 #default_node_size
#         nodes[node]['color'] = colfunc2(nodes[node]['chrom']-1, minval=0, maxval=22)
#         pyvisnet.add_node(node, **nodes[node], shape=shape)
    
    
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

def colfunc2(val, minval=0, maxval=1):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    RED, YELLOW, GREEN  = (1, 0, 0), (1, 1, 0), (0, 1, 0)
    CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
    WHITE = (1, 1, 1)
    f = float(val-minval) / (maxval-minval)
    return mpl.colors.rgb2hex(tuple(f*(b-a)+a for (a, b) in zip(BLUE, GREEN)))

