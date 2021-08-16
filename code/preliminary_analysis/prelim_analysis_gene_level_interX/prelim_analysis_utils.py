# Import standard libraries
import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import scipy.stats as ss
import csv
import pandas as pd
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
    return df1.join(df.drop(explode, 1), how='left')


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
    

def get_gene_locations(gene_locations_filename, gene_id_filename):
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
    
    
def get_adhesome_genes_location(df_components, df_loc):
    '''
    Retrieves the location of each adhesome gene on the hg19 reference genome
    Args:
        df_components: (pandas DataFrame) dataframe of adhesome genes with metadata
        df_loc: (pandas DataFrame) dataframe of gene locations on the hg19 reference genome
    Returns:
        A pandas DataFrame indicating the position of every adhesome gene on the hg19 reference genome
    '''
    # Find location of adhesome genes by merging df_components and df_loc
    adhesome_loc_df0 = pd.merge(df_components, df_loc, on=['geneSymbol'], how='inner')
    missing_adhesome_genes = list(set(df_components['geneSymbol']).difference(set(df_loc['geneSymbol'])))
    print('Adhesome genes absent from UCSC genes: '+str(missing_adhesome_genes))
    
    # Only keep relevant columns for subsequent analysis
    adhesome_loc_df = adhesome_loc_df0[['geneSymbol','#chrom','chromStart','chromEnd']]
    adhesome_loc_df.columns = ['gene','chrom','genoStart','genoEnd']
    adhesome_loc_df.loc[:,'geneLength'] = list(adhesome_loc_df['genoEnd']-adhesome_loc_df['genoStart'])
    
    return(adhesome_loc_df, missing_adhesome_genes)
 

def get_adhesome_genes_loci(adhesome_loc_df, resol):
    '''
    Finds all loci on the hg19 reference genome corresponding to adhesome genes
    Args:
        adhesome_loc_df: (pandas DataFrame) dataframe containing the location of adhesome genes on the hg19 reference genome
        resol: (int) the resolution of HiC data
    Returns:
        A pandas DataFrame where each row corresponds to one adhesom gene locus, including gene coverage information (i.e. the 
        proportion of the locus occupied by the corresponding gene)
    '''
    # Specify start locus and end locus
    adhesome_loc_df.loc[:,'startLocus_id'] = list((adhesome_loc_df['genoStart']//resol+1))
    adhesome_loc_df.loc[:,'endLocus_id'] = list((adhesome_loc_df['genoEnd']//resol+1))
    adhesome_loc_df.loc[:,'startLocus'] = list((adhesome_loc_df['genoStart']//resol+1)*resol)
    adhesome_loc_df.loc[:,'endLocus'] = list((adhesome_loc_df['genoEnd']//resol+1)*resol)
    
    # Compute coverage of the gene on its start locus and end locus
    mask = adhesome_loc_df['startLocus']==adhesome_loc_df['endLocus']
    adhesome_loc_df.loc[mask,'startLocus_coverage'] = adhesome_loc_df.loc[mask,'geneLength']/resol
    adhesome_loc_df.loc[mask,'endLocus_coverage'] = adhesome_loc_df.loc[mask,'startLocus_coverage']
    mask = adhesome_loc_df['startLocus']<adhesome_loc_df['endLocus']
    adhesome_loc_df.loc[mask,'startLocus_coverage'] = (adhesome_loc_df.loc[mask,'startLocus']-adhesome_loc_df.loc[mask,'genoStart'])/resol
    adhesome_loc_df.loc[mask,'endLocus_coverage'] = (adhesome_loc_df.loc[mask,'genoEnd']-adhesome_loc_df.loc[mask,'endLocus']+resol)/resol
    
    # Add column with list of loci
    ngenes = adhesome_loc_df.shape[0]
    adhesome_loc_df.loc[:,'loci'] = [np.arange(adhesome_loc_df.loc[i,'startLocus_id'], adhesome_loc_df.loc[i,'endLocus_id']+1) for i in range(ngenes)]
    
    # Explode dataframe based on loci
    adhesome_loc_df = unnesting(adhesome_loc_df,['loci'])
    
    # Reorganize columns of dataframe
    adhesome_loc_df = adhesome_loc_df[['gene','chrom','loci',
                                       'genoStart','genoEnd','geneLength',
                                       'startLocus_id','startLocus','startLocus_coverage',
                                       'endLocus_id','endLocus','endLocus_coverage']]
    return(adhesome_loc_df)   


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


def plot_heatmap(df, xticklabels, yticklabels, xlabel, ylabel, size, vmax, save_to=''):
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
    Returns:
        void
    '''
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111)
    im = ax.imshow(df, origin='upper', interpolation='none', cmap='rocket')
    # Place ticks at the middle of every pixel
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_ylim(len(yticklabels)-0.5, -0.5)
    # Use input dataframe row and column names as ticks
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=3)
    ax.set_yticklabels(yticklabels, fontsize=3)
    # Put x axis labels on top
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    # Define axis name
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Add colorbar
    fig.colorbar(im)
    im.set_clim(0, vmax)
    if save_to != '':
        # Save plot
        plt.savefig(save_to+'.pdf', format='pdf')
    plt.show()




