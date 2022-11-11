import sys, getopt
import json
import os, os.path
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn
import pandas as pd
import pickle
import pybedtools
from joblib import Parallel, delayed

''' This script counts the number of peaks per bin of Hi-C (250kb in the paper) for each genomic feature 
and outputs a matrix of feature x region for each chromosome separately.'''


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
    
    # Get dataframe listing available epigenomic data
    print('Get dataframe of available epigenomic data...')
    df = pd.read_csv(epigenome_dir+'filenames_belyaeva.csv', sep=',', header=0)
    
    # Identify location of all genes
    print('Identify location of all genes...')
    # Load gene location in hg19
    gene_locations_filename = genome_dir+'chrom_hg19.loc_canonical'
    gene_id_filename = genome_dir+'chrom_hg19.name'
    df_loc = get_gene_locations(gene_locations_filename, gene_id_filename)
    # Find location of all genes
    all_genes_loc = df_loc[['geneSymbol','#chrom','chromStart','chromEnd']]
    all_genes_loc['geneLength'] = all_genes_loc['chromEnd']-all_genes_loc['chromStart']
    all_genes_loc.columns = ['gene','chrom','start','end','length']
    all_genes_loc = all_genes_loc.sort_values(by=['chrom','start'])
    # Divide genome into portions corresponding to genes
    all_genes_pos = all_genes_loc[['chrom','start','end']]
    # Convert to bed file
    bed_all_genes = pybedtools.BedTool.from_dataframe(all_genes_pos)
    bed_all_genes_df = bed_all_genes.to_dataframe()
    
    # Process all epigenomic features
    for i in range(len(df)):
        f = df.iloc[i]['filename']
        feature = df.loc[i,'name']
        print('Process '+feature)
        # Get bed file of the feature
        bed = pybedtools.BedTool(epigenome_dir + f).sort()
        # Get counts for this feature and this chromosome
        out = pybedtools.bedtool.BedTool.map(bed_all_genes, bed, c = [2,3], o = 'count_distinct')
        counts = out.to_dataframe()['name'].values
        # Store results into matrix
        all_genes_loc[feature] = counts
        # Normalize by gene length
        all_genes_loc['norm_'+feature] = np.log(1+1000000*all_genes_loc[feature]/all_genes_loc['length'])
        # z-score
        mean = all_genes_loc['norm_'+feature].mean()
        std = all_genes_loc['norm_'+feature].std()
        all_genes_loc['z_'+feature] = (all_genes_loc['norm_'+feature]-mean)/std
    
    # Normalize epigenomic features
    all_genes_loc = all_genes_loc[[col for col in all_genes_loc.columns if ('z_' in col) or (col == 'gene')]]
    all_genes_loc = all_genes_loc.set_index('gene').transpose()
    all_genes_loc = all_genes_loc.sort_index(axis=1)
    all_genes_loc.index = all_genes_loc.index.str.strip('z_')
    
    # Store the results
    print('Store result')
    all_genes_loc.to_csv(saving_dir+'features_matrix_all_genes_norm.csv', header=True, index=True)


if __name__ == "__main__":
    main()
    




