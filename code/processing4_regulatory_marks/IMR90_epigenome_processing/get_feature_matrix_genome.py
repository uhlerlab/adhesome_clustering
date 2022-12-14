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
    
    # Identify start and stop sites of adhesome genes and active genes of TFs targeting adhesome genes
    print('Identify start and stop sites of adhesome genes...')
    # Load active adhesome genes and active genes of TFs targeting adhesome genes
    active_adh_tf_genes = pickle.load(open(saving_dir+'genes_in_cluster.pkl', 'rb'))
    # Load gene location in hg19
    gene_locations_filename = genome_dir+'chrom_hg19.loc_canonical'
    gene_id_filename = genome_dir+'chrom_hg19.name'
    df_loc = get_gene_locations(gene_locations_filename, gene_id_filename)
    # Find location of adhesome genes
    adh_tf_loc = df_loc[df_loc['geneSymbol'].isin(active_adh_tf_genes)][['geneSymbol','#chrom','chromStart','chromEnd']]
    adh_tf_loc['geneLength'] = adh_tf_loc['chromEnd']-adh_tf_loc['chromStart']
    adh_tf_loc.columns = ['gene','chrom','start','end','length']
    adh_tf_loc = adh_tf_loc.sort_values(by=['chrom','start'])
    # Divide genome into portions corresponding to adhesome genes of interest 
    df_adh_tf_pos = adh_tf_loc[['chrom','start','end']]
    # Convert to bed file
    bed_adh_tf = pybedtools.BedTool.from_dataframe(df_adh_tf_pos)
    bed_adh_tf_df = bed_adh_tf.to_dataframe()
    
    # Process all epigenomic features
    for i in range(len(df)):
        f = df.iloc[i]['filename']
        feature = df.loc[i,'name']
        print('Process '+feature)
        # Get bed file of the feature
        bed = pybedtools.BedTool(epigenome_dir + f).sort()
        # Get counts for this feature and this chromosome
        out = pybedtools.bedtool.BedTool.map(bed_adh_tf, bed, c = [2,3], o = 'count_distinct')
        counts = out.to_dataframe()['name'].values
        # Store results into matrix
        adh_tf_loc[feature] = counts
        # Normalize by gene length
        adh_tf_loc['norm_'+feature] = np.log(1+1000000*adh_tf_loc[feature]/adh_tf_loc['length'])
        # z-score
        mean = adh_tf_loc['norm_'+feature].mean()
        std = adh_tf_loc['norm_'+feature].std()
        adh_tf_loc['z_'+feature] = (adh_tf_loc['norm_'+feature]-mean)/std
        
    # Store the results
    print('Store result')
    pickle.dump(adh_tf_loc, open(saving_dir+'adh_tf_with_epigenomics.pkl', 'wb'))
    
    # Load processing of all epigenomic features
    adh_tf_loc = adh_tf_loc[[col for col in adh_tf_loc.columns if ('z_' in col) or (col == 'gene')]]
    adh_tf_loc = adh_tf_loc.set_index('gene').transpose()

    # Compute Spearman correlation between adhesome genes
    adh_tf_loc_corr = adh_tf_loc.corr(method='spearman')
    pickle.dump(adh_tf_loc_corr, open(saving_dir+'selected_genes_loc_corr.pkl','wb'))
    

if __name__ == "__main__":
    main()
    




