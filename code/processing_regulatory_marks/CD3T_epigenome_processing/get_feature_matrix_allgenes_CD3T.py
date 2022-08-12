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
    epigenome_peaks_dir = config['EPIGENOME_PEAKS_DIR']
    epigenome_rnaseq_dir = config['EPIGENOME_RNASEQ_DIR']
    processed_epigenome_data_dir = config['PROCESSED_EPIGENOME_DIR']
    saving_dir = config['SAVING_DIR']
    
    # Load mapping between Ensembl genes and HGN gene names
    print('Load mapping between Ensembl genes and HGN gene names...')
    map1 = pd.read_csv(genome_dir+'ensemblGene2Transcript2Protein', sep='\t', header=0, index_col=None)
    map1 = map1.set_index('transcript')
    map2 = pd.read_csv(genome_dir+'ensemblGene2Name', sep='\t', header=0, index_col=None)
    map2 = map2.set_index('#name')
    ensembl2name = map1.join(map2, how='inner')
    ensembl2name = ensembl2name.reset_index()
    ensembl2name.columns = ['transcript', 'gene', 'protein', 'name']
    ensembl2name = ensembl2name[['gene', 'name']].drop_duplicates(['gene', 'name'])
    ensembl2name = ensembl2name.set_index('gene')
    
    # Load gene location in hg19
    print('Load gene locations in hg19...')
    gene_locations_filename = genome_dir+'chrom_hg19.loc_canonical'
    gene_id_filename = genome_dir+'chrom_hg19.name'
    df_loc = get_gene_locations(gene_locations_filename, gene_id_filename)
    df_loc = df_loc[['geneSymbol','#chrom','chromStart','chromEnd']]
    df_loc['geneLength'] = df_loc['chromEnd']-df_loc['chromStart']
    df_loc.columns = ['gene','chrom','start','end','length']
    df_loc = df_loc.sort_values(by=['chrom','start']).reset_index().iloc[:,1:]
    all_genes = df_loc['gene'].unique()
    
    # Divide genome into portions corresponding to all genes
    df_pos = df_loc[['chrom','start','end']]
    # Convert to bed file
    bed_all_genes = pybedtools.BedTool.from_dataframe(df_pos)
    bed_all_genes_df = bed_all_genes.to_dataframe()
    
    # Get dataframe listing available epigenomic data
    print('Get dataframe of available epigenomic data...')
    func_gen_table = pd.read_csv(epigenome_peaks_dir+'filenames.csv', sep=',', header=0)
    
    # Call ChIP-seq peaks
    print('Process ChIP-seq data...')
    for feature in func_gen_table['name'].values:
        print('Process '+feature)
        
        # Get bed file of feature
        f = func_gen_table[func_gen_table['name']==feature].iloc[0]['filename']+'.gz'
        bed = pybedtools.BedTool(epigenome_peaks_dir + f).sort()
        # Get counts for this feature in the segmented genome
        out = pybedtools.bedtool.BedTool.map(bed_all_genes, bed, c = [2,3], o = 'count_distinct')
        counts = out.to_dataframe()['name'].values
        
        # Add results to df_loc
        df_loc[feature] = counts
        # Normalize by gene length
        df_loc['norm'+feature] = np.log(1+1000000*df_loc[feature]/df_loc['length'])
        # z-score
        mean = df_loc['norm'+feature].mean()
        std = df_loc['norm'+feature].std()
        df_loc['z_'+feature] = (df_loc['norm'+feature]-mean)/std
    
    # Process RNA-seq data
    print('Process RNA-seq data...')
    sample_df = pd.read_csv(epigenome_rnaseq_dir+'GSM2400247_ENCFF383EXA_gene_quantifications_hg19.tsv', 
                            sep='\t', header=0, index_col=None)
    sample_df = sample_df[['gene_id', 'length', 'effective_length', 'expected_count', 'TPM', 'FPKM']]
    sample_df['gene_id'] = sample_df['gene_id'].str.split('.', expand=True)[0]
    sample_df = sample_df.set_index('gene_id')
    sample_df = ensembl2name.join(sample_df, how='inner')
    sample_df = sample_df[sample_df['name'].isin(all_genes)]
    sample_df = sample_df.drop_duplicates('name')
    sample_df['logTPM'] = np.log(1+sample_df['TPM'])
    sample_df = sample_df.reset_index()[['name', 'logTPM']].set_index('name')
    mean = sample_df['logTPM'].mean()
    std = sample_df['logTPM'].std()
    sample_df['z_RNAseq'] = (sample_df['logTPM']-mean)/std
    
    # Combine ChiP-seq peaks with RNA-seq
    print('Combine ChiP-seq peaks and RNA-seq...')
    df_loc_combined = df_loc.set_index('gene').join(sample_df, how='inner')
    df_loc_combined = df_loc_combined.rename_axis('gene').reset_index()

    # Normalize epigenomic features
    df_loc_combined = df_loc_combined[[col for col in df_loc_combined.columns if ('z_' in col) or (col == 'gene')]]
    df_loc_combined = df_loc_combined.set_index('gene').transpose()
    df_loc_combined = df_loc_combined.sort_index(axis=1)
    df_loc_combined.index = df_loc_combined.index.str.strip('z_')

    # Store the results
    print('Store result')
    df_loc_combined.to_csv(saving_dir+'features_matrix_all_genes_norm_GM12878.csv', header=True, index=True)


if __name__ == "__main__":
    main()
    




