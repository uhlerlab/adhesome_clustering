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
    arrowhead_dir = config['ARROWHEAD_DIR']
    saving_dir = config['SAVING_DIR']
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    normalization = config['TAD_NORMALIZATION']
    resolution = config['TAD_RESOLUTION']
    chr_list = config['chrs']
    
    ################################################
    # Get bed object for TADs
    ################################################
    print('Load TADs')
    # Load TAD data
    fname = arrowhead_dir+f'{normalization}/{resolution}/processed_tads_list.csv'
    processed_tads_df = pd.read_csv(fname, header=0, index_col=0)
    # Create BED object
    processed_tads_df.loc[:, 'name'] =  processed_tads_df.apply(
        lambda row: f"tad_chr{row['chromo']}_{row['start']}_{row['end']}", axis=1
    )
    processed_tads_df.loc[:, 'chromo'] = 'chr'+processed_tads_df['chromo'].astype(str)
    processed_tads_df = processed_tads_df[['chromo', 'start', 'end', 'name']]
    processed_tads_df.columns = ['chrom', 'start', 'end', 'name']
    processed_tads_bed = pybedtools.BedTool.from_dataframe(processed_tads_df)
    processed_tads_bed = processed_tads_bed.sort()
    
    # Create output file
    out_cistromic = processed_tads_df.copy()
    out_cistromic.loc[:, 'length'] = out_cistromic['end']-out_cistromic['start']
    
    ################################################
    # Load and process RNA-seq
    ################################################
    feature = 'rnaseq'
    
    # Minus strand
    print('Process RNA-seq (- strand)')
    rnaseq_dir = "/home/louiscam/projects/gpcr/data/regulatory_data/rnaseq-gingeras/"
    minus_rnaseq_df = pd.read_csv(rnaseq_dir+"ENCFF392ICW.bedGraph", sep='\t', header=None)
    minus_rnaseq_df.columns = ['chrom', 'start', 'end', 'score']
    minus_rnaseq_df = minus_rnaseq_df[~minus_rnaseq_df['chrom'].isin(['chrX', 'chrY'])]
    minus_rnaseq_df.loc[:, 'name'] = minus_rnaseq_df.index.astype(str)
    minus_rnaseq_df = minus_rnaseq_df[['chrom', 'start', 'end', 'name', 'score']]
    minus_rnaseq_bed = pybedtools.BedTool.from_dataframe(minus_rnaseq_df)
    minus_rnaseq_bed = minus_rnaseq_bed.sort()
    
    minus_rnaseq_out = pybedtools.bedtool.BedTool.map(processed_tads_bed, minus_rnaseq_bed, c = [5], o = 'sum')
    minus_rnaseq_out = minus_rnaseq_out.to_dataframe()
    minus_rnaseq_out.columns = ['chrom', 'start', 'end', 'name', feature]
    minus_rnaseq_out[feature] = minus_rnaseq_out[feature].astype(str)
    minus_rnaseq_out[feature] = minus_rnaseq_out[feature].str.replace('.', '0').astype(float)    
    
    # Plus strand
    print('Process RNA-seq (+ strand)')
    rnaseq_dir = "/home/louiscam/projects/gpcr/data/regulatory_data/rnaseq-gingeras/"
    plus_rnaseq_df = pd.read_csv(rnaseq_dir+"ENCFF491UTZ.bedGraph", sep='\t', header=None)
    plus_rnaseq_df.columns = ['chrom', 'start', 'end', 'score']
    plus_rnaseq_df = plus_rnaseq_df[~plus_rnaseq_df['chrom'].isin(['chrX', 'chrY'])]
    plus_rnaseq_df.loc[:, 'name'] = plus_rnaseq_df.index.astype(str)
    plus_rnaseq_df = plus_rnaseq_df[['chrom', 'start', 'end', 'name', 'score']]
    plus_rnaseq_bed = pybedtools.BedTool.from_dataframe(plus_rnaseq_df)
    plus_rnaseq_bed = plus_rnaseq_bed.sort()

    plus_rnaseq_out = pybedtools.bedtool.BedTool.map(processed_tads_bed, plus_rnaseq_bed, c = [5], o = 'sum')
    plus_rnaseq_out = plus_rnaseq_out.to_dataframe()
    plus_rnaseq_out.columns = ['chrom', 'start', 'end', 'name', feature]
    plus_rnaseq_out[feature] = plus_rnaseq_out[feature].astype(str)
    plus_rnaseq_out[feature] = plus_rnaseq_out[feature].str.replace('.', '0').astype(float)
    
    # Combine
    out_cistromic.loc[:, feature] = minus_rnaseq_out[feature]+plus_rnaseq_out[feature]
    # Normalize by gene length
    out_cistromic['norm_'+feature] = np.log(1+1000000*out_cistromic[feature]/out_cistromic['length'])
    # z-score
    mean = out_cistromic['norm_'+feature].mean()
    std = out_cistromic['norm_'+feature].std()
    out_cistromic['z_'+feature] = (out_cistromic['norm_'+feature]-mean)/std
    
    ################################################
    # Process other epigenomic marks
    ################################################
    
    # List of all available epigenomic marks
    epigenomic_marks_df = pd.read_csv(epigenome_dir+'filenames_belyaeva.csv', sep=',', header=0)
    
    # Process all epigenomic features other than RNA-seq
    for i in range(len(epigenomic_marks_df)):
        f = epigenomic_marks_df.iloc[i]['filename']
        feature = epigenomic_marks_df.loc[i,'name']
        # Get bed file of the feature
        bed = pybedtools.BedTool(epigenome_dir + f).sort()
        if (feature=='RNAseq'):
            continue
        else:
            print('Process '+feature)
            # Get counts for this feature and this chromosome
            out = pybedtools.bedtool.BedTool.map(processed_tads_bed, bed, c = [4], o = 'count_distinct')
            out = out.to_dataframe()
            # Store results into out_cistromic
            out_cistromic.loc[:, feature] = out['score'].values

        # Normalize by gene length
        out_cistromic['norm_'+feature] = np.log(1+1000000*out_cistromic[feature]/out_cistromic['length'])
        # z-score
        mean = out_cistromic['norm_'+feature].mean()
        std = out_cistromic['norm_'+feature].std()
        out_cistromic['z_'+feature] = (out_cistromic['norm_'+feature]-mean)/std
    
    # Normalize epigenomic features
    out_cistromic = out_cistromic[
        [col for col in out_cistromic.columns if (('z_' in col) or (col=='name'))]
    ]
    out_cistromic = out_cistromic.set_index(['name']).transpose()
    out_cistromic = out_cistromic.sort_index(axis=1)
    out_cistromic.index = out_cistromic.index.str.strip('z_')
    out_cistromic = out_cistromic.transpose()
                               
    ################################################
    # Save results
    ################################################
    
    # Store the results
    print('Store result')
    out_cistromic.to_csv(saving_dir+f'features_matrix_all_tads_norm{normalization}_resolution{resolution}.csv', header=True, index=True)


if __name__ == "__main__":
    main()