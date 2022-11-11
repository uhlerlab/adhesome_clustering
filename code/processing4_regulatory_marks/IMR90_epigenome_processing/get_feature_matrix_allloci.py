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
    
    # Divide genome into portions corresponding to loci
    print('Identify location of all genes...')
    df_sizes = get_chrom_sizes(genome_dir, resol)
    df_chrom_list = []
    for chrom in chr_list:
        # Get chromosome size
        chrom_size = int(df_sizes.loc[df_sizes['chr']==str(chrom)]['size'])
        # Divide the chromosome into segments of HIC_RESOLN length
        stop_pos = np.arange(resol, chrom_size + resol, resol, dtype = 'int')
        df_chrom = pd.DataFrame()
        df_chrom['chrom'] = ['chr' + str(chrom)]*len(stop_pos)
        df_chrom['start'] = stop_pos - resol
        df_chrom['stop'] = stop_pos
        df_chrom_list.append(df_chrom)
    all_loci_pos = pd.concat(df_chrom_list, axis=0)
    
    # Create dataframe of loci location
    all_loci_loc = all_loci_pos.copy()
    all_loci_loc['chrom'] = all_loci_loc['chrom'].str.strip('chr').astype(int)
    all_loci_loc['locus'] = ['chr_'+str(all_loci_loc.iloc[i]['chrom'])+'_loc_'+str(all_loci_loc.iloc[i]['start'])
                              for i in range(all_loci_loc.shape[0])]
    all_loci_loc.columns = ['#chrom', 'chromStart', 'chromEnd', 'locus']
    all_loci_loc = all_loci_loc[['locus', '#chrom', 'chromStart', 'chromEnd']]
    all_loci_loc['locusLength'] = resol
    all_loci_loc.columns = ['locus','chrom','start','end','length']
    all_loci_loc = all_loci_loc.sort_values(by=['chrom','start'])
    
    # Convert all_loci_pos to bed file
    bed_all_loci = pybedtools.BedTool.from_dataframe(all_loci_pos)
    bed_all_loci = bed_all_loci.sort()
    bed_all_loci_df = bed_all_loci.to_dataframe()
    
    # Process all epigenomic features
    for i in range(len(df)):
        f = df.iloc[i]['filename']
        feature = df.loc[i,'name']
        print('Process '+feature)
        # Get bed file of the feature
        bed = pybedtools.BedTool(epigenome_dir + f).sort()
        # Get counts for this feature and this chromosome
        out = pybedtools.bedtool.BedTool.map(bed_all_loci, bed, c = [2,3], o = 'count_distinct')
        counts = out.to_dataframe()['name'].values
        # Store results into matrix
        all_loci_loc[feature] = counts
        # Normalize by locus length
        all_loci_loc['norm_'+feature] = np.log(1+1000000*all_loci_loc[feature]/resol)
        # z-score
        mean = all_loci_loc['norm_'+feature].mean()
        std = all_loci_loc['norm_'+feature].std()
        all_loci_loc['z_'+feature] = (all_loci_loc['norm_'+feature]-mean)/std
    
    # Track non-centerdized features
    all_loci_loc_nonorm = all_loci_loc[[col for col in all_loci_loc.columns if ('norm_' in col) or (col == 'locus')]]
    all_loci_loc_nonorm = all_loci_loc_nonorm.set_index('locus').transpose()
    all_loci_loc_nonorm = all_loci_loc_nonorm.sort_index(axis=1)
    all_loci_loc_nonorm.index = all_loci_loc_nonorm.index.str.strip('z_')
    
    # Normalize epigenomic features
    all_loci_loc = all_loci_loc[[col for col in all_loci_loc.columns if ('z_' in col) or (col == 'locus')]]
    all_loci_loc = all_loci_loc.set_index('locus').transpose()
    all_loci_loc = all_loci_loc.sort_index(axis=1)
    all_loci_loc.index = all_loci_loc.index.str.strip('z_')
    
    # Store the results
    print('Store result')
    all_loci_loc_nonorm.to_csv(saving_dir+'features_matrix_all_loci_nonorm.csv', header=True, index=True)
#     all_loci_loc.to_csv(saving_dir+'features_matrix_all_loci_norm.csv', header=True, index=True)


if __name__ == "__main__":
    main()
    




