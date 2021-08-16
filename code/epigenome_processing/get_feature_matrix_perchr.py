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


def make_chrom_bed(chrom, chrom_size, resol):
    '''
    Create dataframe dividing the chromosome chrom into segments of resol length
    Args:
        chrom: (int) chromosome of interest
        chrom_size: (int) size of chrom
        resol: (int) Hi-C resolution
    Returns:
        A BED file containing the dataframe
    '''
    # Divide the chromosome into segments of HIC_RESOLN length
    stop_pos = np.arange(resol, chrom_size + resol, resol, dtype = 'int')
    df_chrom = pd.DataFrame()
    df_chrom['chrom'] = ['chr' + str(chrom)]*len(stop_pos)
    df_chrom['start'] = stop_pos - resol
    df_chrom['stop'] = stop_pos

    # Convert to bed file
    bed_chrom = pybedtools.BedTool.from_dataframe(df_chrom)
    return(bed_chrom)


def get_feature_matrix_chrom(bed_chrom, df, epigenome_dir):
    '''
    Create dataframe with features as indices and loci as columns for loci in bed_chrom
    Args:
        bed_chrom: (BED file) loci of interest (usually all loci of a given chromosome)
        df: (pandas DataFrame) epigenomic features metadata
        epigenome_dir: (string) directory of raw epigenomic data
    Returns:
        A BED file containing the dataframe
    '''    
    bed_chrom_df = bed_chrom.to_dataframe()
    feature_matrix = pd.DataFrame(index = df['feature'].values, columns = bed_chrom_df['start'].values)
    
    for i in range(len(df)):
        f = df.loc[i,'filename']
        feature = df.loc[i,'feature']
        # Get bed file of the feature
        bed = pybedtools.BedTool(epigenome_dir + f).sort()
        # Get counts for this feature and this chromosome
        out = pybedtools.bedtool.BedTool.map(bed_chrom, bed, c = df.loc[i,'identifier_col'], o = 'count_distinct')
        counts = out.to_dataframe()['name'].values
        # Store results into matrix
        feature_matrix.loc[feature, :] = counts
    return(feature_matrix)


def get_filtered_chipseq(chrom, blacklist, processed_epigenome_data_dir):
    '''
    Removes blacklisted loci from epigenomic data for chromosome chrom
    Args:
        chrom: (int) chromosome of interest
        blacklist: (dict) dictionary of blacklisted loci for every chromosome
        processed_epigenome_data_dir: (string) directory of processes epigenomic data
    Returns:
        Epigenomic data for chromosome chrom without blacklisted loci
    '''
    df_chipseq = pd.read_csv(processed_epigenome_data_dir+'features_matrix_chr'+str(chrom)+'.csv', index_col = 0)
    # get all blacklisted loccations
    blacklist_chr = blacklist[chrom]
    # get a list of columns to keep
    allcols = set(map(int,df_chipseq.columns))
    cols2keep = allcols - blacklist_chr
    df_chipseq_filt = df_chipseq[list(map(str,cols2keep))]
    return df_chipseq_filt    
    

def get_mean_std(chr_list, blacklist, processed_epigenome_data_dir):
    '''
    Get mean and standard deviation for each epigenomic feature across all the genome (without considering
    blacklisted loci)
    Args:
        chr_list: (list) list of chromosomes
        blacklist: (dict) dictionary of blacklisted loci for every chromosome
        processed_epigenome_data_dir: (string) directory of processes epigenomic data
    Returns:
        An array containing the mean and standard deviation for each epigenomic feature
    '''
    # collect chipseq data across all chromosomes into one dataframe
    df_all = pd.DataFrame()
    for chrom in chr_list:
        df_chipseq_filt = get_filtered_chipseq(chrom, blacklist, processed_epigenome_data_dir)
        df_all = pd.concat([df_all, df_chipseq_filt], axis=1)
    # transform
    df_all = np.log(df_all + 1)
    # find mean and standard dev
    mean_features = np.mean(df_all, axis=1)
    std_features = np.std(df_all, axis=1)
    return mean_features, std_features


def normalize_chipseq(chr_list, mean_features, std_features, processed_epigenome_data_dir):
    '''
    For each chromosome, center each epigenomic feature using mean_features and scale it using std_features
    Args:
        chr_list: (list) list of chromosomes
        mean_features: (Numpy array) array of means for centering each epigenomic feature
        std_features: (Numpy array) array of standard deviations for scaling each epigenomic feature
        processed_epigenome_data_dir: (string) directory of processes epigenomic data
    Returns:
    '''
    for chrom in chr_list:
        # get chipseq data
        df_chipseq = pd.read_csv(processed_epigenome_data_dir+'features_matrix_chr'+str(chrom)+'.csv', index_col = 0)
        # transform
        df_chipseq = np.log(df_chipseq + 1)
        # normalize
        df_norm = (df_chipseq.T - mean_features)/std_features
        # transpose back
        df_norm = df_norm.T
        # save
        df_norm.to_csv(processed_epigenome_data_dir+'features_matrix_chr'+str(chrom)+'_norm.csv')

        
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
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    chr_list = config['chrs']
    
    # Get chromosome sizes
    df_sizes = get_chrom_sizes(genome_dir, resol)
    # Get dataframe listing available epigenomic data
    df = pd.read_csv(epigenome_dir + 'features.csv', sep='\t', header=0, index_col=0)
    
    # Get feature matrix for every chromosome in chr_list
    print('Get feature matrix for all chromosomes')
    for chrom in chr_list:
        print('Get feature matrix for chr'+str(chrom))
        # Get chromosome size
        chrom_size = int(df_sizes.loc[df_sizes['chr']==str(chrom)]['size'])
        # Divide the chromosome into segments of HIC_RESOLN length
        bed_chrom = make_chrom_bed(chrom, chrom_size, resol)
        # Process epigenomic data for chrom
        feature_matrix = get_feature_matrix_chrom(bed_chrom, df, epigenome_dir)  
        # write feature matrix to file
        feature_matrix.to_csv(processed_epigenome_data_dir + 'features_matrix_chr' + str(chrom) + '.csv')
  
    # Normalize feature matrix for all chromosomes
    print('Normalize epigenomic data for all chromosomes')
    blacklist = pickle.load(open(processed_hic_data_dir+'blacklist.pickle', 'rb'))
    mean_features, std_features = get_mean_std(chr_list, blacklist, processed_epigenome_data_dir)
    normalize_chipseq(chr_list, mean_features, std_features, processed_epigenome_data_dir)


if __name__ == "__main__":
    main()
    




