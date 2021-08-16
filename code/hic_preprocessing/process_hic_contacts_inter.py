import sys, getopt
import json
import os, os.path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import random
from scipy import sparse
import scipy.stats
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pickle
from collections import defaultdict
import operator
from scipy.sparse import csr_matrix
import itertools

import math
import pybedtools


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


def get_centromere_locations(genome_dir):
    '''
    Constructs dataframe with centromere location for all chromosomes
    Args:
        genome_dir: (string) directory of genome data
    Returns:
    A pandas DataFrame with centromere information
    '''
    centrom_filename = genome_dir+'chrom_hg19.centromeres'
    df_centrom = pd.read_csv(centrom_filename, sep = '\t')
    return(df_centrom)


def get_raw_hic_sparse(raw_hic_dir, cell_type, resol_str, resol, quality, chr1, chr2):
    '''
    Constructs dataframe of raw HiC data for a given cell type, resolution, mapping quality and
    chromosome pair, including Knight-Ruiz balancing coefficients.
    Args:
        raw_hic_dir: (string) directory containing all raw hic data
        cell_type: (string) cell type of interest
        resol_str: (str) HiC resolution in string format
        resol: (int) HiC resolution
        quality: (str) maping quality, one of MAPQ0 or MAPQ30
        chr1: the first chromosome of the pair (rows)
        chr2: the second chromosome of the pair (columns)
    Returns:
        A sparse dataframe including contact values for pairs of loci and KR balancing coefficients for 
        each locus of a pair
    '''
    # Construct data directory
    dir_interchrom = raw_hic_dir+cell_type+'_interchromosomal/'+resol_str+'_resolution_interchromosomal/'
    chrom_pair_dir = 'chr'+str(chr1)+'_'+'chr'+str(chr2)+'/'
    dir_quality = quality+'/'
    data_dir = dir_interchrom + chrom_pair_dir + dir_quality
    raw_mat_path = data_dir+'chr'+str(chr1)+'_'+str(chr2)+'_'+resol_str+'.RAWobserved'
    KRnorm_chr1_path = data_dir+'chr'+str(chr1)+'_'+resol_str+'.KRnorm'
    KRnorm_chr2_path = data_dir+'chr'+str(chr2)+'_'+resol_str+'.KRnorm'
    
    # Load raw data
    data = np.loadtxt(raw_mat_path, delimiter = '\t')
    row_ind = data[:,0]/resol
    row_ind = row_ind.astype(int)
    col_ind = data[:,1]/resol
    col_ind = col_ind.astype(int)
    contact_values = data[:,2]
    
    # Load Knight-Ruiz normalization vectors
    norm_chr1 = np.loadtxt(KRnorm_chr1_path, delimiter = '\t')
    norm_chr2 = np.loadtxt(KRnorm_chr2_path, delimiter = '\t')
    
    # Convert sparse Hi-C data to pandas dataframe
    raw_hic_data = pd.DataFrame(data={'locus_chr1':row_ind,'locus_chr2':col_ind,'value':contact_values})
    raw_hic_data['norm_locus_chr1'] = norm_chr1[raw_hic_data['locus_chr1']]
    raw_hic_data['norm_locus_chr2'] = norm_chr2[raw_hic_data['locus_chr2']]

    return(raw_hic_data)


def normalize_raw_hic_sparse(raw_hic_data):
    '''
    Normalize raw HiC data using the Knight-Ruiz algorithm for matrix balancing
    Args:
        raw_hic_data: (pandas DataFrame) raw HiC data with KR coefficients
    Returns:
        A pandas DataFrame including KR-normalized HiC data
    '''
    normalized_hic_data = raw_hic_data.copy()
    normalized_hic_data['norm_value'] = normalized_hic_data['value']/(normalized_hic_data['norm_locus_chr1']*normalized_hic_data['norm_locus_chr2'])
    return(normalized_hic_data)


def get_dense_hic_dataframe(normalized_hic_data, chr1_size, chr2_size, resol):
    '''
    Constructs dense HiC dataframe from HiC data in sparse format
    Args:
        normalized_hic_data: (pandas DataFrame) HiC data in sparse format
        chr1_size: (int) size of chromosome 1
        chr2_size: (int) size of chromosome 2
        resol: (int) HiC resolution
    Returns:
        A dense HiC dataframe
    '''
    hic_sparse_matrix = csr_matrix((normalized_hic_data['norm_value'], 
                             (normalized_hic_data['locus_chr1'], normalized_hic_data['locus_chr2'])), 
                            shape = (chr1_size, chr2_size))
    hic_dense = np.asarray(hic_sparse_matrix.todense())
    row_labels = np.arange(chr1_size)*resol
    col_labels = np.arange(chr2_size)*resol
    df = pd.DataFrame(hic_dense, index = row_labels, columns = col_labels)
    df = df.fillna(0)
    return(df)


def filter_centromeres(df, chrom, row_or_col, df_centrom, filter_size, resol):
    '''
    Filter out loci corresponding to centromeric and pericentromeric regions (defined by
    filter_size) from dense HiC dataframe
    Args:
        df: (pandas DataFrame) dense HiC dataframe
        chrom: (int) chromosome of interest
        row_or_col: (string) whether the chromosome chrom corresponds to the rows or columns of df
        df_centrom: (pandas DataFrame) centromere information
        filter_size: (int) size of filter for the pericentromeric regions
        resol: (int) HiC resolution
    Returns:
        A pandas DataFrame where centromeric and pericentromeric loci have been zeroed out
    '''
    chr_centrom_data = df_centrom[df_centrom['chrom'] == 'chr' + str(chrom)]
    centrom_start = int((chr_centrom_data['chromStart']//resol)*resol)
    centrom_end = int((chr_centrom_data['chromEnd']//resol)*resol)
    centrom_start = centrom_start - filter_size
    centrom_end = centrom_end + filter_size
    if (row_or_col == 'row'):
        df.loc[centrom_start:centrom_end, :] = 0
    if (row_or_col == 'col'):
        df.loc[:, centrom_start:centrom_end] = 0
    return(df)
    

def load_repeats_data(genome_dir):
    '''
    Load information of repeats
    Args:
        genome_dir: (str) directory of genome data
    Returns:
        A pandas DataFrame with information on repeats
    '''
    # Load repeats data
    repeats_filename = 'rmsk.txt'
    repeats_colnames = ['bin','swScore','milliDiv','milliDel','milliIns','genoname','genoStart','genoEnd',
                    'genoLeft','strand','repname','repClass','repFamily','repStart','repEnd','repLeft','id']
    df_repeats0 = pd.read_csv(genome_dir+repeats_filename, header=None , sep= '\t', names = repeats_colnames)
    
    # Select relevant columns and add column with length of repeat
    df_repeats = df_repeats0[['genoname','genoStart','genoEnd']]
    df_repeats.insert(3,'repLength', df_repeats['genoEnd']-df_repeats['genoStart'], True)

    # Filter out alt, fix, random, Un chromosomes
    df_repeats = df_repeats[~df_repeats['genoname'].str.contains('alt|fix|rand|Un|hap')]
    
    return(df_repeats)

def find_repeat_locations(df_repeats, chr_list, df_sizes, resol):
    '''
    Find repeat locations (loci that intersect with repeats) to filter out from data (filtering is done
    for loci whose coverage is higher than the 95th percentile of coverage values across the genome)
    Args:
        df_repeats: (pandas DataFrame) dataframe with information on repeats
        chr_list: (list of ints) list of all chromosomes under consideration
        df_sizes: (pandas DataFrame) a dataframe with chromosome size information
        resol: (int) HiC resolution
    Returns:
        A dictionary with chromosomes as keys and loci to filter out as values
    '''
    # Dataframe including all coverage values across all chromosomes
    df_coverage = pd.DataFrame()
    # Convert df_repeats to BED format for genome arithmetics
    df_repeats_bed = pybedtools.BedTool.from_dataframe(df_repeats)
    
    # Compute coverage values for all start sites for all chromosomes
    for chrom in chr_list:
        # Create dataframe of all possible start sites for this chr
        chrom_size = int(df_sizes[df_sizes['chr']==str(chrom)]['size'])
        start_regions = range(0, chrom_size, resol)
        df_chr = pd.DataFrame({'chr':['chr' + str(chrom)]*len(start_regions), 'start':start_regions})
        df_chr['stop'] = df_chr['start'] + resol
        df_chr_bed = pybedtools.BedTool.from_dataframe(df_chr)
        
        # Compute coverage
        coverage_chr = df_chr_bed.coverage(df_repeats_bed).to_dataframe(names = ['chr', 'start', 'end', 'bases covered', 
                                                                                 'length intersection', 'length locus', 
                                                                                 'coverage'])
        df_coverage = pd.concat([df_coverage, coverage_chr])
    
    # Determine coverage threshold for filtering out (95th percentile)
    df_coverage.columns = ['chr','start','end','bases covered','length intersection','length locus','coverage']
    threshold = np.quantile(a=df_coverage['coverage'],q=0.95)
    
    # Dictionary with chromosomes as keys and loci with repeat coverage information
    dic_repeats_tofilter = {}
    for chrom in chr_list:
        coverage_chr = df_coverage[df_coverage['chr']=='chr'+str(chrom)]
        tofilter = coverage_chr[coverage_chr['coverage'] >= threshold]['start'].values
        dic_repeats_tofilter[str(chrom)] = tofilter
        
    return(dic_repeats_tofilter)


def filter_repeats(df, chrom, dic_repeats_tofilter, row_or_col):
    '''
    Filters out repeat-covered loci for chromsome chrom from the dense HiC dataframe df
    Args:
        df: (pandas DataFrame) dense HiC dataframe
        chrom: (int) chromosome of interest
        dic_repeats_tofilter: (dict) dictionary with chromosomes as keys and loci to filter out as values
        row_or_col: (string) whether the chromosome chrom corresponds to the rows or columns of df
    Returns:
        The dense HiC dataframe where repeat-covered loci for chrom have been zeroed out
    '''
    regions2filter = dic_repeats_tofilter[str(chrom)]
    if (row_or_col == 'row'):
        df.loc[regions2filter,:] = 0
    if (row_or_col == 'col'):
        df.loc[:,regions2filter] = 0
    return df


def detect_upper_outliers(array):
    '''
    Function to detect the upper outliers of an array, defined as all elements that are bigger than 
    QR3+1.5*IQR where QR3 is the 3rd quartile and IQR is the interquartile range
    Args:
        array: (numpy array) array of interest
    Returns:
        Threshold to determine upper outlier values
    '''
    p25 = np.percentile(array, 25)
    p75 = np.percentile(array, 75)
    upper = p75 + 1.5*(p75-p25)
    return upper


def log_transform(df):
    '''
    Log-transform dense HiC dataframe
    Args:
        df: (pandas Dataframe) dense HiC dataframe
    Returns:
        The log-transformed dense HiC dataframe
    '''
    return(np.log(1+df))


def filter_outliers(df_transformed):
    '''
    Filters out outliers of the log-transformed dense HiC dataframe, where outliers for chr1 are
    defined as loci of chr1 whose total contact values with chr2 are bigger than QR3+1.5*IQR where QR3 is 
    the 3rd quartile and IQR is the interquartile range of the total contact values of all chr1 loci with chr2
    (mutatis mutandis for chr2 and chr1)
    Args:
        df_transformed: (pandas Dataframe) log-transformed dense HiC dataframe
    Returns:
        The dense HiC dataframe where outliers have been zeroed out
    '''
    # Get all the row and column sums
    row_orig = np.sum(df_transformed, axis = 1).as_matrix()
    col_orig = np.sum(df_transformed, axis = 0).as_matrix()
    
    # Get rid of zeros (in order to compute row and col thresholds)
    row = row_orig[np.nonzero(row_orig)]
    col = col_orig[np.nonzero(col_orig)]

    # Compute a threshold for outliers for chr1 and chr2
    threshold_row = detect_upper_outliers(row)
    threshold_col = detect_upper_outliers(col)
    
    # Filter out outliers for chr1 (rows)
    ind_row = np.arange(len(row_orig))[row_orig>threshold_row]
    df_transformed.iloc[ind_row,:] = 0
    # Filter out outliers for chr2 (columns)
    ind_col = np.arange(len(col_orig))[col_orig>threshold_col]
    df_transformed.iloc[:,ind_col] = 0
    
    return(df_transformed)
    

def map_rownum2pos(df, row_num):
    '''
    Helper function for plot_dense_hic_dataframe, for a given row number 
    returns the index which is equal to the genome location of that locus on chr1
    Args:
        df: (pandas DataFrame) dense HiD dataframe
        row_num: (int) row number
    Returns:
        An int equal to the genome location of the locus of interest on chr1
    '''
    positions = df.index.values
    return positions[row_num]


def map_colnum2pos(df, col_num):
    '''
    Helper function for plot_dense_hic_dataframe, for a given column number 
    returns the index which is equal to the genome location of that locus on chr2
    Args:
        df: (pandas DataFrame) dense HiD dataframe
        col_num: (int) column number
    Returns:
        An int equal to the genome location of the locus of interest on chr2
    '''
    positions = df.columns.values
    return float(positions[col_num])


def plot_dense_hic_dataframe(df, chr1, chr2, plotname, hic_plots_dir):
    '''
    Plot a dense HiC dataframe as a heatmap
    Args:
        df: (pandas DataFrame) dense HiC dataframe
        chr1: (int) chromosome 1 (rows)
        chr2: (int) chromosome 2 (columns)
        plotname: (str) name of plot
        hic_plots_dir: (str) directory where to save HiC plots
    Returns:
        void
    '''
    # Plot heatmap with colorbar
    data = df.as_matrix()
    plt.figure()
    plt.imshow(data, cmap = 'Reds')
    cbar = plt.colorbar()
    cbar.set_label('balanced contacts')
    cbar.solids.set_rasterized(True)

    # label ticks with genomic position (Mb)
    xaxis = range(0, df.shape[1], 100)
    xlabels = [str(map_colnum2pos(df, x)/1000000.0) for x in xaxis]
    plt.xticks(xaxis, xlabels)
    yaxis = range(0, df.shape[0], 100)
    ylabels = [str(map_rownum2pos(df, y)/1000000.0) for y in yaxis]
    plt.yticks(yaxis, ylabels)
    
    # Add axis labels
    plt.xlabel('chr' + str(chr2) + ' (Mb)', fontsize = 14)
    plt.ylabel('chr' + str(chr1) + ' (Mb)', fontsize = 14)

    # Save and close
    plt.savefig(hic_plots_dir+plotname, format='eps')
    plt.close()


def whole_genome_mean_std(nonzero_entries):
    '''
    Computes mean and standard deviation of nonzero_entries
    Args:
        nonzero_entries: (Numpy array) array f nonzero entries from all Hi-C matrices
    Returns:
        The mean and standard deviation of the input array
    '''
    mean = np.mean(nonzero_entries)
    std = np.std(nonzero_entries)
    return mean, std


def z_score_hic_matrix(mean, std, chr_pairs, processed_hic_data_dir):
    '''
    Z-scores all Hi-C matrices for pairs of chromosomes in chr_pairs and pickles the result
    Args:
        mean: (float) mean for centering
        std: (float) standard deviation for scaling
        chr_pairs: (list) list of chromosome pairs
        processed_hic_data_dir: (string) directory of processed Hi-C data
    Returns:
    '''
    for pair in chr_pairs:
        chr1, chr2 = pair
        # Read in
        hic_filename = processed_hic_data_dir+'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter3'+'.pkl'
        df = pickle.load(open(hic_filename, 'rb'))
        # z-score matrix
        df  = (df - mean)/std
        # save new matrix
        df.to_csv(processed_hic_data_dir+'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_zscore.txt')


def output_blacklist_locations(chr_pairs, processed_hic_data_dir):
    '''
    Output a file with locations that have been removed (these locations will have an observed value of 0) 
    for each chromosome
    Args:
        chr_pairs: (list) list of chromosome pairs
        processed_hic_data_dir: (string) directory of processed Hi-C data
    Returns:
    '''
    # initialize empty dictionary
    blacklist = defaultdict(list)
    for pair in chr_pairs:
        chr1, chr2 = pair
        # read in
        hic_filename = processed_hic_data_dir+'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter3'+'.pkl'
        df = pickle.load(open(hic_filename, 'rb'))
        # find out which loci of chr2 (columns) have all zeros in them
        zero_cols = df.columns[(df == 0).all(axis=0)]
        blacklist[chr2].append(zero_cols)
        # find out which loci of chr1 (rows) have all zeros in them
        zero_rows = df.index[(df == 0).all(axis=1)]
        blacklist[chr1].append(zero_rows)
    # process the list
    for chrom in blacklist.keys():
        values_list = blacklist[chrom]
        blacklist[chrom] = set(map(int, list(itertools.chain.from_iterable(values_list))))
    # pickle this dictionary
    f = open(processed_hic_data_dir+'blacklist.pickle', 'wb')
    pickle.dump(blacklist, f)


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
    raw_hic_dir = config['HIC_DIR']
    genome_dir = config['GENOME_DIR']
    hic_plots_dir = config['HIC_PLOTS_DIR']
    processed_hic_data_dir = config['PROCESSED_HIC_DATA_DIR']
    hic_celltype = config['HIC_CELLTYPE']
    resol_str = config['HIC_RESOLUTION_STR']
    resol = config['HIC_RESOLUTION']
    quality = config['HIC_QUALITY']
    chr_list = config['chrs']
    filter_size = config['CENTROMERE_FILTER_SIZE']
    
    # We first find repeat locations to filter out (used later)
    print('Load locations of repeats to filter out later...')
    df_repeats = load_repeats_data(genome_dir)
    df_sizes = get_chrom_sizes(genome_dir,resol)
    dic_repeats_tofilter = find_repeat_locations(df_repeats, chr_list, df_sizes, resol)
    
    # We also load centromere information (used later)
    print('Load centromere information to be used later...')
    df_centrom = get_centromere_locations(genome_dir)
    
    # Record nonzero values over all Hi-C matrices
    nonzero_entries = []
    
    # List all chromosome pairs and process interchromosomal HiC data for all pairs
    chr_pairs = list(itertools.combinations(chr_list, 2))
    print("Process HiC data for all "+str(len(chr_pairs))+" chromosome pairs")
    for pair in chr_pairs:
        print('Process HiC data for chromsome pair '+str(pair))
        print('---------------------------------------------------------------')
        chr1, chr2 = pair
        
        # Load raw HiC data
        print('Load raw HiC data')
        raw_hic_data = get_raw_hic_sparse(raw_hic_dir, hic_celltype, resol_str, resol, quality, chr1, chr2)
        
        # Normalize HiC data
        print('KR-normalize HiC data ')
        normalized_hic_data = normalize_raw_hic_sparse(raw_hic_data)
        # Get chromosome sizes (in number of loci)
        chr1_size = int(df_sizes[df_sizes['chr']==str(chr1)]['size_loci'])
        chr2_size = int(df_sizes[df_sizes['chr']==str(chr2)]['size_loci'])
        # Construct normalized dense HiC dataframe
        df = get_dense_hic_dataframe(normalized_hic_data, chr1_size, chr2_size, resol)
        # Plot normalized dense HiC dataframe
        plotname = 'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter0'+'.eps'
        plot_dense_hic_dataframe(df, chr1, chr2, plotname, hic_plots_dir)
        
        # Filter out centromeric and pericentromeric regions
        print('Filter out centromeres ')
        df = filter_centromeres(df, chr1, 'row', df_centrom, filter_size, resol)
        df = filter_centromeres(df, chr2, 'col', df_centrom, filter_size, resol)
        # Plot HiC data after filtering out centromeres and repeats
        plotname = 'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter1'+'.eps'
        plot_dense_hic_dataframe(df, chr1, chr2, plotname, hic_plots_dir)
        
        # Filter repeats for chr1 and chr2
        print('Filter out repeats for chromsome pair '+str(pair))
        df = filter_repeats(df, chr1, dic_repeats_tofilter, 'row')
        df = filter_repeats(df, chr2, dic_repeats_tofilter, 'col')
        # Plot HiC data after filtering out centromeres
        plotname = 'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter2'+'.eps'
        plot_dense_hic_dataframe(df, chr1, chr2, plotname, hic_plots_dir)
        
        # Log-transform dataframe
        print('Log-transform HiC data')
        df_transformed = log_transform(df)
        
        # Filter out outliers
        print('Filter out outliers')
        df_transformed = filter_outliers(df_transformed)

        # Plot and save to pickle HiC data after filtering out centromeres, repeats and outliers
        plotname = 'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter3'+'.eps'
        plot_dense_hic_dataframe(df_transformed, chr1, chr2, plotname, hic_plots_dir)
        picklename = processed_hic_data_dir+'hic_'+'chr'+str(chr1)+'_'+'chr'+str(chr2)+'_norm1_filter3'+'.pkl'
        pickle.dump(df_transformed, open(picklename, 'wb'))
        
        # Record nonzero entries
        data = df_transformed.as_matrix()
        data_nonzero = data[np.nonzero(data)]
        nonzero_entries.append(data_nonzero)

    # Center and scale every Hi-C matrix by the mean and standard deviation across all matrices
    print('Z-score Hi-C matrices using nonzero mean and standard deviation')
    nonzero_entries = np.asarray(list(itertools.chain.from_iterable(nonzero_entries)))
    np.savetxt(processed_hic_data_dir + 'whole_genome_nonzero.logtrans.txt', nonzero_entries)
    mean, std = whole_genome_mean_std(nonzero_entries)
    z_score_hic_matrix(mean, std, chr_pairs, processed_hic_data_dir)
    
    # Output a file with locations that have been removed
    print('Output blacklisted loci file')
    output_blacklist_locations(chr_pairs, processed_hic_data_dir)
    

if __name__ == "__main__":
    main()
    




